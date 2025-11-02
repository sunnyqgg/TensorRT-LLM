/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fmhaRunnerParams.h"
#include "prepareCustomMask.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __host__ inline int32_t ceilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

__global__ void prepareCustomMaskBuffersKernelForKeepsMmaAb(
    TllmGenFmhaRunnerParams runnerParams, TllmGenFmhaKernelMetaInfo kernelMeta)
{
    int32_t batchSize = runnerParams.mBatchSize;
    int32_t numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t tileSizeQ = kernelMeta.mTileSizeQ;
    int32_t tileSizeKv = kernelMeta.mTileSizeKv;
    int32_t numInstsQ = kernelMeta.mStepQ / kernelMeta.mTileSizeQ;
    ;
    int32_t numInstsKv = kernelMeta.mStepKv / kernelMeta.mTileSizeKv;
    // Pad tileSizeKv to multiple of 32 for keepsMmaAb kernel
    int32_t tileSizeKvPadded = ceilDiv(tileSizeKv, 32) * 32;
    int32_t tileSizeQPerCta = tileSizeQ * numInstsQ;
    int32_t tileSizeKvPerCta = tileSizeKvPadded * numInstsKv;
    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    // int32_t const* cumSeqLensQPtr = runnerParams.cumSeqLensQPtr;
    int64_t* customMaskOffsetsPtr = runnerParams.customMaskOffsetsPtr;
    uint32_t* customMaskPtr = runnerParams.customMaskPtr;
    int32_t const* customMaskInputPtr = runnerParams.generalPackedCustoMaskPtr;

    int32_t* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    int32_t batchIdx = blockIdx.x;
    int32_t flattenedThreadIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batchIdx >= batchSize)
        return;

    // int32_t seqLenQ = cumSeqLensQPtr[batchIdx + 1] - cumSeqLensQPtr[batchIdx];
    int32_t seqLenQ = runnerParams.spec_decoding_generation_lengths[batchIdx];
    int32_t seqLenKv = seqLensKvPtr[batchIdx];
    int32_t totalQTokens = seqLenQ * numHeadsQPerKv;

    if (flattenedThreadIdx >= totalQTokens)
        return;

    int32_t tokenIdxQ = flattenedThreadIdx / numHeadsQPerKv;
    int32_t headIdxInGrp = flattenedThreadIdx % numHeadsQPerKv;

    int32_t firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[batchIdx];
    int32_t firstSparseMaskTileOffsetKv = firstSparseMaskOffsetKv / tileSizeKvPerCta;
    int32_t adjustedFirstSparseMaskOffsetKv = firstSparseMaskTileOffsetKv * tileSizeKvPerCta;

    if (flattenedThreadIdx == 0)
    {
        firstSparseMaskOffsetsKvPtr[batchIdx] = adjustedFirstSparseMaskOffsetKv;
    }

    int32_t numCustomMaskTilesKv = ceilDiv(seqLenKv, tileSizeKvPerCta) - firstSparseMaskTileOffsetKv;
    int64_t customMaskOffset = customMaskOffsetsPtr[batchIdx];
    uint32_t* localCustomMaskPtr = customMaskPtr + customMaskOffset;

    int32_t qMaskBaseIdx = (batchIdx * seqLenQ + tokenIdxQ) * ceilDiv(seqLenKv - firstSparseMaskOffsetKv, 32);

    int32_t customMaskTokenIdxQ = tokenIdxQ * numHeadsQPerKv + headIdxInGrp;
    int32_t tileIdxQ = customMaskTokenIdxQ / tileSizeQPerCta;
    int32_t instIdxQ = (customMaskTokenIdxQ % tileSizeQPerCta) / tileSizeQ;
    int32_t tokenIdxInTileQ = (customMaskTokenIdxQ % tileSizeQPerCta) % tileSizeQ;

    for (int32_t tokenIdxKv = adjustedFirstSparseMaskOffsetKv; tokenIdxKv < seqLenKv; ++tokenIdxKv)
    {
        int32_t randomMask;
        if (tokenIdxKv < firstSparseMaskOffsetKv)
        {
            randomMask = 1;
        }
        else
        {
            int32_t packedMaskIdx = qMaskBaseIdx + ((tokenIdxKv - firstSparseMaskOffsetKv) >> 5);
            int32_t bitPos = (tokenIdxKv - firstSparseMaskOffsetKv) & 0x1F;
            randomMask = (customMaskInputPtr[packedMaskIdx] >> bitPos) & 1;
        }

        int32_t customMaskTokenIdxKv = tokenIdxKv - adjustedFirstSparseMaskOffsetKv;
        int32_t tileIdxKv = customMaskTokenIdxKv / tileSizeKvPerCta;
        int32_t instIdxKv = (customMaskTokenIdxKv % tileSizeKvPerCta) / tileSizeKvPadded;
        int32_t tokenIdxInTileKv = (customMaskTokenIdxKv % tileSizeKvPerCta) % tileSizeKvPadded;

        int64_t tileOffset = tileIdxQ * numCustomMaskTilesKv + tileIdxKv;
        int64_t instOffset = tileOffset * numInstsQ * numInstsKv + (instIdxQ * numInstsKv + instIdxKv);
        int64_t maskOffset
            = instOffset * tileSizeQ * tileSizeKvPadded + (tokenIdxInTileQ * tileSizeKvPadded + tokenIdxInTileKv);

        int64_t offsetAsUInt32 = maskOffset >> 5;
        int64_t bitPosInUInt32 = maskOffset & 0x1F;

        atomicOr(&localCustomMaskPtr[offsetAsUInt32], (uint32_t(randomMask) << bitPosInUInt32));
    }
}

__global__ void computeCustomMaskOffsetsKernel(
    TllmGenFmhaKernelMetaInfo kernelMeta, TllmGenFmhaRunnerParams runnerParams, unsigned long long* globalCounter)
{
    int32_t batchSize = runnerParams.mBatchSize;
    int32_t numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t tileSizeQ = kernelMeta.mTileSizeQ;
    int32_t tileSizeKv = kernelMeta.mTileSizeKv;
    int32_t numInstsQ = kernelMeta.mStepQ / kernelMeta.mTileSizeQ;
    ;
    int32_t numInstsKv = kernelMeta.mStepKv / kernelMeta.mTileSizeKv;
    int32_t tileSizeKvPadded = ceilDiv(tileSizeKv, 32) * 32;
    int32_t tileSizeQPerCta = tileSizeQ * numInstsQ;
    int32_t tileSizeKvPerCta = tileSizeKvPadded * numInstsKv;
    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    // int32_t const* cumSeqLensQPtr = runnerParams.cumSeqLensQPtr;
    int32_t const* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    typedef cub::BlockScan<int64_t, 256> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t maskSize = 0;

    if (idx == 0)
    {
        printf(
            "batchSize: %d, seqLensKvPtr: %p, firstSparseMaskOffsetsKvPtr: %p, spec_decoding_generation_lengths: %p, "
            "customMaskOffsetsPtr: %p\n",
            batchSize, seqLensKvPtr, firstSparseMaskOffsetsKvPtr, runnerParams.spec_decoding_generation_lengths,
            runnerParams.customMaskOffsetsPtr);
    }

    if (idx < batchSize)
    {
        // int32_t seqLenQ = cumSeqLensQPtr[idx + 1] - cumSeqLensQPtr[idx];
        int32_t seqLenQ = runnerParams.spec_decoding_generation_lengths[idx];
        int32_t seqLenKv = seqLensKvPtr[idx];
        int32_t firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];
        if (idx == 0)
        {
            printf(
                "seqLenQ: %d, seqLenKv: %d, firstSparseMaskOffsetKv: %d\n", seqLenQ, seqLenKv, firstSparseMaskOffsetKv);
        }

        int32_t numTilesQ = (seqLenQ * numHeadsQPerKv + tileSizeQPerCta - 1) / tileSizeQPerCta;
        int32_t firstSparseTile = firstSparseMaskOffsetKv / tileSizeKvPerCta;
        int32_t numCustomMaskTilesKv = (seqLenKv + tileSizeKvPerCta - 1) / tileSizeKvPerCta - firstSparseTile;

        maskSize = static_cast<int64_t>(
            numTilesQ * numCustomMaskTilesKv * numInstsQ * numInstsKv * (tileSizeQ * tileSizeKvPadded) / 32);
    }

    int64_t prefixOffset;
    int64_t blockSum;
    BlockScan(temp_storage).ExclusiveSum(maskSize, prefixOffset, blockSum);

    __shared__ unsigned long long blockBase;
    if (threadIdx.x == 0)
        blockBase = atomicAdd(globalCounter, (unsigned long long) blockSum);
    __syncthreads();

    if (idx < batchSize)
        runnerParams.customMaskOffsetsPtr[idx] = static_cast<int64_t>(blockBase) + prefixOffset;
    if (idx == 0)
    {
        printf("runnerParams.customMaskOffsetsPtr[idx]: %lld\n", runnerParams.customMaskOffsetsPtr[idx]);
    }
}

void launchComputeCustomMaskOffsetsKernel(
    TllmGenFmhaKernelMetaInfo const& kernelMeta, TllmGenFmhaRunnerParams const& runnerParams, cudaStream_t stream)
{

    int32_t batchSize = runnerParams.mBatchSize;

    unsigned long long* d_globalCounter;
    cudaMallocAsync(&d_globalCounter, sizeof(unsigned long long), stream);
    cudaMemsetAsync(d_globalCounter, 0, sizeof(unsigned long long), stream);

    int blockSize = 256;
    int gridSize = (batchSize + blockSize - 1) / blockSize;
    computeCustomMaskOffsetsKernel<<<gridSize, blockSize, 0, stream>>>(kernelMeta, runnerParams, d_globalCounter);

    cudaFreeAsync(d_globalCounter, stream);
}

void launchPrepareCustomMaskBuffersKernelForKeepsMmaAb(
    TllmGenFmhaRunnerParams const& runnerParams, TllmGenFmhaKernelMetaInfo const& kernelMeta, cudaStream_t stream)
{
    int32_t batchSize = runnerParams.mBatchSize;
    int32_t maxSeqLenQ = runnerParams.mMaxSeqLenQ;
    int32_t numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;

    int32_t maxThreadsPerQ = maxSeqLenQ * numHeadsQPerKv;

    int32_t blockSize = 256;
    int32_t numBlocksY = ceilDiv(maxThreadsPerQ, blockSize);

    dim3 gridDim(batchSize, numBlocksY);
    dim3 blockDim(blockSize);

    prepareCustomMaskBuffersKernelForKeepsMmaAb<<<gridDim, blockDim, 0, stream>>>(runnerParams, kernelMeta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void runPrepareCustomMask(
    TllmGenFmhaKernelMetaInfo const& kernelMeta, TllmGenFmhaRunnerParams const& runnerParams, cudaStream_t stream)
{
    if (isKeepsMmaAbForGenerationKernel(static_cast<FmhaKernelType>(kernelMeta.mKernelType)))
    {
        // Step 1: Compute offsets on GPU using prefix sum
        launchComputeCustomMaskOffsetsKernel(kernelMeta, runnerParams, stream);
        // Step 2: Compute custom mask buffers
        launchPrepareCustomMaskBuffersKernelForKeepsMmaAb(runnerParams, kernelMeta, stream);
        TLLM_CUDA_CHECK(cudaGetLastError());
        printAbsMean(runnerParams.customMaskOffsetsPtr, 1, stream,
            "====gqq runnerParams.customMaskOffsetsPtr at after runPrepareCustomMask ===");
        printAbsMean(runnerParams.firstSparseMaskOffsetsKvPtr, 1, stream,
            "====gqq runnerParams.firstSparseMaskOffsetsKvPtr at after runPrepareCustomMask ===");
        printAbsMean(runnerParams.seqLensKvPtr, 1, stream,
            "====gqq runnerParams.seqLensKvPtr at after runPrepareCustomMask ===");
        printAbsMean(runnerParams.spec_decoding_generation_lengths, 1, stream,
            "====gqq runnerParams.spec_decoding_generation_lengths at after runPrepareCustomMask ===");
        for (int i = 0; i < 10; i++)
        {
            printAbsMean(runnerParams.customMaskPtr + i, 1, stream,
                "====gqq runnerParams.customMaskPtr at after runPrepareCustomMask ===");
        }
        for (int i = 0; i < 10; i++)
        {
            printAbsMean(runnerParams.generalPackedCustoMaskPtr + i, 1, stream,
                "====gqq runnerParams.generalPackedCustoMaskPtr at after runPrepareCustomMask ===");
        }
    }
    else
    {
        TLLM_LOG_ERROR(
            "TRTLLM-GEN does not support kernel type: %d for custom mask preparation", runnerParams.mKernelType);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels
} // namespace tensorrt_llm
