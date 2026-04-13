/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "prepareCustomMask.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __host__ inline int32_t ceilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

// Input: customMaskInput (generalPackedCustoMaskPtr) shape: [batch_size, seqLenQ, ceilDiv(seqLenKv-firstSparse, 32)]
// Output: customMaskInput shape:[batch_size,numTilesQ, numTilesKv, numInstsQ, numInstsKv, tileSizeQ, tileSizeKv]
// Output: customMaskOffsets shape:[batch_size]
// Output: firstSparseMaskOffsetsKv shape:[batch_size]
__global__ void prepareCustomMaskBuffersKernelForKeepsMmaAb(
    TllmGenFmhaRunnerParams runnerParams, TllmGenFmhaKernelMetaInfo kernelMeta)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t const tileSizeQRaw = kernelMeta.mTileSizeQ;
    int32_t const tileSizeKv = kernelMeta.mTileSizeKv;
    int32_t const numInstsQ = kernelMeta.mStepQ / tileSizeQRaw;
    int32_t const numInstsKv = kernelMeta.mStepKv / tileSizeKv;
    int32_t const tileSizeQPerCta = kernelMeta.mStepQ;
    int32_t const tileSizeKvPerCta = kernelMeta.mStepKv;
    // Pad tileSizeQ to 32 for uint32 packing (Mask.h:1494, Fmha.cpp:239)
    int32_t const tileSizeQ = ((tileSizeQRaw + 31) / 32) * 32;

    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int64_t* customMaskOffsetsPtr = runnerParams.customMaskOffsetsPtr;
    uint32_t* customMaskPtr = runnerParams.customMaskPtr;
    int32_t const* customMaskInputPtr = runnerParams.generalPackedCustoMaskPtr;
    int32_t* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const qThreadIdx = static_cast<int32_t>(threadIdx.x);
    int32_t const qGroupIdx = static_cast<int32_t>(blockIdx.y);
    int32_t const kvThreadIdx = static_cast<int32_t>(threadIdx.y);
    int32_t const kvGroupIdx = static_cast<int32_t>(blockIdx.z);

    if (batchIdx >= batchSize)
    {
        return;
    }
    // The first sparseMask offset in the Kv sequence dimension.
    int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[batchIdx];
    int32_t const firstSparseMaskTileOffsetKv = firstSparseMaskOffsetKv / tileSizeKvPerCta;
    int32_t const adjustedFirstSparseMaskOffsetKv = firstSparseMaskTileOffsetKv * tileSizeKvPerCta;

    // The sequence length of tensor Q.
    int32_t const seqLenQ = runnerParams.seqlensQPtr[batchIdx];
    // The sequence length of tensor KV.
    int32_t const seqLenKv = seqLensKvPtr[batchIdx];

    // Packed mask row width in int32 words.  mPackedMaskMaxSeqLenQ is the
    // batch-wide max generation length; when > 0 use it for row width so
    // every request reads the correct number of bits per mask row.
    int32_t const packedMaskMaxSeqLenQ
        = runnerParams.mPackedMaskMaxSeqLenQ > 0 ? runnerParams.mPackedMaskMaxSeqLenQ : seqLenQ;
    int32_t const packedMaskNumBlocks = ceilDiv(packedMaskMaxSeqLenQ, 32);
    // Cumulative Q sequence lengths for packed mask batch indexing.
    // When available, mask is in packed layout (like Hopper XQA):
    //   mask row i of request b starts at (cumSeqLensQ[b] + i) * packedMaskNumBlocks
    // When null, mask is in padded 3D layout:
    //   mask row i of request b starts at (b * packedMaskMaxSeqLenQ + i) * packedMaskNumBlocks
    int32_t const* cumSeqLensQPtr = runnerParams.cumSeqLensQPtr;

    // Calculate global Q token index (flattened across heads)
    int32_t const qTokensPerBlock = static_cast<int32_t>(blockDim.x);
    int32_t const flattenedQIdx = qGroupIdx * qTokensPerBlock + qThreadIdx;
    int32_t const totalQTokens = seqLenQ * numHeadsQPerKv;

    if (flattenedQIdx >= totalQTokens)
    {
        return;
    }

    int32_t const tokenIdxQ = flattenedQIdx / numHeadsQPerKv;
    int32_t const headIdxInGrp = flattenedQIdx % numHeadsQPerKv;

    // Iterate from adjustedFirstSparseMaskOffsetKv to seqLenKv
    int32_t const kvTokensPerBlock = static_cast<int32_t>(blockDim.y);
    int32_t const globalKvIdx = kvGroupIdx * kvTokensPerBlock + kvThreadIdx;
    int32_t const tokenIdxKv = adjustedFirstSparseMaskOffsetKv + globalKvIdx;

    // Check KV bounds
    if (tokenIdxKv >= seqLenKv)
    {
        return;
    }

    // Get the mask value for this (Q, KV) pair
    int32_t randomMask = 0;
    if (tokenIdxKv < firstSparseMaskOffsetKv)
    {
        // Dense region: always attend
        randomMask = 1;
    }
    else
    {
        // Sparse region: check the input mask
        // The KV dimension in the mask corresponds to Q positions (tree mask)
        int32_t const qPosInTree = tokenIdxKv - firstSparseMaskOffsetKv;
        if (qPosInTree < seqLenQ)
        {
            // Packed layout (cumSeqLensQPtr != null): row offset = cumSeqLensQ[b] + tokenIdxQ
            // Padded 3D layout (cumSeqLensQPtr == null): row offset = b * packedMaskMaxSeqLenQ + tokenIdxQ
            int32_t const rowOffset = cumSeqLensQPtr != nullptr ? (cumSeqLensQPtr[batchIdx] + tokenIdxQ)
                                                                : (batchIdx * packedMaskMaxSeqLenQ + tokenIdxQ);
            int32_t const qMaskBaseIdx = rowOffset * packedMaskNumBlocks;
            int32_t const packedMaskIdx = qMaskBaseIdx + (qPosInTree >> 5);
            int32_t const bitPos = qPosInTree & 0x1F;
            randomMask = (customMaskInputPtr[packedMaskIdx] >> bitPos) & 1;
        }
    }

    if (randomMask)
    {
        int32_t const numCustomMaskTilesKv = ceilDiv(seqLenKv, tileSizeKvPerCta) - firstSparseMaskTileOffsetKv;
        int64_t const customMaskOffset = customMaskOffsetsPtr[batchIdx];
        uint32_t* localCustomMaskPtr = customMaskPtr + customMaskOffset;

        // Calculate Q indices in the custom mask
        int32_t const customMaskTokenIdxQ = tokenIdxQ * numHeadsQPerKv + headIdxInGrp;
        int32_t const tileIdxQ = customMaskTokenIdxQ / tileSizeQPerCta;
        int32_t const instIdxQ = (customMaskTokenIdxQ % tileSizeQPerCta) / tileSizeQ;
        int32_t const tokenIdxInTileQ = (customMaskTokenIdxQ % tileSizeQPerCta) % tileSizeQ;

        // Calculate KV indices in the custom mask
        int32_t const customMaskTokenIdxKv = tokenIdxKv - adjustedFirstSparseMaskOffsetKv;
        int32_t const tileIdxKv = customMaskTokenIdxKv / tileSizeKvPerCta;
        int32_t const instIdxKv = (customMaskTokenIdxKv % tileSizeKvPerCta) / tileSizeKv;
        int32_t const tokenIdxInTileKv = (customMaskTokenIdxKv % tileSizeKvPerCta) % tileSizeKv;

        // Calculate final mask offset
        int64_t const tileBase = static_cast<int64_t>(tileIdxQ) * numCustomMaskTilesKv;
        int64_t const tileOffset = tileBase + tileIdxKv;
        int64_t const instOffset = tileOffset * numInstsQ * numInstsKv + (instIdxQ * numInstsKv + instIdxKv);
        int64_t const maskOffset
            = instOffset * tileSizeQ * tileSizeKv + (tokenIdxInTileQ * tileSizeKv + tokenIdxInTileKv);
        // The offset of uint32_t custom mask
        int64_t const offsetAsUInt32 = maskOffset >> 5;
        int32_t const bitPosInUInt32 = maskOffset & 0x1F;
        // Set the bit in uint32_t custom mask
        atomicOr(&localCustomMaskPtr[offsetAsUInt32], (1U << bitPosInUInt32));
    }
}

__global__ void computeCustomMaskOffsetsKernel(
    TllmGenFmhaKernelMetaInfo kernelMeta, TllmGenFmhaRunnerParams runnerParams, unsigned long long* globalCounter)
{
    int32_t batchSize = runnerParams.mBatchSize;
    int32_t numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t tileSizeQPerCta = kernelMeta.mStepQ;
    int32_t tileSizeKvPerCta = kernelMeta.mStepKv;
    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int32_t const* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    typedef cub::BlockScan<int64_t, 128> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t maskSize = 0;

    if (idx < batchSize)
    {

        int32_t seqLenQ = runnerParams.seqlensQPtr[idx];
        int32_t seqLenKv = seqLensKvPtr[idx];
        int32_t firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];

        int32_t numTilesQ = (seqLenQ * numHeadsQPerKv + tileSizeQPerCta - 1) / tileSizeQPerCta;
        int32_t firstSparseTile = firstSparseMaskOffsetKv / tileSizeKvPerCta;
        int32_t numCustomMaskTilesKv = (seqLenKv + tileSizeKvPerCta - 1) / tileSizeKvPerCta - firstSparseTile;

        maskSize = static_cast<int64_t>(numTilesQ * numCustomMaskTilesKv * kernelMeta.mStepQ * kernelMeta.mStepKv / 32);
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
}

void launchComputeCustomMaskOffsetsKernel(
    TllmGenFmhaKernelMetaInfo const& kernelMeta, TllmGenFmhaRunnerParams const& runnerParams, cudaStream_t stream)
{

    int32_t batchSize = runnerParams.mBatchSize;

    unsigned long long* d_globalCounter;
    cudaMallocAsync(&d_globalCounter, sizeof(unsigned long long), stream);
    cudaMemsetAsync(d_globalCounter, 0, sizeof(unsigned long long), stream);

    int blockSize = 128;
    int gridSize = (batchSize + blockSize - 1) / blockSize;
    computeCustomMaskOffsetsKernel<<<gridSize, blockSize, 0, stream>>>(kernelMeta, runnerParams, d_globalCounter);

    cudaFreeAsync(d_globalCounter, stream);
}

// Post-processing kernel to write adjusted firstSparseMaskOffsetsKv after all work is done
__global__ void adjustFirstSparseMaskOffsetsKernel(
    TllmGenFmhaRunnerParams runnerParams, TllmGenFmhaKernelMetaInfo kernelMeta)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const tileSizeKvPerCta = kernelMeta.mStepKv;
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize)
        return;
    int32_t* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;
    int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];
    // It needs to be adjusted to multiple of tileSizeKvPerCta
    int32_t const adjusted = (firstSparseMaskOffsetKv / tileSizeKvPerCta) * tileSizeKvPerCta;
    firstSparseMaskOffsetsKvPtr[idx] = adjusted;
}

void launchPrepareCustomMaskBuffersKernelForKeepsMmaAb(
    TllmGenFmhaRunnerParams const& runnerParams, TllmGenFmhaKernelMetaInfo const& kernelMeta, cudaStream_t stream)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const maxSeqLenQ = runnerParams.mMaxSeqLenQ;
    int32_t const numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t const tileSizeKvPerCta = kernelMeta.mStepKv;

    // Total Q tokens (flattened across heads)
    int32_t const maxTotalQTokens = maxSeqLenQ * numHeadsQPerKv;

    // Calculate the maximum KV range to process
    // The actual range is [adjustedFirstSparseMaskOffsetKv, seqLenKv)
    // adjustedFirstSparseMaskOffsetKv <= firstSparseMaskOffsetKv = seqLenKv - seqLenQ
    // So the maximum range length is: seqLenKv - adjustedFirstSparseMaskOffsetKv <= maxSeqLenQ + (tileSizeKvPerCta - 1)
    int32_t const maxKvRangeLength = maxSeqLenQ + (tileSizeKvPerCta - 1);

    int32_t const qTokensPerBlock = 64;
    int32_t const kvTokensPerBlock = 4;

    int32_t const numBlocksY = ceilDiv(maxTotalQTokens, qTokensPerBlock);
    int32_t const numBlocksZ = ceilDiv(maxKvRangeLength, kvTokensPerBlock);

    dim3 gridDim(batchSize, numBlocksY, numBlocksZ);
    dim3 blockDim(qTokensPerBlock, kvTokensPerBlock, 1);

    prepareCustomMaskBuffersKernelForKeepsMmaAb<<<gridDim, blockDim, 0, stream>>>(runnerParams, kernelMeta);
    // Ensure adjusted firstSparse offsets are written only after all blocks finish
    {
        int const blockSize = 128;
        int const gridSize = (batchSize + blockSize - 1) / blockSize;
        adjustFirstSparseMaskOffsetsKernel<<<gridSize, blockSize, 0, stream>>>(runnerParams, kernelMeta);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// SwapsMmaAb + Custom mask with groupsTokensHeadsQ=false:
// - Each CTA handles 1 token × tileSizeQ heads
// - Mask layout per tile: [numInstsQ][numInstsKv][tileSizeKv][tileSizeQPadded / 32] uint32s
// - tileSizeQPadded = roundUp(tileSizeQ, 32), e.g., Q8 → 32
// - All heads of the same token share the same mask value (tree mask is per-token)
__global__ void prepareCustomMaskBuffersKernelForSwapsMmaAb(
    TllmGenFmhaRunnerParams runnerParams, TllmGenFmhaKernelMetaInfo kernelMeta)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t const tileSizeKv = kernelMeta.mTileSizeKv;
    int32_t const tileSizeQRaw = kernelMeta.mTileSizeQ; // Raw hardware tile size (8 for Q8)
    int32_t const numInstsQ = kernelMeta.mStepQ / tileSizeQRaw;
    int32_t const numInstsKv = kernelMeta.mStepKv / kernelMeta.mTileSizeKv;
    int32_t const tileSizeKvPerCta = kernelMeta.mStepKv;
    // Pad tileSizeQ to 32 for uint32 packing (must match Fmha.cpp:147 and Mask.h:1494)
    int32_t const tileSizeQ = ((tileSizeQRaw + 31) / 32) * 32;
    // tileSizeQPerCta uses padded tileSizeQ (Fmha.cpp:151)
    int32_t const tileSizeQPerCta = tileSizeQ * numInstsQ;
    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int64_t* customMaskOffsetsPtr = runnerParams.customMaskOffsetsPtr;
    uint32_t* customMaskPtr = runnerParams.customMaskPtr;
    int32_t const* customMaskInputPtr = runnerParams.generalPackedCustoMaskPtr;
    int32_t* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x);
    // threadIdx.x = token index within block, blockIdx.y = token group
    int32_t const tokenThreadIdx = static_cast<int32_t>(threadIdx.x);
    int32_t const tokenGroupIdx = static_cast<int32_t>(blockIdx.y);
    // threadIdx.y = kv index within block, blockIdx.z = kv group
    int32_t const kvThreadIdx = static_cast<int32_t>(threadIdx.y);
    int32_t const kvGroupIdx = static_cast<int32_t>(blockIdx.z);

    if (batchIdx >= batchSize)
    {
        return;
    }

    // First sparse mask offset
    int32_t const firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[batchIdx];
    int32_t const firstSparseMaskTileOffsetKv = firstSparseMaskOffsetKv / tileSizeKvPerCta;
    int32_t const adjustedFirstSparseMaskOffsetKv = firstSparseMaskTileOffsetKv * tileSizeKvPerCta;

    // Sequence lengths
    int32_t const seqLenQ = runnerParams.seqlensQPtr[batchIdx];
    int32_t const seqLenKv = seqLensKvPtr[batchIdx];

    // Packed mask input dimensions
    int32_t const packedMaskMaxSeqLenQ
        = runnerParams.mPackedMaskMaxSeqLenQ > 0 ? runnerParams.mPackedMaskMaxSeqLenQ : seqLenQ;
    int32_t const packedMaskNumBlocks = ceilDiv(packedMaskMaxSeqLenQ, 32);
    int32_t const* cumSeqLensQPtr = runnerParams.cumSeqLensQPtr;

    // Token index (this thread's Q token)
    int32_t const tokensPerBlock = static_cast<int32_t>(blockDim.x);
    int32_t const tokenIdxQ = tokenGroupIdx * tokensPerBlock + tokenThreadIdx;
    if (tokenIdxQ >= seqLenQ)
    {
        return;
    }

    // KV index
    int32_t const kvTokensPerBlock = static_cast<int32_t>(blockDim.y);
    int32_t const globalKvIdx = kvGroupIdx * kvTokensPerBlock + kvThreadIdx;
    int32_t const tokenIdxKv = adjustedFirstSparseMaskOffsetKv + globalKvIdx;
    if (tokenIdxKv >= seqLenKv)
    {
        return;
    }

    // Determine mask value for this (tokenQ, tokenKv) pair
    int32_t randomMask = 0;
    if (tokenIdxKv < firstSparseMaskOffsetKv)
    {
        // Dense region: always attend
        randomMask = 1;
    }
    else
    {
        // Sparse region: check the input packed mask
        int32_t const qPosInTree = tokenIdxKv - firstSparseMaskOffsetKv;
        if (qPosInTree < seqLenQ)
        {
            int32_t const rowOffset = cumSeqLensQPtr != nullptr ? (cumSeqLensQPtr[batchIdx] + tokenIdxQ)
                                                                : (batchIdx * packedMaskMaxSeqLenQ + tokenIdxQ);
            int32_t const qMaskBaseIdx = rowOffset * packedMaskNumBlocks;
            int32_t const packedMaskIdx = qMaskBaseIdx + (qPosInTree >> 5);
            int32_t const bitPos = qPosInTree & 0x1F;
            randomMask = (customMaskInputPtr[packedMaskIdx] >> bitPos) & 1;
        }
    }

    if (randomMask)
    {
        int32_t const numCustomMaskTilesKv = ceilDiv(seqLenKv, tileSizeKvPerCta) - firstSparseMaskTileOffsetKv;
        int64_t const customMaskOffset = customMaskOffsetsPtr[batchIdx];
        uint32_t* localCustomMaskPtr = customMaskPtr + customMaskOffset;

        // For groupsTokensHeadsQ=false: tileQ = tokenIdxQ (1 tile per token when numHeadsQPerKv <= tileSizeQ)
        int32_t const numTilesQPerToken = ceilDiv(numHeadsQPerKv, tileSizeQPerCta);

        // KV tile indices
        int32_t const customMaskKvIdx = tokenIdxKv - adjustedFirstSparseMaskOffsetKv;
        int32_t const tileIdxKv = customMaskKvIdx / tileSizeKvPerCta;
        int32_t const instIdxKv = (customMaskKvIdx % tileSizeKvPerCta) / tileSizeKv;
        int32_t const kvInTile = customMaskKvIdx % tileSizeKv;

        // Write mask bits using the exact same indexing as trtllm-gen Fmha.cpp lines 268-322.
        // For SwapsMmaAb: iterate over heads, compute LDTM-permuted bit offset.
        for (int32_t headIdxInGrp = 0; headIdxInGrp < numHeadsQPerKv; ++headIdxInGrp)
        {
            // Fmha.cpp:279: customMaskTokenIdxQ = headIdxInGrp
            int32_t const customMaskTokenIdxQ = headIdxInGrp;
            // Fmha.cpp:282: tileIdxQ = customMaskTokenIdxQ / tileSizeQPerCta
            int32_t tileIdxQ = customMaskTokenIdxQ / tileSizeQPerCta;
            // Fmha.cpp:284: tileIdxQ += tokenIdxQ * ceil(numHeadsQPerKv / tileSizeQPerCta)
            tileIdxQ += tokenIdxQ * numTilesQPerToken;
            // Fmha.cpp:286-287
            int32_t const instIdxQ = (customMaskTokenIdxQ % tileSizeQPerCta) / tileSizeQ;
            int32_t const tokenIdxInTileQ = (customMaskTokenIdxQ % tileSizeQPerCta) % tileSizeQ;

            // Fmha.cpp:288-294: tile/inst offset (same for KeepsMmaAb and SwapsMmaAb)
            int64_t const tileOffset
                = static_cast<int64_t>(tileIdxQ) * numCustomMaskTilesKv + tileIdxKv;
            int64_t const instOffset
                = tileOffset * numInstsQ * numInstsKv + (instIdxQ * numInstsKv + instIdxKv);
            int64_t maskOffset = instOffset * tileSizeQ * tileSizeKv;

            // Fmha.cpp:297-316: SwapsMmaAb LDTM 16dp.256bit permutation
            int32_t const tokenIdxInTileKv = kvInTile;
            int32_t const threadIdxQ = (tokenIdxInTileQ % 8) / 2;
            int32_t const threadIdxKv
                = (tokenIdxInTileKv % 8) + (tokenIdxInTileKv / 32) * 8;
            int32_t const tokenIdxInWarpTileKv = tokenIdxInTileKv % 32;
            int32_t const eltIdxInThread = (tokenIdxInTileQ % 2)
                + ((tokenIdxInWarpTileKv / 8) % 2) * 2
                + (tokenIdxInTileQ / 8) * 4
                + (tokenIdxInWarpTileKv / 16) * 4 * (tileSizeQRaw / 8);
            maskOffset += (threadIdxKv * 4 + threadIdxQ) * 32 + eltIdxInThread;

            // Fmha.cpp:318-322: set bit in uint32
            int64_t const offsetAsUInt32 = maskOffset / 32;
            int32_t const bitPosInUInt32 = maskOffset % 32;
            atomicOr(&localCustomMaskPtr[offsetAsUInt32], (1U << bitPosInUInt32));
        }
    }
}

void launchPrepareCustomMaskBuffersKernelForSwapsMmaAb(
    TllmGenFmhaRunnerParams const& runnerParams, TllmGenFmhaKernelMetaInfo const& kernelMeta, cudaStream_t stream)
{
    int32_t const batchSize = runnerParams.mBatchSize;
    int32_t const maxSeqLenQ = runnerParams.mMaxSeqLenQ;
    int32_t const tileSizeKvPerCta = kernelMeta.mStepKv;

    // Max KV range: same as KeepsMmaAb
    int32_t const maxKvRangeLength = maxSeqLenQ + (tileSizeKvPerCta - 1);

    // Thread block: tokens along X, KV along Y
    int32_t const tokensPerBlock = 64;
    int32_t const kvTokensPerBlock = 4;

    int32_t const numBlocksY = ceilDiv(maxSeqLenQ, tokensPerBlock);
    int32_t const numBlocksZ = ceilDiv(maxKvRangeLength, kvTokensPerBlock);

    dim3 gridDim(batchSize, numBlocksY, numBlocksZ);
    dim3 blockDim(tokensPerBlock, kvTokensPerBlock, 1);

    prepareCustomMaskBuffersKernelForSwapsMmaAb<<<gridDim, blockDim, 0, stream>>>(runnerParams, kernelMeta);
    // Adjust firstSparse offsets
    {
        int const blockSize = 128;
        int const gridSize = (batchSize + blockSize - 1) / blockSize;
        adjustFirstSparseMaskOffsetsKernel<<<gridSize, blockSize, 0, stream>>>(runnerParams, kernelMeta);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Offset computation for SwapsMmaAb (different tile size formula)
__global__ void computeCustomMaskOffsetsKernelForSwapsMmaAb(
    TllmGenFmhaKernelMetaInfo kernelMeta, TllmGenFmhaRunnerParams runnerParams, unsigned long long* globalCounter)
{
    int32_t batchSize = runnerParams.mBatchSize;
    int32_t numHeadsQPerKv = runnerParams.mNumHeadsQPerKv;
    int32_t tileSizeQRaw = kernelMeta.mTileSizeQ;
    int32_t tileSizeKv = kernelMeta.mTileSizeKv;
    int32_t numInstsQ = kernelMeta.mStepQ / tileSizeQRaw;
    int32_t numInstsKv = kernelMeta.mStepKv / tileSizeKv;
    int32_t tileSizeKvPerCta = kernelMeta.mStepKv;
    // Pad tileSizeQ to 32 for uint32 packing (must match Mask.h:1494)
    int32_t tileSizeQ = ((tileSizeQRaw + 31) / 32) * 32;
    // tileSizeQPerCta must use padded tileSizeQ (Fmha.cpp:151)
    int32_t tileSizeQPerCta = tileSizeQ * numInstsQ;

    int32_t const* seqLensKvPtr = runnerParams.seqLensKvPtr;
    int32_t const* firstSparseMaskOffsetsKvPtr = runnerParams.firstSparseMaskOffsetsKvPtr;

    typedef cub::BlockScan<int64_t, 128> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t maskSize = 0;

    if (idx < batchSize)
    {
        int32_t seqLenQ = runnerParams.seqlensQPtr[idx];
        int32_t seqLenKv = seqLensKvPtr[idx];
        int32_t firstSparseMaskOffsetKv = firstSparseMaskOffsetsKvPtr[idx];

        // For groupsTokensHeadsQ=false: 1 token per CTA, heads within tile
        int32_t numTilesQPerToken = ceilDiv(numHeadsQPerKv, tileSizeQPerCta);
        int32_t numTilesQ = seqLenQ * numTilesQPerToken;

        int32_t firstSparseTile = firstSparseMaskOffsetKv / tileSizeKvPerCta;
        int32_t numCustomMaskTilesKv = ceilDiv(seqLenKv, tileSizeKvPerCta) - firstSparseTile;

        // Per-tile size in uint32: numInstsQ * numInstsKv * (tileSizeQ * tileSizeKv) / 32
        // This matches trtllm-gen Fmha.cpp line 239-240: bit-level flat layout packed into uint32.
        int32_t perTileSize = numInstsQ * numInstsKv * (tileSizeQ * tileSizeKv) / 32;
        maskSize = static_cast<int64_t>(numTilesQ) * numCustomMaskTilesKv * perTileSize;
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
}

void launchComputeCustomMaskOffsetsKernelForSwapsMmaAb(
    TllmGenFmhaKernelMetaInfo const& kernelMeta, TllmGenFmhaRunnerParams const& runnerParams, cudaStream_t stream)
{
    int32_t batchSize = runnerParams.mBatchSize;

    unsigned long long* d_globalCounter;
    cudaMallocAsync(&d_globalCounter, sizeof(unsigned long long), stream);
    cudaMemsetAsync(d_globalCounter, 0, sizeof(unsigned long long), stream);

    int blockSize = 128;
    int gridSize = (batchSize + blockSize - 1) / blockSize;
    computeCustomMaskOffsetsKernelForSwapsMmaAb<<<gridSize, blockSize, 0, stream>>>(
        kernelMeta, runnerParams, d_globalCounter);

    cudaFreeAsync(d_globalCounter, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void runPrepareCustomMask(
    TllmGenFmhaKernelMetaInfo const& kernelMeta, TllmGenFmhaRunnerParams const& runnerParams, cudaStream_t stream)
{
    auto const kernelType = static_cast<FmhaKernelType>(kernelMeta.mKernelType);

    if (isKeepsMmaAbForGenerationKernel(kernelType))
    {
        int cta_tile_size = kernelMeta.mStepQ * kernelMeta.mStepKv;
        if (cta_tile_size > 128 * 128 * 2)
        {
            TLLM_LOG_ERROR(
                "TRTLLM-GEN needs larger buffer for custom mask preparation please enlarge it according to the "
                "formula: tile_size_q * tile_size_k * num_instances_q * num_instances_k");
            return;
        }
        // Step 1: Compute offsets on GPU using prefix sum
        launchComputeCustomMaskOffsetsKernel(kernelMeta, runnerParams, stream);
        // Step 2: Compute custom mask buffers
        launchPrepareCustomMaskBuffersKernelForKeepsMmaAb(runnerParams, kernelMeta, stream);
        TLLM_CUDA_CHECK(cudaGetLastError());
    }
    else if (isSwapsMmaAbForGenerationKernel(kernelType))
    {
        // SwapsMmaAb + Custom mask with groupsTokensHeadsQ=false.
        // Step 1: Compute offsets (different tile formula from KeepsMmaAb)
        launchComputeCustomMaskOffsetsKernelForSwapsMmaAb(kernelMeta, runnerParams, stream);
        // Step 2: Compute custom mask buffers in SwapsMmaAb layout
        launchPrepareCustomMaskBuffersKernelForSwapsMmaAb(runnerParams, kernelMeta, stream);
        TLLM_CUDA_CHECK(cudaGetLastError());
    }
    else
    {
        TLLM_LOG_ERROR(
            "TRTLLM-GEN does not support kernel type: %d for custom mask preparation", runnerParams.mKernelType);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels

TRTLLM_NAMESPACE_END
