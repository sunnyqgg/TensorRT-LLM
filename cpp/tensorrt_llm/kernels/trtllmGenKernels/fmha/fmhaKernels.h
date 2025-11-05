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

#pragma once

#include "cuda_runtime_api.h"
#include <memory>
#include <mutex>
#include <unordered_map>

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

#include "cubin/kernelMetaInfo.h"
#include "fmhaReduction.h"
#include "fmhaRunnerParams.h"
#include "kernelParams.h"
#include "prepareCustomMask.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr bool isSMCompatible(int gpuSM, int kernelSM)
{
    if (gpuSM == kSM_103)
    {
        return kernelSM == kSM_100f || kernelSM == kSM_103;
    }
    else if (gpuSM == kSM_100)
    {
        return kernelSM == kSM_100f || kernelSM == kSM_100;
    }

    return gpuSM == kernelSM;
}

class TllmGenFmhaKernel
{
public:
    using KernelMeta = TllmGenFmhaKernelMetaInfo;
    using RunnerParams = TllmGenFmhaRunnerParams;
    using SelectKernelParams = TllmGenSelectKernelParams;

    // Ctor.
    TllmGenFmhaKernel(KernelMeta const* pMetaStart, unsigned int nMetaCount, Data_type dtypeQ, Data_type dtypeKv,
        Data_type dtypeOut, unsigned int smArch)
        : mDtypeQ(dtypeQ)
        , mDtypeKv(dtypeKv)
        , mDtypeOut(dtypeOut)
        , mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
        , mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(smArch)
    {
    }

    void loadKernels()
    {
        // Build a lookup map for all kernels.
        for (unsigned int i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            if (isSMCompatible(mSM, kernelMeta.mSM) && kernelMeta.mDataTypeQ == mDtypeQ
                && kernelMeta.mDataTypeKv == mDtypeKv && kernelMeta.mDataTypeO == mDtypeOut)
            {
                // Load CUmodules
                CUmodule hmod{0};
                auto findModuleIter = mModules.find(kernelMeta.mCubin);
                if (findModuleIter != mModules.end())
                {
                    hmod = findModuleIter->second;
                }
                else
                {
                    TLLM_CU_CHECK(mDriver->cuModuleLoadData(&hmod, kernelMeta.mCubin));
                    mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
                }
                // Build a hash map, which maps from kernel meta info to kernel index
                KernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                TLLM_CU_CHECK(mDriver->cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName));
                if (kernelMeta.mSharedMemBytes >= 48 * 1024)
                {
                    TLLM_CU_CHECK(mDriver->cuFuncSetAttribute(funcInfo.mDeviceFunction,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes));
                }
                // Make sure the hashIds are not duplicated.
                TLLM_CHECK_WITH_INFO(mFunctions.find(hashID(kernelMeta)) == mFunctions.end(),
                    "The kernel's hashId has conflicts with others.");
                mFunctions.insert(std::make_pair(hashID(kernelMeta), funcInfo));
            }
        }
    }

    inline uint64_t hashID(int qkvLayout, int maskType, int kernelType, int scheduler, int multiCtasKvMode,
        int headDimPerCtaV, int headDimQk, int headDimV, int tileSizeKv, int numTokensPerPage,
        int maxNumHeadsQPerKvInCta, bool reuseSmemKForV, bool uses2CtaMma) const
    {
        TLLM_CHECK_WITH_INFO((headDimPerCtaV >= 32) && (headDimQk >= 32) && (headDimV >= 32) && (headDimPerCtaV <= 1024)
                && (headDimQk <= 1024) && (headDimV <= 1024) && (numTokensPerPage <= 128),
            "Expect (32 <= headDim <= 1024) && (numTokensPerPage <= 128), got headDimPerCtaV=%d, headDimQk=%d, "
            "headDimV=%d, numTokensPerPage=%d",
            headDimPerCtaV, headDimQk, headDimV, numTokensPerPage);
        TLLM_CHECK_WITH_INFO(maxNumHeadsQPerKvInCta <= 128, "The maxNumHeadsQPerKvInCta <= 128 is required.");
        TLLM_CHECK_WITH_INFO(tileSizeKv == 64 || tileSizeKv == 128, "The tileSizeKv must be 64 or 128.");
        // Format of the hash key:
        // Bit 0  - 3 : qkvLayout.
        // Bit 4  - 7 : maskType.
        // Bit 8  - 11: kernelType.
        // Bit 12 - 15: tileScheduler.
        // Bit 16 - 17: multiCtasKvMode.
        // Bit 18 - 25: (headDimPerCtaV >> 3).
        // Bit 26 - 33: (headDimQk >> 3).
        // Bit 34 - 41: (headDimV >> 3).
        // Bit 42 - 43: (tileSizeKv >> 6).
        // Bit 44 - 48: (numTokensPerPage >> 3).
        // Bit 49 - 56: maxNumHeadsQPerKvInCta.
        // Bit 57 - 57: reuseSmemKForV.
        // Bit 58 - 58: uses2CtaMma.
        return (static_cast<uint64_t>(qkvLayout) << 0) | (static_cast<uint64_t>(maskType) << 4)
            | (static_cast<uint64_t>(kernelType) << 8) | (static_cast<uint64_t>(scheduler) << 12)
            | (static_cast<uint64_t>(multiCtasKvMode) << 16) | (static_cast<uint64_t>(headDimPerCtaV >> 3) << 18)
            | (static_cast<uint64_t>(headDimQk >> 3) << 26) | (static_cast<uint64_t>(headDimV >> 3) << 34)
            | (static_cast<uint64_t>(tileSizeKv >> 6) << 42) | (static_cast<uint64_t>(numTokensPerPage >> 3) << 44)
            | (static_cast<uint64_t>(maxNumHeadsQPerKvInCta) << 49) | (static_cast<uint64_t>(reuseSmemKForV) << 57)
            | (static_cast<uint64_t>(uses2CtaMma) << 58);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(kernelMeta.mQkvLayout, kernelMeta.mMaskType, kernelMeta.mKernelType, kernelMeta.mTileScheduler,
            kernelMeta.mMultiCtasKvMode, kernelMeta.mHeadDimPerCtaV, kernelMeta.mHeadDimQk, kernelMeta.mHeadDimV,
            kernelMeta.mTileSizeKv, kernelMeta.mNumTokensPerPage, kernelMeta.mMaxNumHeadsQPerKvInCta,
            kernelMeta.mReuseSmemKForV, kernelMeta.m2CtaMma);
    }

    std::pair<bool, std::string> checkIfKernelExist(RunnerParams const& params) const
    {
        // Some conditions to check if the kernel is supported.
        // This is meant to avoid occupying unnecessary hashId bits.
        if (params.mHeadDimQk % 8 != 0 || params.mHeadDimV % 8 != 0)
        {
            return std::make_pair(false, "HeadDimQk and HeadDimV must be divisible by 8");
        }
        if (params.mNumTokensPerPage % 8 != 0)
        {
            return std::make_pair(false, "NumTokensPerPage must be divisible by 8");
        }

        // The selectKernelParams that might be updated.
        SelectKernelParams selectKernelParams{params};
        auto [hashId, info] = hashFromRunnerParams(params, selectKernelParams);
        return std::make_pair(mFunctions.find(hashId) != mFunctions.end(), info);
    }

    void run(RunnerParams const& params) const
    {

        // The selectKernelParams that might be updated.
        SelectKernelParams selectKernelParams{params};
        // The iteration index (used to detect a deadlock of selecting new kernels).
        int selectKernelIter = 0;
        // While loop.
        while (true)
        {
            // Any value >= 2 should work here, but we set it larger in case that we might have more complicated
            // heuristic in the future.
            TLLM_CHECK_WITH_INFO(selectKernelIter < 8, "A deadlock is detected when selecting trtllm-gen kernels.");
            auto [hashId, info] = hashFromRunnerParams(params, selectKernelParams);
            auto const findIter = mFunctions.find(hashId);

            // Add debug info when kernels are not found.
            TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "Trtllm-gen kernels not found: " + info);

            auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
            const CUfunction func = findIter->second.mDeviceFunction;

            // Compute the number of CTAs in X, Y and Z dimension and the cluster size in the X dimension.
            auto [maxNumCtasQ, maxNumCtasKv, numCtasX, numCtasY, numCtasZ, clusterDimX]
                = computeNumCtas(params, kernelMeta, selectKernelParams);

            // Need to select a new kernel if mSelectNewKernel is true.
            if (selectKernelParams.mSelectNewKernel)
            {
                selectKernelIter++;
                continue;
            }

            // Prepare the kernel parameters.
            auto kernelParams = KernelParams::setKernelParams(params, kernelMeta, maxNumCtasQ, maxNumCtasKv);
            printf(
                "==== fmha/fmhaKernels.h run(): params.cache_type is %d and params.mHeadDimQk is %d and "
                "params.mHeadDimV is %d and params.mNumHeadsQ is %d and params.mNumHeadsKv is %d and "
                "params.mNumHeadsQPerKv is %d\n",
                params.cache_type, params.mHeadDimQk, params.mHeadDimV, params.mNumHeadsQ, params.mNumHeadsKv,
                params.mNumHeadsQPerKv);
            if (params.layer_idx == 0 && params.is_spec_dec_tree)
            {
                printf("==================runPrepareCustomMask==========and params.is_spec_dec_tree is true========\n");
                runPrepareCustomMask(kernelMeta, params, params.stream);
            }

            // dump Tool
            if (params.layer_idx == 0 && params.is_spec_dec_tree && params.kv_cache_block_array.has_value())
            {
                // 拷贝 KVBlockArray 结构和索引数据到 CPU
                KVBlockArray h_kv = params.kv_cache_block_array.value();
                size_t dataSize = h_kv.mMaxSeqs * h_kv.mMaxBlocksPerSeq * 2;
                int32_t* h_data = new int32_t[dataSize];
                cudaMemcpy(h_data, h_kv.data, dataSize * sizeof(int32_t), cudaMemcpyDeviceToHost);
                h_kv.data = reinterpret_cast<KVCacheIndex*>(h_data);

                // 获取 KV cache 参数
                int32_t const num_kv_heads = params.mNumHeadsKv; // KV head 数量
                int32_t const head_dim = params.mHeadDimQk;      // 每个 head 的维度
                int32_t const batch_idx = 0;                     // batch 索引
                int32_t const total_k_v_tokens = 61;             // 总共要保存 61 个 token

                // 计算每个 token 的数据大小（字节）
                size_t bytes_per_element = 0;
                if (params.cache_type == KvCacheDataType::NVFP4)
                {
                    bytes_per_element = 1; // FP4: 0.5 bytes per element, 但实际按 1 字节对齐
                }
                else if (params.cache_type == KvCacheDataType::FP8 || params.cache_type == KvCacheDataType::INT8)
                {
                    bytes_per_element = 1; // int8_t, FP8
                }
                else
                {
                    bytes_per_element = 2; // half or bf16
                }

                // 每个 token 的总大小 = num_kv_heads × head_dim × bytes_per_element
                size_t token_data_size = num_kv_heads * head_dim * bytes_per_element;

                // 分配临时 GPU buffer 用于存储单个 token 的连续数据
                void* d_temp_k_buffer = nullptr;
                void* d_temp_v_buffer = nullptr;
                cudaMalloc(&d_temp_k_buffer, token_data_size);
                cudaMalloc(&d_temp_v_buffer, token_data_size);

                printf("==== Dumping KV cache for %d tokens ====\n", total_k_v_tokens);
                printf("  num_kv_heads=%d, head_dim=%d, bytes_per_element=%zu\n", num_kv_heads, head_dim,
                    bytes_per_element);
                printf("  token_data_size=%zu bytes\n", token_data_size);
                // 因为 KV cache 在 GPU 内存中是按 [numHeads, tokensPerBlock, hiddenSizePerHead] 布局的，数据不连续。
                //  //需要重新排列成 [numHeads, hiddenSizePerHead] 的连续布局才能保存。
                //  遍历所有要保存的 token
                for (int32_t token_idx = 0; token_idx < total_k_v_tokens; token_idx++)
                {
                    // 获取 K cache 和 V cache 的 block 起始地址，步骤 1: 获取该 token 所在 block 的起始地址
                    auto kBlockPtr = reinterpret_cast<uint8_t*>(h_kv.getKBlockPtr(batch_idx, token_idx));
                    auto vBlockPtr = reinterpret_cast<uint8_t*>(h_kv.getVBlockPtr(batch_idx, token_idx));

                    // 提取该 token 在所有 head 上的数据到临时 buffer
                    // KV cache layout: [numHeads, tokensPerBlock, hiddenSizePerHead]
                    for (int32_t head_idx = 0; head_idx < num_kv_heads; head_idx++)
                    {
                        // 计算该 token 在该 head 中的偏移量
                        int32_t offset_in_block = h_kv.getKVLocalIdx(token_idx, head_idx, head_dim, 0);
                        // offset = head_idx * tokensPerBlock * head_dim + (token_idx % tokensPerBlock) * head_dim + 0

                        // 计算源地址和目标地址,源地址 (k_src)：在 GPU block 中的原始位置（分散的）
                        void* k_src = kBlockPtr + offset_in_block * bytes_per_element;
                        void* k_dst
                            = reinterpret_cast<uint8_t*>(d_temp_k_buffer) + head_idx * head_dim * bytes_per_element;

                        void* v_src = vBlockPtr + offset_in_block * bytes_per_element;
                        void* v_dst
                            = reinterpret_cast<uint8_t*>(d_temp_v_buffer) + head_idx * head_dim * bytes_per_element;

                        // 拷贝该 head 的数据（GPU to GPU）
                        cudaMemcpy(k_dst, k_src, head_dim * bytes_per_element, cudaMemcpyDeviceToDevice);
                        cudaMemcpy(v_dst, v_src, head_dim * bytes_per_element, cudaMemcpyDeviceToDevice);
                    }

                    // 保存 K cache
                    std::string k_path = "./dump_data/key_token_idx_[" + std::to_string(token_idx) + "].bin";
                    DumpTool(k_path.c_str()).write_cuda(d_temp_k_buffer, token_data_size);

                    // 保存 V cache
                    std::string v_path = "./dump_data/v_key_token_idx_[" + std::to_string(token_idx) + "].bin";
                    DumpTool(v_path.c_str()).write_cuda(d_temp_v_buffer, token_data_size);
                }

                printf("==== KV cache dump completed for %d tokens ====\n", total_k_v_tokens);

                // 清理临时 buffer
                cudaFree(d_temp_k_buffer);
                cudaFree(d_temp_v_buffer);
                delete[] h_data;

                // dump q buffer
                std::string q_path = "./dump_data/q_buffer.bin";
                DumpTool(q_path.c_str())
                    .write_cuda(static_cast<void const*>(params.qPtr), 25 * 32 * 128 * sizeof(half));

                // dump custom mask, size = （2 * 128 * 128 / 32）* 4=总字节数目
                std::string custom_mask_path = "./dump_data/custom_mask.bin";
                DumpTool(custom_mask_path.c_str())
                    .write_cuda(static_cast<void const*>(params.customMaskPtr),
                        2 * 128 * 128 / 8); // 2 * 128 * 128 * sizeof(int32_t)/32
            }

            // Prepare kernel parameters list for cuLaunchKernelEx.
            void* kernelParamsList[] = {&kernelParams};
            CUlaunchConfig launch_config;
            launch_config.blockDimX = kernelMeta.mThreadsPerCTA;
            launch_config.blockDimY = 1;
            launch_config.blockDimZ = 1;
            launch_config.gridDimX = numCtasX;
            launch_config.gridDimY = numCtasY;
            launch_config.gridDimZ = numCtasZ;
            launch_config.hStream = params.stream;
            launch_config.sharedMemBytes = kernelMeta.mSharedMemBytes;

            // Debug info.
            TLLM_LOG_DEBUG("TRTLLM-Gen launch info: kernelName = %s", kernelMeta.mFuncName);
            TLLM_LOG_DEBUG(
                "TRTLLM-Gen launch info: maxSeqLenQ = %d, "
                "maxSeqLenKv = %d, "
                "numHeadsQ = %d, "
                "numHeadsKv = %d, batchSize = %d, kernelType = %d",
                params.mMaxSeqLenQ, params.mMaxSeqLenKv, params.mNumHeadsQ, params.mNumHeadsKv, params.mBatchSize,
                static_cast<int>(params.mKernelType));
            TLLM_LOG_DEBUG("TRTLLM-Gen launch info: numCtasX = %d, numCtasY = %d, numCtasZ = %d, clusterDimX = %d",
                numCtasX, numCtasY, numCtasZ, clusterDimX);

            CUlaunchAttribute launch_attribute[3];
            launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            launch_attribute[0].value.clusterDim.x = clusterDimX;
            launch_attribute[0].value.clusterDim.y = 1;
            launch_attribute[0].value.clusterDim.z = 1;
            launch_attribute[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
            launch_attribute[1].value.clusterSchedulingPolicyPreference
                = clusterDimX > 1 ? CU_CLUSTER_SCHEDULING_POLICY_SPREAD : CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;
            launch_attribute[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            launch_attribute[2].value.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();

            launch_config.attrs = launch_attribute;
            launch_config.numAttrs = 3;

            // Add setting for non-portable cluster size.
            if (clusterDimX > 8)
            {
                TLLM_CU_CHECK(mDriver->cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                    1 // Enable non-portable cluster sizes
                    ));
            }

            // Force using GmemReduction for the multiCtasKvMode if the CgaSmemReduction needs more than one wave (due
            // to the cluster occupancy limit).
            // TODO: find a better heuristic of using CgaSmemReduction.
            if (isCgaSmemReduction(selectKernelParams.mMultiCtasKvMode))
            {
                // The maximum number of active clusters that could co-exist.
                int maxActiveClusters = 1;
                TLLM_CU_CHECK(mDriver->cuOccupancyMaxActiveClusters(&maxActiveClusters, func, &launch_config));
                // Use the GmemReduction instead if it needs more than one wave.
                if (maxActiveClusters * clusterDimX < (numCtasX * numCtasY * numCtasZ))
                {
                    selectKernelParams.mForceGmemReduction = true;
                    selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::GmemReduction;
                    // continue to select a new kernel.
                    continue;
                }
            }

            TLLM_CU_CHECK(mDriver->cuLaunchKernelEx(&launch_config, func, kernelParamsList, nullptr));

            // Run the separate reduction kernel if needed.
            runFmhaReduction(kernelMeta, kernelParams, params.mMultiProcessorCount, params.stream);

            // Break the while op.
            break;
        }
    }

private:
    // Is it MLA generation kernel ?
    inline bool isMlaGenKernel(RunnerParams const& params) const
    {
        return params.mHeadDimQk == 576 && params.mHeadDimV == 512;
    }

    // Compute the number of CTAs in X, Y and Z dimension and the cluster size in the X dimension.
    using CtaInfo = std::tuple<int, int, int, int, int, int>;

    CtaInfo computeNumCtas(
        RunnerParams const& params, KernelMeta const& kernelMeta, SelectKernelParams& selectKernelParams) const
    {
        bool isDsv3MinLatencyMode = params.mBatchSize == 1 && params.mMaxSeqLenQ >= 1 && params.mMaxSeqLenQ <= 16
            && params.mHeadDimQk == 576 && params.mHeadDimV == 512;
        // Do we need to select a new kernel ?
        selectKernelParams.mSelectNewKernel = false;

        // The number of Ctas per Q sequence.
        int numCtasPerSeqQ = (params.mMaxSeqLenQ + kernelMeta.mStepQ - 1) / kernelMeta.mStepQ;
        // Each CTA handles one tokenQ by default for spec-decoding generation kernel, which is used to emulate causal
        // masking (like MTP or Eagle3). Note this will be changed later when the high-throughput spec-decoding
        // generation kernels are integrated.
        if (params.mMaxSeqLenQ > 1 && !isContextKernel(params.mKernelType))
        {
            numCtasPerSeqQ = params.mMaxSeqLenQ;
        }

        // Compute the grid dimension Y.
        int numHeadsPerCta
            = kernelMeta.mGroupsHeadsQ ? std::min(params.mNumHeadsQPerKv, kernelMeta.mMaxNumHeadsQPerKvInCta) : 1;
        int numCtasForAllHeadsQ = params.mNumHeadsQ / numHeadsPerCta;
        TLLM_CHECK_WITH_INFO(
            numHeadsPerCta * numCtasForAllHeadsQ == params.mNumHeadsQ, "The numHeadsQ/numHeadsKv is not supported.");
        // Take the number of headDim CTAs.
        TLLM_CHECK_WITH_INFO(
            kernelMeta.mHeadDimV % selectKernelParams.mHeadDimPerCtaV == 0, "The headDimPerCtaV is not supported.");
        int numCtasPerHeadDim = kernelMeta.mHeadDimV / selectKernelParams.mHeadDimPerCtaV;
        // Compute the current numCtasX.
        int numCtasX = numCtasPerSeqQ;
        // Update the numCtasY.
        int numCtasY = numCtasForAllHeadsQ * numCtasPerHeadDim;
        // Compute the grid dimension Z.
        int numCtasZ = params.mBatchSize;
        // The 2CtaMma kernels will use 2 Ctas in the x dimension (only used by MLA generation kernels) for heads,
        // so numCtasPerHeadDim and numCtasForAllHeadsQ will be handled by the 2Ctas in the x dimension.
        if (isMlaGenKernel(params) && selectKernelParams.mUses2CtaMma)
        {
            TLLM_CHECK_WITH_INFO(
                numCtasForAllHeadsQ == 2 && numCtasPerHeadDim == 2, "Internal error: numCtasPerHeadDim should be 2.");
            numCtasX *= 2;
            numCtasY /= (numCtasForAllHeadsQ * numCtasPerHeadDim);
        }

        // First split the seqLenKv into multiple CTAs if the utilization is not full.
        // The number of Ctas per KV sequence.
        int numCtasPerSeqKv = 1;
        // Consider the multiCtasKvMode for better GPU utilization.
        if (isMultiCtasKvEnabled(selectKernelParams.mMultiCtasKvMode))
        {
            // The maximum attention window (the maximum number of tokensKv that will be attended to).
            int maxAttentionWindow{params.mMaxSeqLenKv};
            // Some of the tilesKv will be skipped if the sliding window attention or chunked attention is used.
            if (isSlidingOrChunkedCausalMask(selectKernelParams.mMaskType))
            {
                if (params.mMaxSeqLenKv > params.mAttentionWindowSize)
                {
                    // Consider that the first tileKv might contain tokensKv that is out of the attention window.
                    maxAttentionWindow
                        = std::min(params.mMaxSeqLenKv, params.mAttentionWindowSize + kernelMeta.mTileSizeKv - 1);
                }
                else
                {
                    maxAttentionWindow = std::min(params.mMaxSeqLenKv, params.mChunkedAttentionSize);
                }
            }

            // The maximum number Ctas per Kv sequence, which makes sure that each CtaKv has work to do.
            int const maxNumCtasPerSeqKv = (maxAttentionWindow + kernelMeta.mStepKv - 1) / kernelMeta.mStepKv;
            // Compute numCtasPerSeqKv.
            numCtasPerSeqKv = std::min(maxNumCtasPerSeqKv,
                std::max(1, int32_t(params.mMultiProcessorCount / (numCtasX * numCtasY * numCtasZ))));
            // Update the numCtasX.
            numCtasX *= numCtasPerSeqKv;
            // The current total number of CTAs.
            int totalNumCtas = numCtasX * numCtasZ * numCtasY;
            // Disable the multiCtasKvMode if there is only one CtaKv.
            if (numCtasPerSeqKv <= 1)
            {
                selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::Disabled;
                // Enable the persistent scheduler for better performance.
                selectKernelParams.mTileScheduler = TileScheduler::Persistent;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }
            else if (totalNumCtas < params.mMultiProcessorCount && isMlaGenKernel(params)
                && selectKernelParams.mTileSizeKv == 128 && tensorrt_llm::common::getEnvUseTileSizeKv64ForTrtllmGen())
            {
                // Use smaller tileSizeKv to fully utilize the SMs.
                selectKernelParams.mTileSizeKv = 64;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }

            // Enable the CgaSmemReduction if the numCtasPerSeqKv <= 16 as the maximum cluster dimension is 16.
            // Only the swapsMmaAbForGeneration kernel supports the CgaSmemReduction for now.
            if (!isDsv3MinLatencyMode && numCtasPerSeqKv > 1 && numCtasPerSeqKv <= 16
                && isSwapsMmaAbForGenerationKernel(selectKernelParams.mKernelType)
                && isGmemReduction(selectKernelParams.mMultiCtasKvMode) && !selectKernelParams.mForceGmemReduction)
            {
                selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::CgaSmemReduction;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }

            // Add the debug info when multiCtasKvMode is enabled.
            if (numCtasPerSeqKv > 1)
            {
                TLLM_LOG_DEBUG(
                    "TRTLLM-Gen launch info: multiCtasKvMode is enabled with tileSizeKv = %d, numCtasPerSeqKv = %d, "
                    "numCtasPerSeqQ = "
                    "%d, numCtasY = %d, numCtasZ = %d",
                    selectKernelParams.mTileSizeKv, numCtasPerSeqKv, numCtasPerSeqQ, numCtasY, numCtasZ);
            }
        }

        // The cluster size in the X dimension.
        int clusterDimX = selectKernelParams.mUses2CtaMma ? 2 : 1;
        if (isCgaSmemReduction(selectKernelParams.mMultiCtasKvMode))
        {
            // Note 2CtaMma and CgaSmemReduction cannot be used together currently.
            clusterDimX *= numCtasPerSeqKv;
        }

        // Compute the current number of CTAs in total.
        int totalNumCtas = numCtasX * numCtasZ * numCtasY;

        // Then split the headDimV into multiple CTAs if there are still unused SMs.
        if (isMlaGenKernel(params) && !selectKernelParams.mReuseSmemKForV && !selectKernelParams.mSelectNewKernel
            && !selectKernelParams.mUses2CtaMma)
        {
            // Split the headDimV into multiple CTAs if the utilization is not full.
            // It doesn't work with reuseSmemKForV currently.
            // TODO: find better heuristic of splitting headDimV across multiple CTAs.

            int corrFactor = isDsv3MinLatencyMode ? 1 : 2;
            if (selectKernelParams.mHeadDimPerCtaV == 512 && totalNumCtas * corrFactor <= params.mMultiProcessorCount)
            {
                // Use smaller headDimPerCtaV to fully utilize the SMs.
                selectKernelParams.mHeadDimPerCtaV
                    = totalNumCtas * 2 * corrFactor <= params.mMultiProcessorCount ? 128 : 256;
                // Need to select a different kernel.
                selectKernelParams.mSelectNewKernel = true;
            }
        }

        // Return the number of CTAs for X, Y and Z dimension and the cluster size in the X dimension.
        return std::make_tuple(numCtasPerSeqQ, numCtasPerSeqKv, numCtasX, numCtasY, numCtasZ, clusterDimX);
    }

    // Determine if we should use the SwapsMmaAbForGeneration kernel for MLA generation.
    bool useSwapsMmaAbMlaGenKernel(RunnerParams const& params) const
    {
        // Use the SwapsMmaAbForGeneration kernel for MLA generation when the following conditions are met:
        // 1. The seqLenPerCtaKv <= 1024 based on the benchmark results (this might be fine-tuned later).
        // 2. The numCtas (after splitting the heads across multiple CTAs) <= params.mMultiProcessorCount.

        // The maximum number Ctas per Kv sequence, which makes sure that each CtaKv has work to do.
        // Here we assume the stepKv is 256.
        int const maxNumCtasPerSeqKv = (params.mMaxSeqLenKv + 256 - 1) / 256;
        // The number of Ctas.
        int const numCtas
            = static_cast<int32_t>(params.mBatchSize * params.mMaxSeqLenQ * tc::divUp(params.mNumHeadsQPerKv, 16));
        // Compute numCtasPerSeqKv.
        int const numCtasPerSeqKv
            = std::min(maxNumCtasPerSeqKv, std::max(1, int32_t(params.mMultiProcessorCount / numCtas)));
        // Compute the seqLenPerCtaKv.
        int const seqLenPerCtaKv = (params.mMaxSeqLenKv + numCtasPerSeqKv - 1) / numCtasPerSeqKv;
        // Whether we should use the SwapsMmaAbForGeneration kernel for MLA generation.
        return seqLenPerCtaKv <= 1024 && numCtas <= params.mMultiProcessorCount;
    }

    std::pair<uint64_t, std::string> hashFromRunnerParams(
        RunnerParams const& params, SelectKernelParams& selectKernelParams) const
    {
        // The updated kernel type.
        FmhaKernelType& kernelType = selectKernelParams.mKernelType;
        printf("========params.mKernelType: %d and params.mNumHeadsQPerKv is %d\n", params.mKernelType,
            params.mNumHeadsQPerKv);
        // Generation kernelType will use either SwapsMmaAbForGeneration or KeepsMmaAbForGeneration.
        if (isGenerationKernel(params.mKernelType) && isMlaGenKernel(params))
        {
            // We use the low-latency kernel (SwapsMmaAbForGeneration with tileSizeQ = 16) when any of the following
            // conditions are met:
            // 1. The number of headsQPerKv is <= 32.
            // 2. The seqLenPerCtaKv <= 1024 based on the benchmark results (this might be fine-tuned later) and
            //    the numCtas (after splitting the heads across multiple CTAs) <= params.mMultiProcessorCount.

            // Check the conditions.
            if (params.mNumHeadsQPerKv <= 32 || useSwapsMmaAbMlaGenKernel(params))
            {
                kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
            }
            else
            {
                // Otherwise, we use the high-throughput kernel.
                kernelType = FmhaKernelType::KeepsMmaAbForGeneration;
                // Always use the separate reduction kernel.
                if (isMultiCtasKvEnabled(selectKernelParams.mMultiCtasKvMode))
                {
                    selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::GmemReductionWithSeparateKernel;
                }
                // The 2CTA keepsMmaAbForGeneration kernel is used when the numHeadsQPerKv is 128.
                if (params.mNumHeadsQPerKv == 128)
                {
                    selectKernelParams.mUses2CtaMma = true;
                    // Each Cta only handles 256 headDimV.
                    selectKernelParams.mHeadDimPerCtaV = 256;
                }
            }
        }
        else if (isGenerationKernel(params.mKernelType))
        {
            if (params.is_spec_dec_tree)
            {

                if (params.mNumHeadsQPerKv <= 16 && (params.mHeadDimQk == 64 || params.mHeadDimQk == 128))
                {
                    printf("======select kernel with KeepsMmaAbForGeneration =====\n");
                    kernelType = FmhaKernelType::KeepsMmaAbForGeneration;
                }
                else
                {
                    printf("======select kernel with SwapsMmaAbForGeneration =====\n");
                    kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
                }
            }
            else
            {
                kernelType = (params.mNumHeadsQPerKv <= 16 && params.mHeadDimQk != 32)
                    ? FmhaKernelType::SwapsMmaAbForGeneration
                    : FmhaKernelType::KeepsMmaAbForGeneration;
            }
        }

        // The maximum number of headsQPerKv that the kernel can support in one Cta.
        int maxNumHeadsQPerKvInCta = 1;
        if (isSwapsMmaAbForGenerationKernel(kernelType))
        {
            // Set the corresponding maxNumHeadsQPerKvInCta (tileSizeQ) for low-latency generation kernels.
            maxNumHeadsQPerKvInCta = (params.mNumHeadsQPerKv <= 8) ? 8 : 16;
            TLLM_CHECK_WITH_INFO((maxNumHeadsQPerKvInCta == 8 || maxNumHeadsQPerKvInCta == 16)
                    && (params.mNumHeadsQPerKv < maxNumHeadsQPerKvInCta
                        || params.mNumHeadsQPerKv % maxNumHeadsQPerKvInCta == 0),
                "Not supported");
        }
        else if (isKeepsMmaAbForGenerationKernel(kernelType))
        {
            // Use the maxNumHeadsQPerKvInCta (tileSizeQ) = 64 for MLA high-throughput generation kernels.
            maxNumHeadsQPerKvInCta = isMlaGenKernel(params) ? 64 : 32;
            if (params.is_spec_dec_tree)
            {
                maxNumHeadsQPerKvInCta = 128;
            }
            TLLM_CHECK_WITH_INFO((params.mNumHeadsQPerKv < maxNumHeadsQPerKvInCta
                                     || params.mNumHeadsQPerKv % maxNumHeadsQPerKvInCta == 0),
                "Not supported");
        }
        else if (isContextKernel(kernelType))
        {
            TLLM_CHECK_WITH_INFO(maxNumHeadsQPerKvInCta == 1, "Not supported");
        }

        // The mask type.
        selectKernelParams.mMaskType = params.mMaskType;
        // Enable sliding window or chunked causal if the max kv sequence length exceeds attention window size or
        // chunked attention size.
        // This is supported by causal-mask context kernels and generation-phase kernels.
        if ((selectKernelParams.mMaskType == TrtllmGenAttentionMaskType::Causal || !isContextKernel(params.mKernelType))
            && (params.mMaxSeqLenKv > params.mAttentionWindowSize || params.mChunkedAttentionSize != INT_MAX))
        {
            TLLM_CHECK_WITH_INFO(params.mMaxSeqLenKv <= params.mAttentionWindowSize
                    || params.mMaxSeqLenKv <= params.mChunkedAttentionSize,
                "Sliding window attention and chunked attention should not be used together");
            selectKernelParams.mMaskType = TrtllmGenAttentionMaskType::SlidingOrChunkedCausal;
        }
        // NumTokensPerPage is set to 0 when not selecting pagedKv-layout kernels.
        int numTokensPerPage = (!isPagedKv(params.mQkvLayout)) ? 0 : params.mNumTokensPerPage;
        printf(
            "========kernelType: %d and selectKernelParams.mMaskType is %d and params.mHeadDimQk is %d and "
            "params.mHeadDimV is %d\n",
            kernelType, selectKernelParams.mMaskType, params.mHeadDimQk, params.mHeadDimV);

        // Debug info.
        std::string info = "dtypeQ=" + std::to_string(static_cast<int>(mDtypeQ)) + ", dtypeKv="
            + std::to_string(static_cast<int>(mDtypeKv)) + ", dtypeOut=" + std::to_string(static_cast<int>(mDtypeOut))
            + ", sm=" + std::to_string(mSM) + ", qkvLayout=" + std::to_string(static_cast<int>(params.mQkvLayout))
            + ", maskType=" + std::to_string(static_cast<int>(selectKernelParams.mMaskType))
            + ", kernelType=" + std::to_string(static_cast<int>(kernelType))
            + ", tileScheduler=" + std::to_string(static_cast<int>(selectKernelParams.mTileScheduler))
            + ", multiCtasKvMode=" + std::to_string(static_cast<int>(selectKernelParams.mMultiCtasKvMode))
            + ", headDimPerCtaV=" + std::to_string(selectKernelParams.mHeadDimPerCtaV)
            + ", headDimQk=" + std::to_string(params.mHeadDimQk) + ", headDimV=" + std::to_string(params.mHeadDimV)
            + ", tileSizeKv=" + std::to_string(selectKernelParams.mTileSizeKv) + ", numTokensPerPage="
            + std::to_string(numTokensPerPage) + ", maxNumHeadsQPerKvInCta=" + std::to_string(maxNumHeadsQPerKvInCta)
            + ", reuseSmemKForV=" + std::to_string(selectKernelParams.mReuseSmemKForV)
            + ", uses2CtaMma=" + std::to_string(selectKernelParams.mUses2CtaMma);
        TLLM_LOG_DEBUG("Searching for kernel traits: " + info);

        return std::make_pair(
            hashID(static_cast<int>(params.mQkvLayout), static_cast<int>(selectKernelParams.mMaskType),
                static_cast<int>(kernelType), static_cast<int>(selectKernelParams.mTileScheduler),
                static_cast<int>(selectKernelParams.mMultiCtasKvMode), selectKernelParams.mHeadDimPerCtaV,
                params.mHeadDimQk, params.mHeadDimV, selectKernelParams.mTileSizeKv, numTokensPerPage,
                maxNumHeadsQPerKvInCta, selectKernelParams.mReuseSmemKForV, selectKernelParams.mUses2CtaMma),
            info);
    }

    Data_type mDtypeQ, mDtypeKv, mDtypeOut;
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;
    KernelMeta const* mKernelMeta;
    unsigned int mKernelMetaCount;
    unsigned int mSM;
    std::unordered_map<unsigned char const*, CUmodule> mModules;

    struct KernelInfo
    {
        unsigned int mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };

    std::unordered_map<uint64_t, KernelInfo> mFunctions;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class TllmFmhaKernelFactory
{
public:
    using KernelType = TllmGenFmhaKernel;

    KernelType const* getKernels(const typename KernelType::KernelMeta* pKernelList, unsigned int nbKernels,
        Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut, unsigned int sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        auto const id = hashID(dtypeQ, dtypeKv, dtypeOut, sm);
        auto const findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            KernelType* newKernel = new KernelType{pKernelList, nbKernels, dtypeQ, dtypeKv, dtypeOut, sm};
            newKernel->loadKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<KernelType>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TllmFmhaKernelFactory& Get()
    {
        int deviceId;
        cudaGetDevice(&deviceId);
        static std::unique_ptr<TllmFmhaKernelFactory> sFactory[32] = {nullptr};
        if (sFactory[deviceId] == nullptr)
        {
            TLLM_CHECK_WITH_INFO(deviceId < 32, "Invalid deviceId %d", deviceId);
            sFactory[deviceId] = std::make_unique<TllmFmhaKernelFactory>(TllmFmhaKernelFactory());
        }
        return *(sFactory[deviceId]);
    }

private:
    TllmFmhaKernelFactory() = default;

    inline uint64_t hashID(Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut, unsigned int sm) const
    {
        return static_cast<uint64_t>(sm) | static_cast<uint64_t>(dtypeQ) << 16 | static_cast<uint64_t>(dtypeKv) << 20
            | static_cast<uint64_t>(dtypeOut) << 24;
    }

    std::unordered_map<uint64_t, const std::unique_ptr<KernelType>> mKernels;
};

inline TllmGenFmhaKernel const* getTllmFmhaKernels(
    Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut, unsigned int sm)
{

#if !defined(EXCLUDE_SM_100) || !defined(EXCLUDE_SM_103)
    return TllmFmhaKernelFactory::Get().getKernels(sTllmGenFmhaKernelMetaInfos,
        sizeof(sTllmGenFmhaKernelMetaInfos) / sizeof(sTllmGenFmhaKernelMetaInfos[0]), dtypeQ, dtypeKv, dtypeOut, sm);
#else
    return nullptr;
#endif // EXCLUDE_SM_100
}

} // namespace kernels
} // namespace tensorrt_llm
