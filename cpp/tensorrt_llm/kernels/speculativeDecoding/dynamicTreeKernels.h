/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/runtime/common.h"
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding
{

//! \brief Tree mask mode for dynamic tree building
enum class TreeMaskMode : int32_t
{
    QLEN_ONLY = 1,            //! Query length only (bool array) [bs, draftTokenNum, draftTokenNum]
    QLEN_ONLY_BITPACKING = 2, //! Query length with bit packing (32 bits per int32)
};

//! \brief Build dynamic tree structure efficiently
//! Ported from SGLang's eagle_utils.cu for high-performance tree construction.
//! \param parentList [batchSize, topK * (depth - 1) + 1], on GPU. int64.
//! Parent indices for each token in the tree (layer-local relative indices).
//! \param selectedIndex [batchSize, numDraftTokens - 1], on GPU. int64.
//! Selected token indices (excluding root node).
//! \param treeMask output buffer [varies by mode], on GPU. int32.
//! Attention mask for tree structure. Shape depends on treeMaskMode.
//! \param positions output buffer [batchSize * numDraftTokens], on GPU. int32.
//! Position IDs for each draft token.
//! \param retrieveIndex output buffer [batchSize, numDraftTokens], on GPU. int32.
//! Local indices for retrieving tokens (0, 1, ..., numDraftTokens-1).
//! \param retrieveNextToken output buffer [batchSize, numDraftTokens], on GPU. int32.
//! Index of the first child token for each node.
//! \param retrieveNextSibling output buffer [batchSize, numDraftTokens], on GPU. int32.
//! Index of the next sibling token for each node.
//! \param batchSize runtime::SizeType32. Batch size.
//! \param topK runtime::SizeType32. Number of top-K tokens per node.
//! \param depth runtime::SizeType32. Tree depth.
//! \param numDraftTokens runtime::SizeType32. Total number of draft tokens.
//! \param treeMaskMode TreeMaskMode. Attention mask mode.
//! \param stream cuda stream
//! \param numInt32PerRow For QLEN_ONLY_BITPACKING: treeMask row stride in int32s (must be > 0). Ignored otherwise.
void invokeBuildDynamicTree(int64_t const* parentList, int64_t const* selectedIndex, void* treeMask, int32_t* positions,
    int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling, runtime::SizeType32 batchSize,
    runtime::SizeType32 topK, runtime::SizeType32 depth, runtime::SizeType32 numDraftTokens, TreeMaskMode treeMaskMode,
    cudaStream_t stream, runtime::SizeType32 numInt32PerRow);

void invokeVerifyDynamicTreeGreedyPacked(int32_t* predicts, int32_t* acceptIndex, int32_t* acceptTokenNum,
    int32_t* acceptToken, int32_t const* candidates, int32_t const* retrievePacked, int32_t const* targetPredict,
    bool const* treeValid, runtime::SizeType32 batchSize, runtime::SizeType32 numDraftTokens,
    runtime::SizeType32 numSpecStep, cudaStream_t stream);

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
