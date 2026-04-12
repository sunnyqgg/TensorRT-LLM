/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
 * Portions Copyright (c) 2025 by SGLang team (original implementation).
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

#include "dynamicTreeKernels.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

using namespace tensorrt_llm::runtime;

namespace kernels::speculative_decoding
{

//! \param parentList           [in]  layer-wise parent indices [bs, topK*(depth-1)+1]
//! \param selectedIndex        [in]  resampled history buffer indices [bs, draftTokenNum-1]
//! \param treeMask             [out] attention mask (which nodes each node can see)
//! \param positions            [out] position id per node [bs, draftTokenNum]
//! \param retrieveIndex        [out] tree node -> local index mapping [bs, draftTokenNum]
//! \param retrieveNextToken    [out] first-child pointer [bs, draftTokenNum], -1=none
//! \param retrieveNextSibling  [out] next-sibling pointer [bs, draftTokenNum], -1=none
//! \param topK                 top-K value per layer
//! \param depth                max tree depth (number of draft layers)
//! \param draftTokenNum        total tree nodes per batch (including root)
__global__ void buildDynamicTreeKernel(int64_t const* parentList, int64_t const* selectedIndex, int32_t* treeMask,
    int32_t* positions, int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling,
    SizeType32 topK, SizeType32 depth, SizeType32 draftTokenNum)
{
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;

    if (tid >= draftTokenNum)
    {
        return;
    }

    // treeMask layout: [batchSize, draftTokenNum, draftTokenNum] (QLEN_ONLY mode)
    int32_t tokenTreeIdx = draftTokenNum * draftTokenNum * bid + draftTokenNum * tid + 1;

    treeMask[tokenTreeIdx - 1] = 1; // self-attention diagonal
    for (int32_t i = 0; i < draftTokenNum - 1; i++)
    {
        treeMask[tokenTreeIdx + i] = 0;
    }

    int32_t position = 0;

    if (tid == 0)
    {
        positions[bid * draftTokenNum] = 0;

        // Reverse iteration: inserting at list head produces forward sibling order
        for (int32_t i = draftTokenNum - 1; i > 0; --i)
        {
            retrieveIndex[bid * draftTokenNum + i] = i;

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + i - 1] / topK;
            int32_t parentPosition = 0;

            if (parentTbIdx > 0)
            {
                int64_t parentTokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
                for (; parentPosition < draftTokenNum; ++parentPosition)
                {
                    if (selectedIndex[bid * (draftTokenNum - 1) + parentPosition] == parentTokenIdx)
                    {
                        ++parentPosition; // +1 because position 0 is root
                        break;
                    }
                }
            }

            if (parentPosition == draftTokenNum)
            {
                printf(
                    "WARNING: Invalid dynamic tree! Detected a token with no parent token selected. "
                    "Please check if the logprob has nan. The token will be ignored.\n");
                continue;
            }

            if (retrieveNextToken[bid * draftTokenNum + parentPosition] == -1)
            {
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
            }
            else
            {
                int32_t originNextToken = retrieveNextToken[bid * draftTokenNum + parentPosition];
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
                retrieveNextSibling[bid * draftTokenNum + i] = originNextToken;
            }
        }
        retrieveIndex[bid * draftTokenNum] = 0;
    }
    else
    {
        // Walk up to root, setting treeMask ancestor bits and counting depth
        int32_t curPosition = tid - 1;
        while (position < depth + 1)
        {
            position += 1;
            treeMask[tokenTreeIdx + curPosition] = 1;

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + curPosition] / topK;
            if (parentTbIdx == 0)
            {
                break;
            }

            int64_t tokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
            for (curPosition = 0; curPosition < draftTokenNum; ++curPosition)
            {
                if (selectedIndex[bid * (draftTokenNum - 1) + curPosition] == tokenIdx)
                {
                    break;
                }
            }
            if (curPosition == draftTokenNum)
            {
                break;
            }
        }
        positions[bid * draftTokenNum + tid] = position;
    }
}

//! Bit-packed variant of buildDynamicTreeKernel.
//! \param numInt32PerRow  int32 count per treeMask row (buffer stride; >= ceil(draftTokenNum/32) if padded)
__global__ void buildDynamicTreeKernelPacked(int64_t const* parentList, int64_t const* selectedIndex, int32_t* treeMask,
    int32_t* positions, int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling,
    SizeType32 topK, SizeType32 depth, SizeType32 draftTokenNum, SizeType32 numInt32PerRow)
{
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;

    if (tid >= draftTokenNum)
    {
        return;
    }

    int32_t rowBaseIdx = (bid * draftTokenNum + tid) * numInt32PerRow;

    treeMask[rowBaseIdx] = 1; // bit 0 = root, always visible

    int32_t position = 0;

    if (tid == 0)
    {
        positions[bid * draftTokenNum] = 0;

        for (int32_t i = draftTokenNum - 1; i > 0; --i)
        {
            retrieveIndex[bid * draftTokenNum + i] = i;

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + i - 1] / topK;
            int32_t parentPosition = 0;

            if (parentTbIdx > 0)
            {
                int64_t parentTokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
                for (; parentPosition < draftTokenNum; ++parentPosition)
                {
                    if (selectedIndex[bid * (draftTokenNum - 1) + parentPosition] == parentTokenIdx)
                    {
                        ++parentPosition;
                        break;
                    }
                }
            }

            if (parentPosition == draftTokenNum)
            {
                printf("WARNING: Invalid dynamic tree! Detected a token with no parent token selected.\n");
                continue;
            }

            if (retrieveNextToken[bid * draftTokenNum + parentPosition] == -1)
            {
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
            }
            else
            {
                int32_t originNextToken = retrieveNextToken[bid * draftTokenNum + parentPosition];
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
                retrieveNextSibling[bid * draftTokenNum + i] = originNextToken;
            }
        }
        retrieveIndex[bid * draftTokenNum] = 0;
    }
    else
    {
        int32_t curPosition = tid - 1;
        while (position < depth + 1)
        {
            position += 1;

            int32_t bitPosition = curPosition + 1; // +1 because bit 0 is root
            int32_t int32Idx = bitPosition / 32;
            int32_t bitIdx = bitPosition % 32;

            if (int32Idx < numInt32PerRow)
            {
                atomicOr(&treeMask[rowBaseIdx + int32Idx], 1 << bitIdx);
            }

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + curPosition] / topK;
            if (parentTbIdx == 0)
            {
                break;
            }

            int64_t tokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
            for (curPosition = 0; curPosition < draftTokenNum; ++curPosition)
            {
                if (selectedIndex[bid * (draftTokenNum - 1) + curPosition] == tokenIdx)
                {
                    break;
                }
            }
            if (curPosition == draftTokenNum)
            {
                break;
            }
        }
        positions[bid * draftTokenNum + tid] = position;
    }
}

void invokeBuildDynamicTree(int64_t const* parentList, int64_t const* selectedIndex, void* treeMask, int32_t* positions,
    int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling, SizeType32 batchSize,
    SizeType32 topK, SizeType32 depth, SizeType32 numDraftTokens, TreeMaskMode treeMaskMode, cudaStream_t stream,
    SizeType32 numInt32PerRow)
{
    dim3 grid(batchSize);
    dim3 block(numDraftTokens);

    if (treeMaskMode == TreeMaskMode::QLEN_ONLY_BITPACKING)
    {
        TLLM_CHECK_WITH_INFO(
            numInt32PerRow > 0, "numInt32PerRow must be the packed treeMask row stride in int32s (from buffer shape).");
        buildDynamicTreeKernelPacked<<<grid, block, 0, stream>>>(parentList, selectedIndex,
            static_cast<int32_t*>(treeMask), positions, retrieveIndex, retrieveNextToken, retrieveNextSibling, topK,
            depth, numDraftTokens, numInt32PerRow);
    }
    else
    {
        buildDynamicTreeKernel<<<grid, block, 0, stream>>>(parentList, selectedIndex, static_cast<int32_t*>(treeMask),
            positions, retrieveIndex, retrieveNextToken, retrieveNextSibling, topK, depth, numDraftTokens);
    }

    sync_check_cuda_error(stream);
}
