/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"

TRTLLM_NAMESPACE_BEGIN

using namespace tensorrt_llm::runtime;

namespace kernels::speculative_decoding
{

//! \brief CUDA kernel: 从 draft model 的层级采样结果构建动态树结构
//!
//! 每个 block 处理一个 batch 元素，每个线程处理一个树节点 (tid ∈ [0, draftTokenNum))。
//!   - tid==0 (root 线程): 构建"左孩子-右兄弟"链表 (retrieveIndex/NextToken/NextSibling)
//!   - tid>0 (非 root 线程): 向上追溯到 root，沿途填写 treeMask 和 positions
//!
//! 输入数据来自 draft model 的逐层采样：
//!   - parentList: 每层每个候选 token 的父节点在 history buffer 中的位置
//!   - selectedIndex: 经过 top-K 重采样后被选中的 token 在 history buffer 中的全局索引
//!
//! 示例 (topK=2, depth=3, draftTokenNum=7, batch 0, seqLen=10):
//!
//!   draft model 逐层采样过程:
//!     Layer 0: root → 采 top-2 → 得到 token A, B
//!     Layer 1: A → 采 top-2 → 得到 C, D;  B → 采 top-2 → 得到 E, F
//!     Layer 2: top-K 重采样，从 {A,B,C,D,E,F} 中选 6 个 (draftTokenNum-1=6)
//!
//!   假设 selectedIndex (重采样后的 history buffer 索引):
//!     selectedIndex[0..5] = [0, 1, 2, 3, 4, 5]
//!     含义: 选中了 history 中位置 0(A), 1(B), 2(C), 3(D), 4(E), 5(F)
//!
//!   parentList (每个 history 位置的父节点索引, parentTbIdx = selectedIndex[i] / topK):
//!     parentTbIdx=0 → 父节点是 root
//!     parentTbIdx=1 → parentList[1] 指向 A 在 history 中的索引
//!     parentTbIdx=2 → parentList[2] 指向 B 在 history 中的索引
//!
//!   构建结果 (树内位置, 0=root, 1..6=draft tokens):
//!
//!         root(0)
//!         /     \
//!       A(1)    B(2)         ← 深度1
//!       / \     / \
//!     C(3) D(4) E(5) F(6)   ← 深度2
//!
//!   输出 (batch 0):
//!     retrieveIndex:       [0, 1, 2, 3, 4, 5, 6]  (本地索引, 所有 batch 相同)
//!     retrieveNextToken:   [1, 3, 5, -1, -1, -1, -1]  (第一个子节点)
//!     retrieveNextSibling: [-1, 2, -1, 4, -1, 6, -1]  (下一个兄弟)
//!     positions:           [10, 11, 11, 12, 12, 12, 12]  (seqLen + depth)
//!     treeMask: 每个节点 tid 对自身及所有祖先置 1
//!
//! \param parentList           [输入] 层级父节点索引 [bs, topK*(depth-1)+1]
//! \param selectedIndex        [输入] 重采样选中的 history 索引 [bs, draftTokenNum-1]
//! \param verifiedSeqLen       [输入] 每个 batch 的已验证序列长度 [bs]
//! \param treeMask             [输出] 注意力掩码 (每个节点可见哪些节点)
//! \param positions            [输出] 每个节点的 position id [bs, draftTokenNum]
//! \param retrieveIndex        [输出] 树节点 → 一维全局下标 [bs, draftTokenNum]
//! \param retrieveNextToken    [输出] 左孩子指针 [bs, draftTokenNum], -1=无
//! \param retrieveNextSibling  [输出] 右兄弟指针 [bs, draftTokenNum], -1=无
//! \param topK                 每层采样的 top-K 值
//! \param depth                树的最大深度 (draft 层数)
//! \param draftTokenNum        每个 batch 的树节点总数 (含 root)
__global__ void buildDynamicTreeKernel(int64_t const* parentList, int64_t const* selectedIndex,
    SizeType32 const* verifiedSeqLen, int32_t* treeMask, int32_t* positions, int32_t* retrieveIndex,
    int32_t* retrieveNextToken, int32_t* retrieveNextSibling, SizeType32 topK, SizeType32 depth,
    SizeType32 draftTokenNum)
{
    int32_t bid = blockIdx.x;  // batch 索引
    int32_t tid = threadIdx.x; // 树节点索引 (0=root, 1..draftTokenNum-1=draft tokens)

    if (tid >= draftTokenNum)
    {
        return; // 超出树节点数的线程直接退出
    }

    // ==================== treeMask 初始化 (QLEN_ONLY 模式) ====================
    // treeMask 布局: [batchSize, draftTokenNum, draftTokenNum]，只包含 draft 区域
    // 每行 tid 表示该节点可见的其他节点 (1=可见, 0=不可见)
    int32_t seqLen = verifiedSeqLen[bid]; // 当前 batch 的已验证序列长度
    // tokenTreeIdx: treeMask 中"当前节点行、第 1 列 (跳过 root 列)"的一维偏移
    // root 列在 tokenTreeIdx-1 处
    int32_t tokenTreeIdx = draftTokenNum * draftTokenNum * bid + draftTokenNum * tid + 1;

    // 每个节点总是可见自己 → treeMask 对角线置 1 (root 列在 tokenTreeIdx-1)
    treeMask[tokenTreeIdx - 1] = 1;
    // 初始化本行剩余列为 0 (后续非 root 线程会在追溯祖先时将祖先列置 1)
    for (int32_t i = 0; i < draftTokenNum - 1; i++)
    {
        treeMask[tokenTreeIdx + i] = 0;
    }

    int32_t position = 0; // 当前节点到 root 的深度 (非 root 线程使用)

    // ==================== tid==0: Root 线程 ====================
    // 负责构建"左孩子-右兄弟"链表结构 (retrieveIndex/NextToken/NextSibling)
    if (tid == 0)
    {
        // root 的 position = seqLen (紧接已验证序列之后)
        positions[bid * draftTokenNum] = seqLen;

        // retrieveIndex 是树节点 → batch 内本地下标的映射
        // 对于 batch bid: retrieveIndex[i] = i (即恒等映射)
        // verify kernel 访问扁平化 targetPredict/predicts 时自行加 bx * N 偏移

        // 从最后一个节点到第一个节点，逆序构建链表
        // 逆序遍历的效果：同一父节点的子节点按原始顺序排列在兄弟链表中
        // (因为每次新节点插入到链表头部，逆序插入后最终顺序即为正序)
        for (int32_t i = draftTokenNum - 1; i > 0; --i)
        {
            // 设置当前节点的 retrieveIndex (本地索引)
            retrieveIndex[bid * draftTokenNum + i] = i;

            // ---- 查找节点 i 在树中的父节点位置 ----
            // selectedIndex[i-1] 是节点 i 在 history buffer 中的全局索引
            // 除以 topK 得到父节点在 parentList 中的索引 (parentTbIdx)
            // parentTbIdx==0 表示父节点是 root
            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + i - 1] / topK;
            int32_t parentPosition = 0; // 父节点在树内的位置 (0=root)

            if (parentTbIdx > 0)
            {
                // 非 root 父节点: 通过 parentList 查找父节点在 history 中的索引
                int64_t parentTokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
                // 在 selectedIndex 中查找这个 history 索引对应的树内位置
                for (; parentPosition < draftTokenNum; ++parentPosition)
                {
                    if (selectedIndex[bid * (draftTokenNum - 1) + parentPosition] == parentTokenIdx)
                    {
                        ++parentPosition; // 树内位置 = selectedIndex 中的索引 + 1 (因为 0 是 root)
                        break;
                    }
                }
            }
            // parentTbIdx==0 时 parentPosition 保持为 0，即父节点是 root

            if (parentPosition == draftTokenNum)
            {
                // 找不到父节点 (数据异常，可能是 logprob 有 nan)
                printf(
                    "WARNING: Invalid dynamic tree! Detected a token with no parent token selected. "
                    "Please check if the logprob has nan. The token will be ignored.\n");
                continue;
            }

            // ---- 将节点 i 插入父节点的子节点链表 ----
            // 使用"左孩子-右兄弟"表示法:
            //   retrieveNextToken[parent]  = 第一个子节点
            //   retrieveNextSibling[child] = 下一个兄弟节点
            if (retrieveNextToken[bid * draftTokenNum + parentPosition] == -1)
            {
                // 父节点还没有子节点 → 直接设为第一个子节点
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
            }
            else
            {
                // 父节点已有子节点 → 将节点 i 插入链表头部
                // 原来的第一个子节点变成 i 的兄弟
                int32_t originNextToken = retrieveNextToken[bid * draftTokenNum + parentPosition];
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
                retrieveNextSibling[bid * draftTokenNum + i] = originNextToken;
            }
        }
        // root 节点自身的 retrieveIndex (本地索引 = 0)
        retrieveIndex[bid * draftTokenNum] = 0;
    }
    // ==================== tid>0: 非 root 线程 ====================
    // 负责计算自身的 position (depth) 和 treeMask (标记所有祖先可见)
    else
    {
        // curPosition: 当前正在查看的 selectedIndex 下标 (即树内位置-1)
        // 从自身开始 (tid-1)，沿 parentList 向上追溯直到 root
        int32_t curPosition = tid - 1;
        while (true)
        {
            position += 1; // 每向上走一步，深度+1
            // 在 treeMask 中将祖先节点的列置为 1 (表示当前节点可以看到该祖先)
            treeMask[tokenTreeIdx + curPosition] = 1;

            // 查找父节点: selectedIndex[curPosition] / topK = parentTbIdx
            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + curPosition] / topK;
            if (parentTbIdx == 0)
            {
                break; // 到达 root，停止追溯
            }

            // 通过 parentList 找到父节点在 history 中的索引，
            // 再在 selectedIndex 中查找其树内位置
            int64_t tokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
            for (curPosition = 0; curPosition < draftTokenNum; ++curPosition)
            {
                if (selectedIndex[bid * (draftTokenNum - 1) + curPosition] == tokenIdx)
                {
                    break;
                }
            }
        }
        // position = 从当前节点到 root 的步数 = 深度
        // 加上 seqLen 得到在完整序列中的绝对位置
        positions[bid * draftTokenNum + tid] = position + seqLen;
    }
}

//! \brief CUDA kernel: 构建动态树 (bit-packed 掩码版本)
//!
//! 与 buildDynamicTreeKernel 逻辑完全相同，唯一区别是 treeMask 的存储格式:
//!   - buildDynamicTreeKernel:  treeMask 每行 draftTokenNum 个 int32 (每个元素 0/1)
//!   - buildDynamicTreeKernelPacked: treeMask 每行 ceil(draftTokenNum/32) 个 int32 (bit-packed)
//!     每个 int32 存 32 个 bit，bit i 表示当前节点是否可以 attend 到节点 i
//!
//! 参数含义同 buildDynamicTreeKernel，额外参数:
//! \param numInt32PerRow  treeMask 每行的 int32 数量 = ceil(draftTokenNum / 32)
__global__ void buildDynamicTreeKernelPacked(int64_t const* parentList, int64_t const* selectedIndex,
    SizeType32 const* verifiedSeqLen, int32_t* treeMask, int32_t* positions, int32_t* retrieveIndex,
    int32_t* retrieveNextToken, int32_t* retrieveNextSibling, SizeType32 topK, SizeType32 depth,
    SizeType32 draftTokenNum, SizeType32 numInt32PerRow)
{
    int32_t bid = blockIdx.x;  // batch 索引
    int32_t tid = threadIdx.x; // 树节点索引 (0=root, 1..draftTokenNum-1=draft tokens)

    if (tid >= draftTokenNum)
    {
        return; // 超出树节点数的线程直接退出
    }

    int32_t seqLen = verifiedSeqLen[bid]; // 当前 batch 的已验证序列长度

    // ==================== treeMask 初始化 (bit-packed) ====================
    // treeMask 布局: [batchSize, draftTokenNum, numInt32PerRow]
    // 每行 numInt32PerRow 个 int32，共 numInt32PerRow*32 个 bit
    // bit j 表示当前节点 tid 是否可以 attend 到节点 j
    int32_t rowBaseIdx = (bid * draftTokenNum + tid) * numInt32PerRow;

    // 每个节点总是可见 root (bit 0) → 将第一个 int32 的 bit 0 置 1
    treeMask[rowBaseIdx] = 1;

    int32_t position = 0; // 当前节点到 root 的深度 (非 root 线程使用)

    // ==================== tid==0: Root 线程 ====================
    // 负责构建"左孩子-右兄弟"链表结构 (retrieveIndex/NextToken/NextSibling)
    // 逻辑与 buildDynamicTreeKernel 完全相同
    if (tid == 0)
    {
        // root 的 position = seqLen (紧接已验证序列之后)
        positions[bid * draftTokenNum] = seqLen;

        // retrieveIndex: 树节点 → batch 内本地下标, retrieveIndex[i] = i

        // 从最后一个节点到第一个节点，逆序构建链表
        // (逆序插入链表头部，最终兄弟顺序为正序)
        for (int32_t i = draftTokenNum - 1; i > 0; --i)
        {
            // 设置当前节点的 retrieveIndex (本地索引)
            retrieveIndex[bid * draftTokenNum + i] = i;

            // ---- 查找节点 i 在树中的父节点位置 ----
            // selectedIndex[i-1] / topK = parentTbIdx (父节点在 parentList 中的索引)
            // parentTbIdx==0 表示父节点是 root
            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + i - 1] / topK;
            int32_t parentPosition = 0; // 父节点在树内的位置 (0=root)

            if (parentTbIdx > 0)
            {
                // 非 root 父节点: 通过 parentList 查找父节点在 history 中的索引
                int64_t parentTokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
                // 在 selectedIndex 中查找这个 history 索引对应的树内位置
                for (; parentPosition < draftTokenNum; ++parentPosition)
                {
                    if (selectedIndex[bid * (draftTokenNum - 1) + parentPosition] == parentTokenIdx)
                    {
                        ++parentPosition; // 树内位置 = selectedIndex 下标 + 1 (0 是 root)
                        break;
                    }
                }
            }
            // parentTbIdx==0 时 parentPosition 保持为 0，即父节点是 root

            if (parentPosition == draftTokenNum)
            {
                // 找不到父节点 (数据异常，可能是 logprob 有 nan)
                printf("WARNING: Invalid dynamic tree! Detected a token with no parent token selected.\n");
                continue;
            }

            // ---- 将节点 i 插入父节点的子节点链表 ----
            if (retrieveNextToken[bid * draftTokenNum + parentPosition] == -1)
            {
                // 父节点还没有子节点 → 直接设为第一个子节点
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
            }
            else
            {
                // 父节点已有子节点 → 将节点 i 插入链表头部
                int32_t originNextToken = retrieveNextToken[bid * draftTokenNum + parentPosition];
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
                retrieveNextSibling[bid * draftTokenNum + i] = originNextToken;
            }
        }
        // root 节点自身的 retrieveIndex (本地索引 = 0)
        retrieveIndex[bid * draftTokenNum] = 0;
    }
    // ==================== tid>0: 非 root 线程 ====================
    // 负责计算自身的 position (depth) 和 treeMask (bit-packed 标记所有祖先可见)
    else
    {
        // curPosition: 当前查看的 selectedIndex 下标 (= 树内位置 - 1)
        // 从自身 (tid-1) 开始沿 parentList 向上追溯到 root
        int32_t curPosition = tid - 1;
        while (true)
        {
            position += 1; // 每向上走一步，深度+1

            // 在 bit-packed treeMask 中将祖先节点的 bit 置 1
            // bitPosition = curPosition + 1 (因为 bit 0 是 root，已在初始化时置 1)
            int32_t bitPosition = curPosition + 1;
            int32_t int32Idx = bitPosition / 32; // 该 bit 在第几个 int32 中
            int32_t bitIdx = bitPosition % 32;   // 该 bit 在 int32 内的偏移

            if (int32Idx < numInt32PerRow)
            {
                // atomicOr 保证多个线程并发写同一个 int32 时的正确性
                atomicOr(&treeMask[rowBaseIdx + int32Idx], 1 << bitIdx);
            }

            // 查找父节点: selectedIndex[curPosition] / topK = parentTbIdx
            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + curPosition] / topK;
            if (parentTbIdx == 0)
            {
                break; // 到达 root，停止追溯
            }

            // 通过 parentList 找到父节点在 history 中的索引，
            // 再在 selectedIndex 中查找其树内位置
            int64_t tokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
            for (curPosition = 0; curPosition < draftTokenNum; ++curPosition)
            {
                if (selectedIndex[bid * (draftTokenNum - 1) + curPosition] == tokenIdx)
                {
                    break;
                }
            }
        }
        // position = 从当前节点到 root 的步数 = 深度
        // 加上 seqLen 得到在完整序列中的绝对位置
        positions[bid * draftTokenNum + tid] = position + seqLen;
    }
}

void invokeBuildDynamicTree(int64_t const* parentList, int64_t const* selectedIndex, SizeType32 const* verifiedSeqLen,
    void* treeMask, int32_t* positions, int32_t* retrieveIndex, int32_t* retrieveNextToken,
    int32_t* retrieveNextSibling, SizeType32 batchSize, SizeType32 topK, SizeType32 depth, SizeType32 numDraftTokens,
    TreeMaskMode treeMaskMode, cudaStream_t stream)
{
    dim3 grid(batchSize);
    dim3 block(numDraftTokens);

    if (treeMaskMode == TreeMaskMode::QLEN_ONLY_BITPACKING)
    {
        // Standard bit-packing: 32 bits per int32
        SizeType32 numInt32PerRow = (numDraftTokens + 31) / 32;

        buildDynamicTreeKernelPacked<<<grid, block, 0, stream>>>(parentList, selectedIndex, verifiedSeqLen,
            static_cast<int32_t*>(treeMask), positions, retrieveIndex, retrieveNextToken, retrieveNextSibling, topK,
            depth, numDraftTokens, numInt32PerRow);
    }
    else
    {
        // QLEN_ONLY 模式: treeMask 只包含 draft 区域 [bs, draftTokenNum, draftTokenNum]
        buildDynamicTreeKernel<<<grid, block, 0, stream>>>(parentList, selectedIndex, verifiedSeqLen,
            static_cast<int32_t*>(treeMask), positions, retrieveIndex, retrieveNextToken, retrieveNextSibling, topK,
            depth, numDraftTokens);
    }

    sync_check_cuda_error(stream);
}

//! \brief CUDA kernel: 贪心验证动态树，找到从 root 出发的最长匹配路径
//!
//! 树结构使用 "左孩子-右兄弟" 表示法 (由 buildDynamicTreeKernel 构建):
//!   - retrieveNextToken[i]:  节点 i 的第一个子节点索引 (-1 表示无子节点，叶子)
//!   - retrieveNextSibling[i]: 节点 i 的下一个兄弟节点索引 (-1 表示无兄弟)
//!
//! 验证策略 (贪心): 从 root 逐层向下，每层在同一父节点的所有子节点 (兄弟链表) 中
//! 寻找与 target model 预测匹配的第一个 token。找到则接受并进入下一层，找不到则终止。
//!
//  每个树节点 curIndex 同时有多个属性，分别存在不同数组中：

//   树节点 curIndex = 4 ("on"):
//     candidates[4]          = 261        ← 这个节点的 token id 是什么
//     retrieveIndex[4]       = 4          ← 这个节点在一维数组中的下标是什么
//     retrieveNextToken[4]   = -1         ← 这个节点的第一个子节点是谁
//     retrieveNextSibling[4] = -1         ← 这个节点的下一个兄弟是谁
//! 示例 (batch 0, numDraftTokens=6, numSpeculativeTokens=3, 即 depth=2):
//!
//!   树结构:
//!          root(0)
//!          /     \
//!       "cat"(1) "dog"(2)         ← 深度1
//!       /    \       \
//!    "is"(3) "on"(4) "ran"(5)     ← 深度2
//!
//!   ========== 输入 ==========
//!   (以下数组均为 batch 0 的片段，全局偏移 = bid * numDraftTokens = 0)
//!
//!     candidates:          [  _,  cat, dog,  is,  on, ran]
//!                            ^root占位  ^树内位置 1..5 对应 draft tokens
//!     retrieveIndex:       [  0,   1,   2,   3,   4,   5]
//!                            ^本地索引 (所有 batch 相同), verify kernel 自行加 bx*N 偏移
//!     retrieveNextToken:   [  1,   3,   5,  -1,  -1,  -1]  (第一个子节点)
//!     retrieveNextSibling: [ -1,   2,  -1,   4,  -1,  -1]  (下一个兄弟)
//!
//!     targetPredict (扁平化, target model 在每个树位置的贪心预测):
//!       targetPredict[0] = "cat"    (root → 预测下一个 token 是 "cat")
//!       targetPredict[1] = "on"     ("cat" → 预测 "on")
//!       targetPredict[2] = "went"   ("dog" → 预测 "went"，本例未用到)
//!       targetPredict[3] = "a"      ("is" → 预测 "a"，本例未用到)
//!       targetPredict[4] = "the"    ("on" → 预测 "the")
//!       targetPredict[5] = "fast"   ("ran" → 预测 "fast"，本例未用到)
//!
//!   ========== 验证过程 (bx=0, N=numDraftTokens=6, S=numSpeculativeTokens=3) ==========
//!     batchOffset = bx * N = 0
//!     初始: lastAcceptedLocalIdx = retrieveIndex[batchOffset + 0] = retrieveIndex[0+0] = 0
//!           acceptIndex[bx*S + 0] = acceptIndex[0*3+0] = 0, numAccepted=0, curIndex=0
//!
//!     j=1: curIndex = retrieveNextToken[batchOffset + curIndex] = retrieveNextToken[0+0] = 1
//!          ↳ curIndex 从 0(root) 更新为 1("cat")，即 root 的第一个子节点
//!          candidates[batchOffset + curIndex] = candidates[0+1] = "cat"
//!          targetPredict[batchOffset + lastAcceptedLocalIdx] = targetPredict[0+0] = "cat"  ✓ 匹配!
//!          → predicts[batchOffset + lastAcceptedLocalIdx] = predicts[0+0] = "cat", numAccepted=1
//!            draftLocalIdx = retrieveIndex[batchOffset + curIndex] = retrieveIndex[0+1] = 1
//!            acceptIndex[bx*S + numAccepted] = acceptIndex[0*3+1] = draftLocalIdx = 1
//!            lastAcceptedLocalIdx = draftLocalIdx = 1
//!
//!     j=2: curIndex = retrieveNextToken[batchOffset + curIndex] = retrieveNextToken[0+1] = 3
//!          ↳ curIndex 从 1("cat") 更新为 3("is")，即 "cat" 的第一个子节点
//!          candidates[batchOffset + curIndex] = candidates[0+3] = "is"
//!          targetPredict[batchOffset + lastAcceptedLocalIdx] = targetPredict[0+1] = "on"  ✗ 不匹配
//!          → 查兄弟: curIndex = retrieveNextSibling[batchOffset + curIndex] = retrieveNextSibling[0+3] = 4
//!            ↳ curIndex 从 3("is") 更新为 4("on")，即 "is" 的下一个兄弟
//!          candidates[batchOffset + curIndex] = candidates[0+4] = "on"
//!          targetPredict[batchOffset + lastAcceptedLocalIdx] = targetPredict[0+1] = "on"  ✓ 匹配!
//!          → predicts[batchOffset + lastAcceptedLocalIdx] = predicts[0+1] = "on", numAccepted=2
//!            draftLocalIdx = retrieveIndex[batchOffset + curIndex] = retrieveIndex[0+4] = 4
//!            acceptIndex[bx*S + numAccepted] = acceptIndex[0*3+2] = draftLocalIdx = 4
//!            lastAcceptedLocalIdx = draftLocalIdx = 4
//!
//!     j=3: 不执行 (j < numSpeculativeTokens=3 不成立，循环结束)
//!
//!   ========== 输出 ==========
//!     acceptTokenNum[bx] = acceptTokenNum[0] = 2    (接受了 2 个 draft token)
//!     acceptIndex[0]    = [0, 1, 4]                  (本地树位置: root(0) → cat(1) → on(4))
//!     predicts[batchOffset + lastAcceptedLocalIdx] = predicts[0+4]
//!       = targetPredict[batchOffset + lastAcceptedLocalIdx] = targetPredict[0+4] = "the"  (bonus)
//!
//!   总结: 输出 3 个 token: "cat", "on", "the" (2 accepted + 1 bonus)
//!
//! ==================== bs=2 示例 ====================
//! 假设 batchSize=2, numDraftTokens=6, numSpeculativeTokens=3 (depth=2)
//! 两个 batch 的树结构、candidates、targetPredict 值完全相同
//!
//!   树结构 (batch 0 和 batch 1 相同):
//!          root(0)
//!          /     \
//!       "cat"(1) "dog"(2)         ← 深度1
//!       /    \       \
//!    "is"(3) "on"(4) "ran"(5)     ← 深度2
//!
//!   ========== 内存布局 (两个 batch 拼接在一起) ==========
//!
//!   candidates [bs=2, numDraftTokens=6], 连续存储:
//!     index:    [ 0,   1,   2,   3,   4,   5,  | 6,   7,   8,   9,  10,  11 ]
//!     value:    [ _,  cat, dog,  is,  on, ran,  | _,  cat, dog,  is,  on, ran ]
//!               |<------- batch 0 -------->|    |<------- batch 1 -------->|
//!               bx=0, 偏移=0*6=0                bx=1, 偏移=1*6=6
//!
//!   retrieveIndex [bs=2, numDraftTokens=6]:
//!     index:    [ 0,  1,  2,  3,  4,  5,  | 6,  7,  8,  9, 10, 11 ]
//!     value:    [ 0,  1,  2,  3,  4,  5,  | 0,  1,  2,  3,  4,  5 ]
//!               |<----- batch 0 ----->|    |<----- batch 1 ----->|
//!               本地索引, 所有 batch 完全相同
//!
//!   retrieveNextToken [bs=2, numDraftTokens=6]:
//!     index:    [ 0,  1,  2,  3,  4,  5,  | 6,  7,  8,  9, 10, 11 ]
//!     value:    [ 1,  3,  5, -1, -1, -1,  | 1,  3,  5, -1, -1, -1 ]
//!               |<----- batch 0 ----->|    |<----- batch 1 ----->|
//!
//!   retrieveNextSibling [bs=2, numDraftTokens=6]:
//!     index:    [ 0,  1,  2,  3,  4,  5,  | 6,  7,  8,  9, 10, 11 ]
//!     value:    [-1,  2, -1,  4, -1, -1,  |-1,  2, -1,  4, -1, -1 ]
//!               |<----- batch 0 ----->|    |<----- batch 1 ----->|
//!
//!   targetPredict [bs*numDraftTokens = 12], 连续存储:
//!     index:    [  0,     1,      2,    3,     4,      5,   | 6,     7,      8,    9,    10,     11  ]
//!     value:    ["cat", "on", "went",  "a", "the", "fast", |"cat", "on", "went",  "a", "the", "fast"]
//!               |<------------ batch 0 ------------>|       |<------------ batch 1 ------------>|
//!
//!   ========== Kernel 启动: grid(2), block(1) ==========
//!   两个 block 并行处理 batch 0 和 batch 1
//!
//!   ---------- Block 0 (bx=0) ----------
//!   batchOffset = bx * N = 0 * 6 = 0
//!
//!   初始: lastAcceptedLocalIdx = retrieveIndex[0+0] = 0
//!         acceptIndex[0*3+0] = 0, numAccepted=0, curIndex=0
//!
//!   j=1: curIndex = retrieveNextToken[0+0] = 1
//!        candidates[0+1]="cat" == targetPredict[0+0]="cat" ✓
//!        → predicts[0+0]="cat", numAccepted=1, acceptIndex[0*3+1]=1, lastAcceptedLocalIdx=1
//!
//!   j=2: curIndex = retrieveNextToken[0+1] = 3
//!        candidates[0+3]="is" != targetPredict[0+1]="on" ✗
//!        → curIndex = retrieveNextSibling[0+3] = 4
//!        candidates[0+4]="on" == targetPredict[0+1]="on" ✓
//!        → predicts[0+1]="on", numAccepted=2, acceptIndex[0*3+2]=4, lastAcceptedLocalIdx=4
//!
//!   j=3: 不执行 (j < numSpeculativeTokens=3 不成立)
//!
//!   输出: acceptTokenNum[0]=2, acceptIndex[0..2]=[0,1,4]  ← 本地树位置
//!         predicts[0+0]="cat", predicts[0+1]="on", predicts[0+4]="the"(bonus)
//!
//!   ---------- Block 1 (bx=1) ----------
//!   batchOffset = bx * N = 1 * 6 = 6
//!
//!   初始: lastAcceptedLocalIdx = retrieveIndex[6+0] = 0  ← 本地索引！不再是 6
//!         acceptIndex[1*3+0] = 0, numAccepted=0, curIndex=0
//!
//!   j=1: curIndex = retrieveNextToken[6+0] = 1
//!        candidates[6+1]="cat" == targetPredict[6+0]="cat" ✓
//!        ↑ targetPredict 索引 = batchOffset + lastAcceptedLocalIdx = 6 + 0 = 6
//!        → predicts[6+0]="cat", numAccepted=1
//!          draftLocalIdx = retrieveIndex[6+1] = 1
//!          acceptIndex[1*3+1] = 1, lastAcceptedLocalIdx = 1
//!
//!   j=2: curIndex = retrieveNextToken[6+1] = 3
//!        candidates[6+3]="is" != targetPredict[6+1]="on" ✗
//!        → curIndex = retrieveNextSibling[6+3] = 4
//!        candidates[6+4]="on" == targetPredict[6+1]="on" ✓
//!        → predicts[6+1]="on", numAccepted=2
//!          draftLocalIdx = retrieveIndex[6+4] = 4
//!          acceptIndex[1*3+2] = 4, lastAcceptedLocalIdx = 4
//!
//!   j=3: 不执行 (j < numSpeculativeTokens=3 不成立)
//!
//!   输出: acceptTokenNum[1]=2, acceptIndex[3..5]=[0,1,4]  ← 本地树位置，和 batch 0 完全相同!
//!         predicts[6+0]="cat", predicts[6+1]="on", predicts[6+4]="the"(bonus)
//!
//!   ========== 最终输出汇总 ==========
//!
//!   acceptTokenNum [bs=2]:  [2, 2]
//!
//!   acceptIndex [bs=2, numSpeculativeTokens=3]:
//!     batch 0: [0, 1, 4]          ← 本地树位置
//!     batch 1: [0, 1, 4]          ← 本地树位置，和 batch 0 相同 (因为树结构相同)
//!     注意: 现在 acceptIndex 存本地索引，Python 端可直接用作 add_token(step=...)
//!
//!   predicts [bs*numDraftTokens=12]:
//!     index:  [  0,    1,  2,  3,    4,   5,  |  6,    7,  8,  9,   10,  11 ]
//!     value:  ["cat","on", _,  _, "the",  _,  |"cat","on", _,  _, "the",  _ ]
//!             |<------- batch 0 -------->|     |<------- batch 1 -------->|
//!     predicts 仍用扁平化全局索引 (batchOffset + localIdx)
//!
//!   关键观察:
//!     1. 所有数组 (retrieveIndex/NextToken/NextSibling) 都使用 batch 内本地索引
//!     2. 访问数组时统一加 batchOffset: array[batchOffset + localIndex]
//!     3. acceptIndex 输出本地树位置，Python 端直接可用，无需偏移转换
//!     4. predicts/targetPredict 是扁平化的 [bs*N]，kernel 自行加 batchOffset
//!
//! 每个 block 处理一个 batch 元素，单线程 (block(1))。
//!
//! \param predicts            [输出] [bs*numDraftTokens] 被接受 token id + 末尾 bonus token
//! \param acceptIndex         [输出] [bs, numSpeculativeTokens] 被接受路径的本地树位置
//! \param acceptTokenNum      [输出] [bs] 每个 batch 接受的 draft token 数 (不含 bonus)
//! \param candidates          [输入] [bs, numDraftTokens] 每个树节点的候选 token id
//! \param retrieveIndex       [输入] [bs, numDraftTokens] 树节点 → 本地下标 (= i)
//! \param retrieveNextToken   [输入] [bs, numDraftTokens] 左孩子指针, -1=无
//! \param retrieveNextSibling [输入] [bs, numDraftTokens] 右兄弟指针, -1=无
//! \param targetPredict       [输入] [bs*numDraftTokens] target model 在每个位置的预测
//! \param batchSize           batch 大小
//! \param numSpeculativeTokens  acceptIndex 第二维大小 (≥ 最大可能接受数 + 1)
//! \param numDraftTokens      每个 batch 的树节点总数 (含 root)
__global__ void verifyDynamicTreeGreedyKernel(int64_t* predicts, int64_t* acceptIndex, int64_t* acceptTokenNum,
    int64_t const* candidates, int32_t const* retrieveIndex, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, int64_t const* targetPredict, uint32_t batchSize, uint32_t numSpeculativeTokens,
    uint32_t numDraftTokens)
{
    // 每个 block 处理一个 batch 元素
    uint32_t bx = blockIdx.x;

    // retrieveIndex 现在存本地索引 (0, 1, 2, ..., N-1)
    // 访问扁平化的 targetPredict/predicts 时需要加 bx * N 偏移
    uint32_t batchOffset = bx * numDraftTokens;

    // 获取 root 节点的本地下标，作为起始的"最后接受位置"
    int32_t lastAcceptedLocalIdx = retrieveIndex[batchOffset]; // = 0
    // acceptIndex[0] 存放 root 的本地位置
    acceptIndex[bx * numSpeculativeTokens] = lastAcceptedLocalIdx;
    // 已接受的 draft token 计数，初始为 0
    uint32_t numAcceptedTokens = 0;
    // 当前遍历的树节点索引，从 root (索引 0) 开始
    int32_t curIndex = 0;

    // 逐层遍历树，j 表示当前深度 (从 1 开始，因为深度 0 是 root)
    for (uint32_t j = 1; j < numSpeculativeTokens; ++j)
    {
        // 进入下一层：获取当前已接受节点 (curIndex) 的第一个子节点
        curIndex = retrieveNextToken[batchOffset + curIndex];
        // 遍历该层同一父节点下的所有兄弟节点，寻找与 target 预测匹配的 token
        while (curIndex != -1)
        {
            // 获取当前候选节点的本地下标
            int32_t draftLocalIdx = retrieveIndex[batchOffset + curIndex];
            // 获取 draft model 在该节点生成的候选 token id
            int64_t draftTokenId = candidates[batchOffset + curIndex];
            // 获取 target model 在"最后一个被接受的位置"预测的 token id
            // 注意: targetPredict 是扁平化的 [bs*N], 需要加 batchOffset
            int64_t targetTokenId = targetPredict[batchOffset + lastAcceptedLocalIdx];

            if (draftTokenId == targetTokenId)
            {
                // 贪心匹配成功：draft token 与 target 预测一致
                // 将该 token 写入 predicts 数组 (在父节点位置记录预测结果)
                predicts[batchOffset + lastAcceptedLocalIdx] = targetTokenId;
                // 增加接受计数
                ++numAcceptedTokens;
                // 记录被接受 token 的本地位置
                acceptIndex[bx * numSpeculativeTokens + numAcceptedTokens] = draftLocalIdx;
                // 更新"最后接受位置"为当前节点
                lastAcceptedLocalIdx = draftLocalIdx;
                break;
            }
            else
            {
                // 不匹配，沿兄弟链表移动到下一个兄弟节点继续尝试
                curIndex = retrieveNextSibling[batchOffset + curIndex];
            }
        }
        // 如果遍历完所有兄弟都没有匹配 (curIndex == -1)，终止验证
        if (curIndex == -1)
            break;
    }

    // 写回该 batch 元素最终接受的 draft token 数量
    acceptTokenNum[bx] = numAcceptedTokens;
    // Bonus token: 将 target model 在最后一个被接受位置的预测写入 predicts
    predicts[batchOffset + lastAcceptedLocalIdx] = targetPredict[batchOffset + lastAcceptedLocalIdx];
}

void invokeVerifyDynamicTreeGreedy(int64_t* predicts, int64_t* acceptIndex, int64_t* acceptTokenNum,
    int64_t const* candidates, int32_t const* retrieveIndex, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, int64_t const* targetPredict, SizeType32 batchSize, SizeType32 numDraftTokens,
    SizeType32 numSpecStep, cudaStream_t stream)
{
    dim3 grid(batchSize);
    dim3 block(1);

    verifyDynamicTreeGreedyKernel<<<grid, block, 0, stream>>>(predicts, acceptIndex, acceptTokenNum, candidates,
        retrieveIndex, retrieveNextToken, retrieveNextSibling, targetPredict, batchSize, numSpecStep, numDraftTokens);

    sync_check_cuda_error(stream);
}

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
