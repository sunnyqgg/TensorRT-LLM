# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dynamic Tree Operations for EAGLE3 Speculative Decoding

This module provides high-performance CUDA kernel wrappers for building and verifying
dynamic tree structures used in EAGLE3 speculative decoding. It integrates SGLang's
optimized CUDA kernels into TensorRT-LLM's PyTorch backend.

Key Features:
- Efficient tree construction from layer-local parent indices
- Greedy tree verification with parallel traversal
- Buffer pre-allocation and reuse for minimal runtime overhead
"""

import torch


class DynamicTreeOpsConverter:
    """
    Converter for dynamic tree operations using CUDA kernels.

    This class handles data format conversion and CUDA kernel invocation for
    building and verifying dynamic trees in EAGLE3 speculative decoding.

    Args:
        dynamic_tree_max_topK: Maximum top-K tokens per node.
        max_draft_len: Maximum draft length (tree depth).
        max_total_draft_tokens: Total number of draft tokens.
        max_batch_size: Maximum batch size.
        device: CUDA device.
    """

    def __init__(
        self,
        dynamic_tree_max_topK: int,
        max_draft_len: int,
        max_total_draft_tokens: int,
        max_batch_size: int,
        device: torch.device,
    ):
        self.K = dynamic_tree_max_topK
        self.depth = max_draft_len

        # Pre-allocated output buffers for verify_dynamic_tree_greedy_out_packed_op
        max_path_len = max_draft_len + 1
        self._verify_accept_index_buf = torch.zeros(
            max_batch_size, max_path_len, dtype=torch.int32, device=device
        )
        self._verify_accept_token_num_buf = torch.zeros(
            max_batch_size, dtype=torch.int32, device=device
        )
        self._verify_accept_token_buf = torch.zeros(
            max_batch_size, max_path_len, dtype=torch.int32, device=device
        )

    def build_dynamic_tree(
        self,
        history_draft_tokens_parent_buffer: torch.Tensor,
        topk_score_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        positions: torch.Tensor,
        retrieve_index: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        use_packed_mask: bool = False,
    ) -> None:
        """
        Build dynamic tree structure using CUDA kernel (in-place, writes to pre-allocated buffers).

        All output tensors are written in-place; nothing is returned.

        Args:
            history_draft_tokens_parent_buffer: [bs, history_size] int64
                Parent indices (directly used as parentList).
            topk_score_indices: [bs, max_total_draft_tokens] int64
                Selected token indices (directly used as selectedIndex).
            tree_mask: [bs, N, packed_bits] int32 (packed) or [bs, N, N] bool
                Pre-allocated output buffer for attention mask.
            positions: [bs, num_draft_tokens] int32
                Pre-allocated output buffer for position IDs.
            retrieve_index: [bs, num_draft_tokens] int32
                Pre-allocated output buffer for token retrieval indices.
            retrieve_next_token: [bs, num_draft_tokens] int32
                Pre-allocated output buffer for first child indices.
            retrieve_next_sibling: [bs, num_draft_tokens] int32
                Pre-allocated output buffer for next sibling indices.
            use_packed_mask: bool
                Use bit-packed mask for memory efficiency.
        """
        bs = topk_score_indices.shape[0]
        # +1 because num_draft_tokens includes root node in SGLang's convention
        num_draft_tokens = topk_score_indices.shape[1] + 1
        tree_mask_mode = 2 if use_packed_mask else 1  # QLEN_ONLY_BITPACKING / QLEN_ONLY

        # Actual buffer row stride (int32s); kernel otherwise computes ceil(num_draft_tokens / 32).
        num_int32_per_row = tree_mask.shape[-1] if use_packed_mask else 0

        # CUDA kernel indexes as ptr[bid * draftTokenNum + tid], so dim1 must equal num_draft_tokens.
        assert positions.shape[-1] == num_draft_tokens, (
            f"positions dim1 ({positions.shape[-1]}) != num_draft_tokens ({num_draft_tokens})"
        )

        # Call CUDA kernel in-place
        try:
            torch.ops.trtllm.build_dynamic_tree_op(
                history_draft_tokens_parent_buffer[:bs],
                topk_score_indices,
                tree_mask,
                positions,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                self.K,
                self.depth,
                num_draft_tokens,
                tree_mask_mode,
                num_int32_per_row,
            )
        except Exception as e:
            raise RuntimeError(
                f"build_dynamic_tree_op failed: {e}\n"
                f"Inputs: bs={bs}, K={self.K}, depth={self.depth}, "
                f"num_draft_tokens={num_draft_tokens}"
            ) from e

    def verify_dynamic_tree_greedy_out_packed(
        self,
        candidates: torch.Tensor,
        retrieve_packed: torch.Tensor,
        target_predict: torch.Tensor,
        num_gens: int,
        num_spec_step: int,
        tree_valid: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """In-place verify with int32 token tensors and packed int32 retrieve layout."""
        N = candidates.size(1)
        accept_index = self._verify_accept_index_buf[:num_gens]
        accept_token_num = self._verify_accept_token_num_buf[:num_gens]
        accept_token = self._verify_accept_token_buf[:num_gens]

        if tree_valid is None:
            tree_valid = torch.ones(num_gens, dtype=torch.bool, device=candidates.device)

        rp = retrieve_packed[:, :N, :].contiguous()
        try:
            torch.ops.trtllm.verify_dynamic_tree_greedy_out_packed_op(
                candidates,
                rp,
                target_predict,
                accept_index,
                accept_token_num,
                accept_token,
                tree_valid,
                num_spec_step,
            )
        except Exception as e:
            raise RuntimeError(
                f"verify_dynamic_tree_greedy_out_packed_op failed: {e}\n"
                f"Inputs: num_gens={num_gens}, N={N}, num_spec_step={num_spec_step}"
            ) from e

        return accept_index, accept_token_num, accept_token
