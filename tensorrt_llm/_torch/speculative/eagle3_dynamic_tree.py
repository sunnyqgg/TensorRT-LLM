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
Eagle3 one-model dynamic tree speculative decoding.

This module separates the dynamic tree logic from the base Eagle3 one-model
worker (eagle3.py) for clearer architecture. The base eagle3.py handles linear
tree mode and shared infrastructure; this file handles:

- Eagle3OneModelDynamicTreeWorker: draft loop, verification, tree construction
- Eagle3OneModelDynamicTreeSampler: sampler with accepted indices tracking
- Buffer management for dynamic tree operations (history, scores, parents, masks)

Goal: One-model dynamic tree accept rate must match two-model dynamic tree
accept rate exactly. See eagle3-dynamic-tree.md section 22 for details.
"""

from typing import TYPE_CHECKING

import torch
from torch import nn

from ..attention_backend import AttentionMetadata
from ..pyexecutor.llm_request import LlmRequestState
from ..pyexecutor.sampler import TorchSampler
from .eagle3 import Eagle3OneModelWorker
from .mtp import MTPSampler
from .spec_tree_manager import SpecTreeManager

if TYPE_CHECKING:
    from ...llmapi.llm_args import EagleDecodingConfig


class Eagle3OneModelDynamicTreeSampler(MTPSampler):
    """Sampler for one-model EAGLE3 dynamic tree mode.

    Extends MTPSampler with accepted draft token indices tracking, which is
    needed for KV cache rewind after dynamic tree verification.
    """

    def __init__(self, args: TorchSampler.Args, spec_config=None):
        super().__init__(args, nextn=args.max_total_draft_tokens)
        seq_slots = args.max_num_sequences
        self._accepted_indices_store = torch.full(
            (seq_slots, args.max_total_draft_tokens), -1, dtype=torch.int32, device="cuda"
        )

    def sample_async(self, scheduled_requests, outputs, num_context_logits_prefix_sum):
        if "accepted_draft_tokens_indices" in outputs:
            requests = scheduled_requests.all_requests()
            slots = torch.as_tensor([r.py_seq_slot for r in requests], device="cuda")
            indices = outputs["accepted_draft_tokens_indices"][: len(requests)]
            self._accepted_indices_store.index_copy_(0, slots, indices)
        return super().sample_async(scheduled_requests, outputs, num_context_logits_prefix_sum)

    def update_requests(self, state, resource_manager=None):
        super().update_requests(state, resource_manager)
        for req in state.scheduled_requests.generation_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            n_accepted = req.py_num_accepted_draft_tokens
            if n_accepted > 0:
                slot = req.py_seq_slot
                req.py_num_accepted_draft_tokens_indices = (
                    self._accepted_indices_store[slot, :n_accepted].cpu().tolist()
                )


class Eagle3OneModelDynamicTreeWorker(Eagle3OneModelWorker):
    """Eagle3 one-model worker with dynamic tree support.

    Inherits linear tree functionality from Eagle3OneModelWorker and adds
    dynamic tree draft loop, verification, and tree construction.
    """

    def __init__(
        self, spec_config: "EagleDecodingConfig", mapping, use_separate_draft_kv_cache: bool = False
    ):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        assert self.use_dynamic_tree, (
            "Eagle3OneModelDynamicTreeWorker requires use_dynamic_tree=True"
        )

        from .dynamic_tree_ops import create_dynamic_tree_ops_converter

        K = spec_config.dynamic_tree_max_topK
        max_draft_len = spec_config.max_draft_len
        max_total_draft_tokens = spec_config.tokens_per_gen_step - 1
        max_batch_size = 256

        self.dynamic_tree_max_topK = K
        self._max_total_draft_tokens = max_total_draft_tokens
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # 1D buffers: store tokens contiguously (matching two-model naming)
        self.draft_tokens_buffer = torch.zeros(
            (max_batch_size * max_total_draft_tokens), dtype=torch.int64, device="cuda"
        )
        self.position_ids_buffer = torch.zeros(
            (max_batch_size * (max_total_draft_tokens + 1)), dtype=torch.int64, device="cuda"
        )
        self.history_draft_tokens_buffer = torch.zeros(
            (max_batch_size, (K + K * K * (max_draft_len - 1))), dtype=torch.int64, device="cuda"
        )
        self.history_score_buffer = torch.zeros(
            (max_batch_size, K + K * K * (max_draft_len - 1)), dtype=torch.float32, device="cuda"
        )
        self.history_draft_tokens_parent_buffer = torch.zeros(
            (max_batch_size, K * (max_draft_len - 1) + 1), dtype=torch.int64, device="cuda"
        )
        self.tree_mask_buffer = torch.zeros(
            (max_batch_size * (max_total_draft_tokens + 1) * (max_total_draft_tokens + 1)),
            dtype=torch.int32,
            device="cuda",
        )
        self.tree_mask_init_buffer = (
            torch.eye(K, dtype=torch.int32, device="cuda").unsqueeze(0).repeat(max_batch_size, 1, 1)
        )
        self.tree_mask_padding_zeros = torch.zeros(
            (max_batch_size, max_total_draft_tokens, max_total_draft_tokens + 1),
            dtype=torch.int32,
            device="cuda",
        )

        self.tree_ops_converter = create_dynamic_tree_ops_converter(
            dynamic_tree_max_topK=K,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            max_batch_size=max_batch_size,
            device=torch.device("cuda"),
        )

        # Initialized by _sample_and_accept_dynamic_tree; None during warmup
        # when linear fallback verification is used.
        self._last_accepted_tree_pos = None
        self._tree_topology_target_tokens = None

        # Draft KV lens tracking: persistent across iterations.
        # Saves the draft model's kv_lens per batch position so that gen
        # requests can override the target's kv_lens at the start of step 0.
        self._saved_draft_kv_lens = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")

        # Hidden state management buffers (initialized in update_hidden_states step 0)
        self._hs_write_buffer = None
        self._hs_read_map = None
        self._accumulated_hs = None
        self._step0_hs = None
        self._hs_dim = None

    # ---- Overridden dispatch methods ----

    def _forward_draft_loop(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_model,
        draft_kv_cache_manager,
        num_contexts,
        num_gens,
        batch_size,
        num_accepted_tokens,
        original_all_rank_num_tokens,
        resource_manager,
    ):
        """Dispatch to dynamic tree draft loop."""
        return self._forward_dynamic_tree_draft_loop(
            inputs,
            attn_metadata,
            spec_metadata,
            draft_model,
            draft_kv_cache_manager,
            num_contexts,
            num_gens,
            batch_size,
            num_accepted_tokens,
            original_all_rank_num_tokens,
            resource_manager,
        )

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        """Override to add accepted_draft_tokens_indices to output."""
        # Eagerly initialize spec_tree_manager so it is available for
        # sample_and_accept_draft_tokens (called inside super().forward()).
        if self.spec_tree_manager is None and resource_manager is not None:
            from ..pyexecutor.resource_manager import ResourceManagerType

            spec_rm = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER
            )
            if spec_rm is not None and hasattr(spec_rm, "spec_tree_manager"):
                self.spec_tree_manager = spec_rm.spec_tree_manager

        output = super().forward(
            input_ids,
            position_ids,
            hidden_states,
            logits,
            attn_metadata,
            spec_metadata,
            draft_model,
            resource_manager,
        )
        if hasattr(self, "_accepted_draft_indices_tensor"):
            output["accepted_draft_tokens_indices"] = self._accepted_draft_indices_tensor
        return output

    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        """Override to handle dynamic tree verification."""
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        if num_gens > 0:
            return self._sample_and_accept_dynamic_tree(
                logits, attn_metadata, spec_metadata, batch_size, num_contexts, num_gens
            )

        # Context-only: sample target token, no draft verification.
        max_total = spec_metadata.max_total_draft_tokens
        accepted_tokens = torch.zeros(
            (batch_size, max_total + 1), dtype=torch.int, device=logits.device
        )
        num_accepted_tokens = torch.ones(batch_size, dtype=torch.int, device=logits.device)
        target_tokens = torch.argmax(logits, dim=-1)
        accepted_tokens[:num_contexts, 0] = target_tokens[:num_contexts]
        return accepted_tokens, num_accepted_tokens

    def prepare_1st_drafter_inputs(
        self,
        input_ids,
        position_ids,
        hidden_states,
        accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        """Override to use tree-topology target tokens for gen requests.

        The base class uses accepted_tokens which has positions 0..n_accepted
        overwritten with path-ordered tokens.  But hidden_states and the tree
        attention mask are in tree-topology order, so input_ids must also be
        in tree-topology order.  Use _tree_topology_target_tokens saved during
        verification instead.
        """
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        num_tokens = input_ids.shape[0]

        # prepare hidden states (same as base)
        hidden_size_up = spec_metadata.hidden_size * len(spec_metadata.layers_to_capture)
        hidden_states = spec_metadata.hidden_states[:num_tokens, :hidden_size_up]
        hidden_states = draft_model.apply_eagle3_fc(hidden_states)

        # context (same as base)
        input_ids_ctx = self._prepare_context_input_ids(
            input_ids,
            attn_metadata.num_ctx_tokens,
            spec_metadata.gather_ids,
            accepted_tokens,
            num_contexts,
        )

        # generation: use tree-topology tokens when available
        if num_gens > 0 and self._tree_topology_target_tokens is not None:
            input_ids_gen = self._tree_topology_target_tokens[:num_gens].flatten()
        else:
            # Warmup / linear fallback
            input_ids_gen = accepted_tokens[num_contexts:, :].flatten()

        input_ids = torch.concat([input_ids_ctx, input_ids_gen], dim=0)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }

    # ---- Dynamic tree draft loop ----

    def _forward_dynamic_tree_draft_loop(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_model,
        draft_kv_cache_manager,
        num_contexts,
        num_gens,
        batch_size,
        num_accepted_tokens,
        original_all_rank_num_tokens,
        resource_manager,
    ):
        """Dynamic tree draft loop with growing context.

        Matches the two-model DynamicTreeDraftingLoopWrapper structure:
        - Step 0: initial forward, sample K tokens, update buffers
        - prepare_for_generation(0): set up KV/position/mask
        - Steps 1+: growing context forward, sample, update, prepare
        """
        K = self.dynamic_tree_max_topK

        # Lazily get spec_tree_manager
        if self.spec_tree_manager is None and resource_manager is not None:
            from ..pyexecutor.resource_manager import ResourceManagerType

            spec_rm = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER
            )
            if spec_rm is not None and hasattr(spec_rm, "spec_tree_manager"):
                self.spec_tree_manager = spec_rm.spec_tree_manager

        spec_tree_manager = self.spec_tree_manager

        # === Step 0: Initial forward ===
        # Extract only accepted path tokens (one-model inputs contain ALL tree
        # tokens in tree-topology order; must gather accepted path only so draft
        # KV matches two-model behavior).
        gen_tokens_per_req = spec_metadata.max_total_draft_tokens + 1

        _used_accepted_path = False
        _has_acc_idx = getattr(self, "_accepted_draft_indices_tensor", None)
        if _has_acc_idx is not None and num_gens > 0:
            _used_accepted_path = True
            acc_gather_list = []

            # Context tokens: keep all
            if attn_metadata.num_ctx_tokens > 0:
                acc_gather_list.append(
                    torch.arange(attn_metadata.num_ctx_tokens, device="cuda", dtype=torch.long)
                )

            # Gen requests: root + accepted draft positions
            for g_idx in range(num_gens):
                req_idx = num_contexts + g_idx
                n_acc = int(num_accepted_tokens[req_idx].item())
                gen_start = attn_metadata.num_ctx_tokens + g_idx * gen_tokens_per_req

                if n_acc <= 1:
                    path_pos = torch.tensor([0], device="cuda", dtype=torch.long)
                else:
                    draft_pos = self._accepted_draft_indices_tensor[req_idx, : n_acc - 1].long() + 1
                    path_pos = torch.cat(
                        [torch.tensor([0], device="cuda", dtype=torch.long), draft_pos]
                    )

                acc_gather_list.append(gen_start + path_pos)

            acc_indices = torch.cat(acc_gather_list)

            # Reconstruct inputs with accepted-path-only tokens
            inputs = {
                "input_ids": inputs["input_ids"][acc_indices],
                "position_ids": inputs["position_ids"][acc_indices],
                "hidden_states": inputs["hidden_states"][acc_indices],
                "attn_metadata": attn_metadata,
                "spec_metadata": spec_metadata,
            }

            # Update seq_lens to num_accepted per gen request
            for g_idx in range(num_gens):
                req_idx = num_contexts + g_idx
                n_acc = int(num_accepted_tokens[req_idx].item())
                attn_metadata._seq_lens[req_idx] = n_acc
                attn_metadata._seq_lens_cuda[req_idx] = n_acc
            attn_metadata.on_update()

            # Compute gather_ids from new layout
            cumsum = torch.cumsum(attn_metadata.seq_lens_cuda[:batch_size], dim=0)
            gather_ids = cumsum.long() - 1
            _orig_gather_ids = acc_indices[gather_ids]
        else:
            # Fallback: original gather_ids (warmup/context-only)
            start_ids_gen = (
                spec_metadata.batch_indices_cuda[:num_gens] * gen_tokens_per_req
            ).long()
            last_pos = num_accepted_tokens[num_contexts:] - 1
            gather_ids_gen = start_ids_gen + last_pos + attn_metadata.num_ctx_tokens
            gather_ids = torch.concat(
                [spec_metadata.gather_ids[:num_contexts], gather_ids_gen], dim=0
            )
            _orig_gather_ids = gather_ids

        # Override kv_lens for gen requests with draft KV cache values
        # (must happen AFTER accepted-path extraction updates seq_lens).
        if num_gens > 0:
            gen_seq_lens = attn_metadata.seq_lens_cuda[num_contexts:batch_size]
            saved_dkv = self._saved_draft_kv_lens[num_contexts:batch_size]
            has_saved = saved_dkv > 0
            if has_saved.any():
                attn_metadata.kv_lens_cuda[num_contexts:batch_size][has_saved] = (
                    saved_dkv[has_saved] + gen_seq_lens[has_saved]
                )

        if original_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        # Set up spec_decoding for multi-token gen requests so the attention
        # kernel processes ALL accepted tokens (not just 1 per gen request).
        if _used_accepted_path and num_gens > 0:
            gen_sl = attn_metadata.seq_lens_cuda[num_contexts:batch_size]

            attn_metadata.spec_decoding_generation_lengths[num_contexts:batch_size] = gen_sl

            # position_offsets: causal [0, 1, ..., n_acc-1]
            tokens_per_req = spec_metadata.max_total_draft_tokens + 1
            total_po_size = attn_metadata.spec_decoding_position_offsets.shape[0]
            max_reqs = total_po_size // tokens_per_req
            pos_2d = attn_metadata.spec_decoding_position_offsets.view(max_reqs, tokens_per_req)
            max_gl = int(gen_sl.max().item())
            causal_offs = torch.arange(max_gl, device="cuda", dtype=torch.int32)
            for g_idx in range(num_gens):
                req_idx = num_contexts + g_idx
                n = int(gen_sl[g_idx].item())
                pos_2d[req_idx, :n] = causal_offs[:n]

            # packed_mask: causal lower-triangular (n_acc <= 32 so word 0 suffices)
            attn_metadata.spec_decoding_packed_mask[num_contexts:batch_size].fill_(0)
            for g_idx in range(num_gens):
                req_idx = num_contexts + g_idx
                n = int(gen_sl[g_idx].item())
                for t in range(n):
                    attn_metadata.spec_decoding_packed_mask[req_idx, t, 0] = (1 << (t + 1)) - 1

            attn_metadata.use_spec_decoding = True
        else:
            attn_metadata.use_spec_decoding = False

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            hidden_states, hidden_states_to_save = draft_model.model(**inputs)

            # Use draft model's pre-norm output as step0_hs (matches two-model
            # where Eagle3DecoderLayer writes MLP_out + residual to shared buffer).
            hs_dim = spec_metadata.hidden_size
            step0_hs = hidden_states_to_save[gather_ids].clone()

            logits = draft_model.logits_processor(
                hidden_states[gather_ids], draft_model.lm_head, attn_metadata, True
            )

            new_draft_tokens, new_draft_scores = self.sample(logits, K, draft_model=draft_model)

            previous_draft_scores = self.update_draft_tokens_and_scores(
                cur_draft_idx=0,
                new_draft_tokens=new_draft_tokens,
                new_draft_scores=new_draft_scores,
                previous_draft_scores=None,
                attn_metadata=attn_metadata,
                spec_tree_manager=spec_tree_manager,
            )

            self.update_hidden_states(
                cur_draft_idx=0,
                batch_size=batch_size,
                step0_hs=step0_hs,
                hs_dim=hs_dim,
                hidden_states_to_save=None,
                selected_parents=None,
            )

            self.prepare_for_generation(
                cur_draft_idx=0,
                attn_metadata=attn_metadata,
                inputs=inputs,
                gather_ids=gather_ids,
                batch_size=batch_size,
                num_contexts=num_contexts,
                num_gens=num_gens,
                num_accepted_tokens=num_accepted_tokens,
                original_all_rank_num_tokens=original_all_rank_num_tokens,
            )

            for layer_idx in range(1, self.max_draft_len):
                num_tokens_per_req = layer_idx * K

                if original_all_rank_num_tokens is not None:
                    if spec_metadata.all_rank_num_seqs is not None:
                        attn_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_seqs

                # Growing context: process ALL accumulated tokens
                num_infer_tokens = batch_size * num_tokens_per_req

                inp_hs = self._accumulated_hs[:batch_size, :num_tokens_per_req, :].reshape(
                    num_infer_tokens, -1
                )
                inp_ids = self.draft_tokens_buffer[:num_infer_tokens].to(torch.int32)
                inp_pos = self.position_ids_buffer[:num_infer_tokens]
                inputs = {
                    "input_ids": inp_ids,
                    "position_ids": inp_pos,
                    "hidden_states": inp_hs,
                    "attn_metadata": attn_metadata,
                    "spec_metadata": spec_metadata,
                }

                hidden_states, hidden_states_to_save = draft_model.model(**inputs)

                # Take last K logits per request
                hs_reshaped = hidden_states.reshape(batch_size, num_tokens_per_req, -1)
                selected_hs = hs_reshaped[:, -K:, :].reshape(batch_size * K, -1)
                logits = draft_model.logits_processor(
                    selected_hs, draft_model.lm_head, attn_metadata, True
                )

                new_draft_tokens, new_draft_scores = self.sample(logits, K, draft_model=draft_model)

                # Reshape for update: [batch_size, K, K]
                new_draft_tokens = new_draft_tokens.reshape(batch_size, K, K)
                new_draft_scores = new_draft_scores.reshape(batch_size, K, K)

                previous_draft_scores = self.update_draft_tokens_and_scores(
                    cur_draft_idx=layer_idx,
                    new_draft_tokens=new_draft_tokens,
                    new_draft_scores=new_draft_scores,
                    previous_draft_scores=previous_draft_scores,
                    attn_metadata=attn_metadata,
                    spec_tree_manager=spec_tree_manager,
                )

                self.update_hidden_states(
                    cur_draft_idx=layer_idx,
                    batch_size=batch_size,
                    step0_hs=None,
                    hs_dim=hs_dim,
                    hidden_states_to_save=hidden_states_to_save,
                    selected_parents=self._last_selected_parents,
                )

                self.prepare_for_generation(
                    cur_draft_idx=layer_idx,
                    attn_metadata=attn_metadata,
                    inputs=None,
                    gather_ids=None,
                    batch_size=batch_size,
                    num_contexts=num_contexts,
                    num_gens=num_gens,
                    num_accepted_tokens=None,
                    original_all_rank_num_tokens=None,
                )

        # Resample final tokens and build tree
        real_draft_tokens, topk_score_indices = self.resampling_final_draft_tokens(batch_size)

        # Build the dynamic tree structure
        if spec_tree_manager is not None:
            self.tree_ops_converter.build_dynamic_tree(
                history_draft_tokens_parent_buffer=self.history_draft_tokens_parent_buffer[
                    :batch_size
                ],
                topk_score_indices=topk_score_indices,
                tree_mask=spec_tree_manager.spec_dec_packed_mask[:batch_size],
                positions=spec_tree_manager.spec_dec_position_offsets[:batch_size],
                retrieve_index=spec_tree_manager.retrieve_index[:batch_size],
                retrieve_next_token=spec_tree_manager.retrieve_next_token[:batch_size],
                retrieve_next_sibling=spec_tree_manager.retrieve_next_sibling[:batch_size],
                use_packed_mask=True,
            )

        return real_draft_tokens.to(torch.int32)

    # ---- Dynamic tree verification ----

    def _sample_and_accept_dynamic_tree(
        self, logits, attn_metadata, spec_metadata, batch_size, num_contexts, num_gens
    ):
        """Dynamic tree verification using CUDA kernel."""
        max_total = spec_metadata.max_total_draft_tokens
        N = max_total + 1  # includes root

        # Allocate return buffers
        accepted_tokens = torch.empty(
            (batch_size, max_total + 1), dtype=torch.int, device=logits.device
        )
        num_accepted_tokens = torch.ones(batch_size, dtype=torch.int, device=logits.device)
        # Accepted draft indices tensor for KV cache rewind [batch_size, max_total]
        self._accepted_draft_indices_tensor = torch.full(
            (batch_size, max_total), -1, dtype=torch.int32, device=logits.device
        )

        # Context requests: sample token
        if num_contexts > 0:
            ctx_tokens = self._sample_tokens_for_batch(
                logits[:num_contexts], spec_metadata, 0, num_contexts
            )
            accepted_tokens[:num_contexts, 0] = ctx_tokens

        # Generation requests: tree verification
        if num_gens > 0:
            spec_tree_manager = self.spec_tree_manager

            # Sample target tokens from logits (greedy)
            target_tokens = torch.argmax(logits, dim=-1)  # [num_tokens]

            # Build target_predict: [num_trees, N]
            gen_target = target_tokens[num_contexts:]
            gen_target = gen_target.reshape(num_gens, N).to(dtype=torch.int64)

            # Build candidates: [num_trees, N] with draft tokens
            num_trees = spec_tree_manager.retrieve_index.shape[0]
            candidates = torch.zeros(num_trees, N, dtype=torch.int64, device="cuda")

            # Get draft tokens for each gen request
            for g_idx in range(num_gens):
                req_slot = g_idx
                draft_toks = spec_metadata.draft_tokens[g_idx * max_total : (g_idx + 1) * max_total]
                candidates[req_slot, 1:] = draft_toks.to(torch.int64)

            # Set root tokens
            target_predict = torch.zeros(num_trees, N, dtype=torch.int64, device="cuda")
            for g_idx in range(num_gens):
                req_slot = g_idx
                target_predict[req_slot, :N] = gen_target[g_idx, :N]
            candidates[:, 0] = target_predict[:, 0]

            # Fill ALL accepted_tokens positions with target predictions
            for g_idx in range(num_gens):
                req_idx = num_contexts + g_idx
                accepted_tokens[req_idx, :N] = gen_target[g_idx, :N].to(torch.int32)

            # Call verification kernel
            _, accept_index, accept_token_num = torch.ops.trtllm.verify_dynamic_tree_greedy_op(
                candidates,
                spec_tree_manager.retrieve_index,
                spec_tree_manager.retrieve_next_token,
                spec_tree_manager.retrieve_next_sibling,
                target_predict,
                self.max_draft_len + 1,
            )

            # Process results for each gen request
            for g_idx in range(num_gens):
                req_idx = num_contexts + g_idx
                req_slot = g_idx
                n_accepted = int(accept_token_num[req_slot].item())
                num_accepted_tokens[req_idx] = n_accepted + 1  # +1 for root

                # Fill accepted_tokens in order
                for j in range(n_accepted + 1):
                    step = int(accept_index[req_slot, j].item())
                    accepted_tokens[req_idx, j] = gen_target[g_idx, step]

                # Store accepted draft indices for KV cache rewind
                if n_accepted > 0:
                    self._accepted_draft_indices_tensor[req_idx, :n_accepted] = (
                        accept_index[req_slot, 1 : n_accepted + 1] - 1
                    ).to(torch.int32)

            # Store tree-topology position of last accepted token
            self._last_accepted_tree_pos = torch.zeros(num_gens, dtype=torch.long, device="cuda")
            for g_idx in range(num_gens):
                req_slot = g_idx
                n_acc = int(accept_token_num[req_slot].item())
                if n_acc > 0:
                    self._last_accepted_tree_pos[g_idx] = accept_index[req_slot, n_acc].long()
                else:
                    self._last_accepted_tree_pos[g_idx] = 0

            # Save tree-topology target tokens for draft model input
            self._tree_topology_target_tokens = gen_target.to(torch.int32)

        num_accepted_tokens = self._apply_force_accepted_tokens(num_accepted_tokens, num_contexts)

        return accepted_tokens, num_accepted_tokens

    # ---- Dynamic tree helper methods (matching two-model naming) ----

    def sample(self, logits: torch.Tensor, max_top_k: int, draft_model=None) -> torch.Tensor:
        """TopK sampling with log softmax for dynamic tree."""
        last_p = self.logsoftmax(logits)
        topk_values, topk_indices = torch.topk(last_p, k=max_top_k, dim=-1)
        # Apply draft-to-target vocab mapping if the draft model has it
        if draft_model is not None and hasattr(draft_model.model, "d2t"):
            d2t = draft_model.model.d2t.data
            topk_indices = topk_indices + d2t[topk_indices]
        return topk_indices, topk_values

    def update_draft_tokens_and_scores(
        self,
        cur_draft_idx: int,
        new_draft_tokens: torch.Tensor,
        new_draft_scores: torch.Tensor,
        previous_draft_scores: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_tree_manager: "SpecTreeManager",
    ):
        """Update draft tokens and scores, write contiguously to buffer."""
        return_draft_scores = None
        batch_size = attn_metadata.num_seqs
        K = self.dynamic_tree_max_topK
        if cur_draft_idx == 0:
            new_draft_scores = new_draft_scores.reshape(batch_size, K)

            num_tokens_layer0 = batch_size * K
            self.draft_tokens_buffer[:num_tokens_layer0] = new_draft_tokens.reshape(-1)

            self.history_draft_tokens_buffer[:batch_size, :K] = new_draft_tokens.reshape(
                batch_size, K
            )
            self.history_score_buffer[:batch_size, :K] = new_draft_scores[:, :]

            # Initialize parent buffer: -1 for root, 0..K-1 for first layer
            self.history_draft_tokens_parent_buffer[:batch_size, : K + 1] = (
                torch.arange(-1, K, device="cuda", dtype=torch.int32)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            self.prepare_tree_mask_and_position_offset(
                cur_draft_idx, attn_metadata, spec_tree_manager, None
            )

            return_draft_scores = new_draft_scores
        else:
            new_draft_tokens = new_draft_tokens.reshape(batch_size, K * K)

            # Accumulate scores from previous layer
            new_draft_scores = new_draft_scores + previous_draft_scores.unsqueeze(2)
            new_draft_scores = new_draft_scores.reshape(batch_size, K * K)

            # Select best K from K*K candidates
            topk_values, topk_indices = torch.topk(new_draft_scores, k=K, dim=-1)
            real_draft_tokens = torch.gather(new_draft_tokens, dim=1, index=topk_indices)
            num_tokens_previous_layer = cur_draft_idx * K
            num_tokens_current_layer = (cur_draft_idx + 1) * K
            old_tokens = self.draft_tokens_buffer[: batch_size * num_tokens_previous_layer].reshape(
                batch_size, num_tokens_previous_layer
            )
            self.draft_tokens_buffer[: batch_size * num_tokens_current_layer] = torch.cat(
                [old_tokens, real_draft_tokens], dim=1
            ).reshape(-1)

            # Save all K*K candidates to history buffers
            write_history_start_offset = K + (cur_draft_idx - 1) * K * K
            write_history_end_offset = write_history_start_offset + K * K
            self.history_draft_tokens_buffer[
                :batch_size, write_history_start_offset:write_history_end_offset
            ] = new_draft_tokens
            self.history_score_buffer[
                :batch_size, write_history_start_offset:write_history_end_offset
            ] = new_draft_scores

            # Determine which parents were selected
            selected_parents = topk_indices // K
            self._last_selected_parents = selected_parents
            self.prepare_tree_mask_and_position_offset(
                cur_draft_idx, attn_metadata, spec_tree_manager, selected_parents
            )

            # Update parent buffer for next layer
            if cur_draft_idx < self.max_draft_len - 1:
                next_layer_start = cur_draft_idx * K + 1
                next_layer_end = next_layer_start + K
                parents_relative_indices = topk_indices + K**2 * (cur_draft_idx - 1) + K
                self.history_draft_tokens_parent_buffer[
                    :batch_size, next_layer_start:next_layer_end
                ] = parents_relative_indices

            return_draft_scores = topk_values
        return return_draft_scores

    def resampling_final_draft_tokens(self, batch_size: int):
        """Reconstruct the tree based on history buffers."""
        topk_score_indices = torch.topk(
            self.history_score_buffer[:batch_size, :], k=self._max_total_draft_tokens, dim=-1
        ).indices
        topk_score_indices = torch.sort(topk_score_indices).values

        real_draft_tokens = torch.gather(
            self.history_draft_tokens_buffer[:batch_size, :], dim=1, index=topk_score_indices
        )

        return real_draft_tokens, topk_score_indices

    def prepare_tree_mask_and_position_offset(
        self,
        cur_draft_idx: int,
        attn_metadata: AttentionMetadata,
        spec_tree_manager: SpecTreeManager,
        selected_parents: torch.Tensor = None,
    ):
        """Prepare the mask and position offsets for the next layer."""
        batch_size = attn_metadata.num_seqs
        K = self.dynamic_tree_max_topK
        num_tokens_current_layer = K * (cur_draft_idx + 1)
        num_tokens_previous_layer = K * cur_draft_idx
        if cur_draft_idx == 0:
            attn_metadata.spec_decoding_packed_mask.fill_(0)
            spec_tree_manager.compute_spec_dec_packed_mask(
                self.tree_mask_init_buffer[:batch_size],
                attn_metadata.spec_decoding_packed_mask[:batch_size],
            )
            self.tree_mask_buffer[
                : batch_size * num_tokens_current_layer * num_tokens_current_layer
            ].copy_(self.tree_mask_init_buffer[:batch_size].view(-1))
            attn_metadata.spec_decoding_position_offsets.fill_(0)
            attn_metadata.spec_decoding_generation_lengths[:batch_size] = num_tokens_current_layer
        else:
            num_parent_mask = batch_size * cur_draft_idx * K * cur_draft_idx * K
            parent_mask = self.tree_mask_buffer[:num_parent_mask].reshape(
                batch_size, cur_draft_idx * K, cur_draft_idx * K
            )

            selected_parents_expanded = selected_parents.unsqueeze(-1).expand(
                batch_size, K, parent_mask.size(-1)
            )
            parent_mask_selected = torch.gather(
                parent_mask[:, -K:, :], dim=1, index=selected_parents_expanded
            )
            current_mask = torch.cat(
                [parent_mask_selected, self.tree_mask_init_buffer[:batch_size]], dim=2
            )
            mask_padding = self.tree_mask_padding_zeros[
                :batch_size, :num_tokens_previous_layer, :num_tokens_current_layer
            ]
            current_mask = torch.cat([mask_padding, current_mask], dim=1)
            spec_tree_manager.compute_spec_dec_packed_mask(
                current_mask, attn_metadata.spec_decoding_packed_mask[:batch_size]
            )
            self.tree_mask_buffer[
                : batch_size * num_tokens_current_layer * num_tokens_current_layer
            ].copy_(current_mask.view(-1))

            attn_metadata.spec_decoding_generation_lengths[:batch_size] = num_tokens_current_layer

            previous_position_offsets = attn_metadata.spec_decoding_position_offsets[
                : batch_size * num_tokens_previous_layer
            ]
            previous_position_offsets = previous_position_offsets.view(
                batch_size, num_tokens_previous_layer
            )
            new_position_offsets = torch.cat(
                [previous_position_offsets, previous_position_offsets[:, -K:] + 1], dim=1
            )
            attn_metadata.spec_decoding_position_offsets[
                : batch_size * num_tokens_current_layer
            ] = new_position_offsets.reshape(-1)

    def prepare_for_generation(
        self,
        cur_draft_idx: int,
        attn_metadata: AttentionMetadata,
        inputs,
        gather_ids,
        batch_size: int,
        num_contexts: int,
        num_gens: int,
        num_accepted_tokens,
        original_all_rank_num_tokens,
    ):
        """Set up attn_metadata for the subsequent drafter layer.

        Matches two-model prepare_for_generation() structure:
        - Step 0: position IDs, seq_lens, KV rewind, host_request_types, save draft kv_lens
        - Steps 1+: extend position IDs, update seq_lens, increment kv_lens
        """
        K = self.dynamic_tree_max_topK
        num_tokens_current_layer = K * (cur_draft_idx + 1)

        if cur_draft_idx == 0:
            # Position IDs: base_pos + 1, replicated K times per batch
            base_pos = inputs["position_ids"][gather_ids] + 1
            self.position_ids_buffer[: batch_size * K] = (
                base_pos.unsqueeze(1).expand(-1, K).reshape(-1)
            )

            # KV cache: rewind to stable state then pre-add K
            seq_lens = attn_metadata.seq_lens_cuda[:batch_size].clone()
            attn_metadata._seq_lens[:batch_size].fill_(K)
            attn_metadata._seq_lens_cuda[:batch_size].fill_(K)
            attn_metadata.on_update()

            if inputs["attn_metadata"].kv_cache_manager is not None:
                attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)
                attn_metadata.num_contexts = 0

            if hasattr(attn_metadata, "kv_lens_cuda"):
                # Save draft kv_lens for ALL requests AFTER step 0 forward and
                # BEFORE rewind (captures correct persistent draft kv_lens).
                self._saved_draft_kv_lens[:batch_size].copy_(
                    attn_metadata.kv_lens_cuda[:batch_size]
                )

                # KV rewind: gen requests only (context needs no rewind).
                if num_gens > 0:
                    attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                        seq_lens[num_contexts:batch_size]
                        - num_accepted_tokens[num_contexts:batch_size]
                    )
                attn_metadata.kv_lens_cuda[:batch_size] += K

            attn_metadata.use_spec_decoding = True

        else:
            # Position IDs: append prev[-K:] + 1
            num_tokens_previous_layer = cur_draft_idx * K
            prev_pos = self.position_ids_buffer[: batch_size * num_tokens_previous_layer].view(
                batch_size, num_tokens_previous_layer
            )
            new_pos = torch.cat([prev_pos, prev_pos[:, -K:] + 1], dim=1)
            self.position_ids_buffer[: batch_size * num_tokens_current_layer] = new_pos.reshape(-1)

            # Growing seq_lens and pre-increment kv_lens
            attn_metadata._seq_lens[:batch_size].fill_(num_tokens_current_layer)
            attn_metadata._seq_lens_cuda[:batch_size].fill_(num_tokens_current_layer)
            attn_metadata.on_update()
            attn_metadata.kv_lens_cuda[:batch_size] += K

    def update_hidden_states(
        self,
        cur_draft_idx: int,
        batch_size: int,
        step0_hs,
        hs_dim: int,
        hidden_states_to_save=None,
        selected_parents=None,
    ):
        """Manage hidden states for the growing context pattern.

        One-model uses accumulated_hs, hs_write_buffer, hs_read_map to replicate
        the two-model's resource-manager-based hidden state tracking:
        - Step 0: initialize buffers from step0_hs (draft model pre-norm at last accepted token)
        - Steps 1+: write prenorm to buffer, set read_map via selected_parents, reconstruct
        """
        K = self.dynamic_tree_max_topK

        if cur_draft_idx == 0:
            self._hs_dim = hs_dim

            # hs_write_buffer: stores prenorm from each growing-context forward
            self._hs_write_buffer = torch.zeros(
                batch_size,
                self.max_draft_len * K,
                hs_dim,
                device=step0_hs.device,
                dtype=step0_hs.dtype,
            )

            # hs_read_map: maps each token (beyond first K) to its parent
            # position in hs_write_buffer
            self._hs_read_map = torch.zeros(
                batch_size, self.max_draft_len * K, dtype=torch.long, device=step0_hs.device
            )

            # All K depth-0 tokens share the draft model pre-norm at last
            # accepted token (matches two-model where start_idx reads from
            # resource manager)
            self._accumulated_hs = step0_hs.unsqueeze(1).expand(-1, K, -1).clone()
            self._step0_hs = step0_hs

        else:
            num_tokens_per_req = cur_draft_idx * K

            hs_to_save_reshaped = hidden_states_to_save.reshape(batch_size, num_tokens_per_req, -1)

            # 1) Write current forward's prenorm to write_buffer
            self._hs_write_buffer[:batch_size, :num_tokens_per_req] = hs_to_save_reshaped

            # 2) Set read_map for new K tokens (depth cur_draft_idx)
            parent_offset = (cur_draft_idx - 1) * K
            self._hs_read_map[:batch_size, cur_draft_idx * K : (cur_draft_idx + 1) * K] = (
                parent_offset + selected_parents
            )

            # 3) Construct accumulated_hs: first K = step0_hs, rest from write_buffer
            num_tokens_next = (cur_draft_idx + 1) * K
            new_acc = self._step0_hs.unsqueeze(1).expand(-1, K, -1).clone()
            if num_tokens_next > K:
                read_idx = self._hs_read_map[:batch_size, K:num_tokens_next]
                gathered = torch.gather(
                    self._hs_write_buffer[:batch_size],
                    1,
                    read_idx.unsqueeze(-1).expand(-1, -1, self._hs_dim),
                )
                new_acc = torch.cat([new_acc, gathered], dim=1)
            self._accumulated_hs = new_acc
