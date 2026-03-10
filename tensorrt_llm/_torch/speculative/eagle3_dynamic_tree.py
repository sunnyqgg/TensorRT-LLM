# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

        # 1D buffers: store tokens contiguously
        self.dt_draft_tokens_buffer = torch.zeros(
            (max_batch_size * max_total_draft_tokens), dtype=torch.int64, device="cuda"
        )
        self.dt_position_ids_buffer = torch.zeros(
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
        # Context requests save their context_len here; gen requests load it.
        # This is the one-model equivalent of two-model's save_metadata_state
        # which saves/restores draft model's own kv_lens around growing context.
        self._saved_draft_kv_lens = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")

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
        """Dispatch to dynamic tree draft loop.

        Unlike the buggy early-return approach, this always runs the full
        dynamic tree draft loop — even for context-only batches (num_gens==0).
        This produces real draft tokens (not zeros) so the first generation
        iteration processes meaningful candidates, matching two-model behavior.
        """
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

        Matches the two-model DynamicTreeDraftingLoopWrapper approach:
        - Step 0: initial forward, sample K tokens, set up KV/position/mask
        - Steps 1+: growing context (reprocess all accumulated tokens),
          tree attention mask, take last K logits for sampling
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
        previous_draft_scores = None

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            for i in range(self.max_draft_len):
                if i == 0:
                    # === Step 0: Initial forward ===
                    # FIX (Bug 4): Extract only accepted path tokens.
                    # In the two-model, draft step 0 only processes the
                    # accepted path (num_accepted+1 tokens). The one-model
                    # inputs contain ALL tree tokens in tree-topology order.
                    # Processing all tree tokens creates KV entries in tree
                    # order, but the accepted path may not be at the start.
                    # KV rewind then keeps wrong entries. Fix: gather only
                    # accepted path tokens so draft KV matches two-model.
                    gen_tokens_per_req = spec_metadata.max_total_draft_tokens + 1

                    _used_accepted_path = False
                    _has_acc_idx = getattr(self, "_accepted_draft_indices_tensor", None)
                    if _has_acc_idx is not None and num_gens > 0:
                        _used_accepted_path = True
                        # Build indices to accepted path tokens in flat input
                        acc_gather_list = []

                        # Context tokens: keep all
                        if attn_metadata.num_ctx_tokens > 0:
                            acc_gather_list.append(
                                torch.arange(
                                    attn_metadata.num_ctx_tokens, device="cuda", dtype=torch.long
                                )
                            )

                        # Gen requests: root + accepted draft positions
                        for g_idx in range(num_gens):
                            req_idx = num_contexts + g_idx
                            n_acc = int(num_accepted_tokens[req_idx].item())
                            gen_start = attn_metadata.num_ctx_tokens + g_idx * gen_tokens_per_req

                            if n_acc <= 1:
                                path_pos = torch.tensor([0], device="cuda", dtype=torch.long)
                            else:
                                draft_pos = (
                                    self._accepted_draft_indices_tensor[req_idx, : n_acc - 1].long()
                                    + 1
                                )
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

                        # Update seq_lens: num_accepted per gen request
                        for g_idx in range(num_gens):
                            req_idx = num_contexts + g_idx
                            n_acc = int(num_accepted_tokens[req_idx].item())
                            attn_metadata._seq_lens[req_idx] = n_acc
                            attn_metadata._seq_lens_cuda[req_idx] = n_acc
                        attn_metadata.on_update()

                        # Compute gather_ids from new layout
                        cumsum = torch.cumsum(attn_metadata.seq_lens_cuda[:batch_size], dim=0)
                        gather_ids = cumsum.long() - 1
                        # Map back to original token positions (for
                        # reading raw target prenorm from spec_metadata)
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
                        # gather_ids already in original token positions
                        _orig_gather_ids = gather_ids

                    # FIX (Bug 5 + Bug 10): Override kv_lens for gen
                    # requests with draft KV cache values.  Must happen
                    # AFTER accepted-path extraction updates seq_lens.
                    #
                    # prepare() pre-incremented kv_lens = cached + seq_lens
                    # for the TARGET.  The draft cache has a different
                    # number of cached entries (_saved_draft_kv_lens).
                    # Correct formula: kv_lens = saved + gen_seq_lens.
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

                    # FIX (Bug 15): Step 0 spec_decoding for multi-token
                    # gen requests.
                    #
                    # When n_acc > 1 accepted tokens per gen request, the
                    # C++ attention kernel must process ALL n_acc tokens.
                    # Without spec_decoding, the kernel processes only 1
                    # token per gen request (standard generation), producing
                    # ZERO attention output for tokens 1+ and wrong logits
                    # at gather_ids (the last accepted token).
                    #
                    # In the accepted-path extraction case, we set proper
                    # spec_decoding params (generation_lengths, position_
                    # offsets, packed_mask) for causal attention among the
                    # accepted tokens. In the fallback/warmup case, we
                    # keep spec_decoding disabled for safety (stale tensors
                    # from target verification would cause OOB access).
                    if _used_accepted_path and num_gens > 0:
                        gen_sl = attn_metadata.seq_lens_cuda[num_contexts:batch_size]

                        # generation_lengths = n_acc per gen request
                        attn_metadata.spec_decoding_generation_lengths[num_contexts:batch_size] = (
                            gen_sl
                        )

                        # position_offsets: causal [0, 1, ..., n_acc-1]
                        tokens_per_req = spec_metadata.max_total_draft_tokens + 1
                        total_po_size = attn_metadata.spec_decoding_position_offsets.shape[0]
                        max_reqs = total_po_size // tokens_per_req
                        pos_2d = attn_metadata.spec_decoding_position_offsets.view(
                            max_reqs, tokens_per_req
                        )
                        max_gl = int(gen_sl.max().item())
                        causal_offs = torch.arange(max_gl, device="cuda", dtype=torch.int32)
                        for g_idx in range(num_gens):
                            req_idx = num_contexts + g_idx
                            n = int(gen_sl[g_idx].item())
                            pos_2d[req_idx, :n] = causal_offs[:n]

                        # packed_mask: causal lower-triangular
                        # Token t sees tokens [0..t] via bit mask in word 0
                        # (n_acc <= max_draft_len+1 <= 32 so word 0 suffices)
                        attn_metadata.spec_decoding_packed_mask[num_contexts:batch_size].fill_(0)
                        for g_idx in range(num_gens):
                            req_idx = num_contexts + g_idx
                            n = int(gen_sl[g_idx].item())
                            for t in range(n):
                                attn_metadata.spec_decoding_packed_mask[req_idx, t, 0] = (
                                    1 << (t + 1)
                                ) - 1

                        attn_metadata.use_spec_decoding = True
                    else:
                        attn_metadata.use_spec_decoding = False

                    hidden_states, hidden_states_to_save = draft_model.model(**inputs)

                    # FIX (Bug 16): Use draft model's pre-norm output as
                    # step0_hs.
                    #
                    # In the 2M, the DRAFT model's Eagle3DecoderLayer calls
                    # maybe_capture_hidden_states(layer=0, MLP_out, residual)
                    # which writes MLP_out + residual to the shared buffer at
                    # columns [0:4096], OVERWRITING the TARGET model's layer 1
                    # data. Step 1's get_hidden_states(is_first_draft=False)
                    # reads these columns, getting the DRAFT model's output.
                    #
                    # Eagle3DraftModel.forward() returns (post_norm, pre_norm)
                    # where pre_norm = MLP_out + residual — exactly what the
                    # 2M's maybe_capture_hidden_states stores. So we use
                    # hidden_states_to_save[gather_ids] (norm ~84) as step0_hs
                    # to match the 2M behavior.
                    #
                    # Bug 14 incorrectly used TARGET L1 HS (norm ~1.4) here,
                    # causing a ~60x magnitude mismatch vs the 2M.
                    hs_dim = spec_metadata.hidden_size
                    step0_hs = hidden_states_to_save[gather_ids].clone()  # [batch, hs_dim]

                    logits = draft_model.logits_processor(
                        hidden_states[gather_ids], draft_model.lm_head, attn_metadata, True
                    )

                    new_draft_tokens, new_draft_scores = self.sample_dynamic(
                        logits, K, draft_model=draft_model
                    )

                    # Update history buffers and tree mask
                    previous_draft_scores = self.dt_update_draft_tokens_and_scores(
                        cur_draft_idx=0,
                        new_draft_tokens=new_draft_tokens,
                        new_draft_scores=new_draft_scores,
                        previous_draft_scores=None,
                        attn_metadata=attn_metadata,
                        spec_tree_manager=spec_tree_manager,
                    )

                    # === Transition to growing context mode ===
                    # Replicate two-model resource manager hidden state pattern:
                    # - Depth-0 tokens read TARGET L1 HS (from resource mgr)
                    # - Depth-d tokens read draft prenorm from depth d-1
                    #   (via write_buffer storing draft model prenorm output,
                    #    which is now at reasonable scale since the initial
                    #    residual comes from target HS, not draft prenorm)

                    # hs_write_buffer: stores prenorm from each GC forward.
                    # Positions get overwritten by each step (matching 2M where
                    # write_indices[0:K] = [start+1,...,start+K] are reused).
                    hs_write_buffer = torch.zeros(
                        batch_size,
                        self.max_draft_len * K,
                        hs_dim,
                        device=step0_hs.device,
                        dtype=step0_hs.dtype,
                    )

                    # hs_read_map: maps each token (beyond first K) to its
                    # parent's position in hs_write_buffer. Set once per depth
                    # via selected_parents, then never changed (matching 2M where
                    # read_indices are set by update_hidden_states_indices and
                    # the values at those positions change via overwrites).
                    hs_read_map = torch.zeros(
                        batch_size, self.max_draft_len * K, dtype=torch.long, device=step0_hs.device
                    )

                    # Initial accumulated_hs: all K depth-0 tokens share the
                    # TARGET L1 HS at the last accepted token (matches
                    # two-model where start_idx reads from resource manager).
                    accumulated_hs = (
                        step0_hs.unsqueeze(1).expand(-1, K, -1).clone()
                    )  # [batch_size, K, hs_dim]

                    # Initialize position buffer (like two-model prepare_for_generation(0))
                    base_pos = inputs["position_ids"][gather_ids] + 1
                    self.dt_position_ids_buffer[: batch_size * K] = (
                        base_pos.unsqueeze(1).expand(-1, K).reshape(-1)
                    )

                    # KV cache: rewind to stable_kv (prompt_len), then pre-add K
                    # Clone seq_lens BEFORE fill overwrites it
                    seq_lens = attn_metadata.seq_lens_cuda[:batch_size].clone()
                    attn_metadata._seq_lens[:batch_size].fill_(K)
                    attn_metadata._seq_lens_cuda[:batch_size].fill_(K)
                    attn_metadata.on_update()

                    if inputs["attn_metadata"].kv_cache_manager is not None:
                        attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)
                        attn_metadata.num_contexts = 0

                    if hasattr(attn_metadata, "kv_lens_cuda"):
                        # FIX (Bug 7): Save draft kv_lens for ALL requests
                        # AFTER step 0 forward and BEFORE rewind. This captures
                        # the correct persistent draft kv_lens:
                        #   context: kv_lens = context_len
                        #   gen: kv_lens = prev_saved + n_acc (accumulated)
                        #
                        # In the two-model, save_metadata_state saves kv_lens
                        # at this point (after step 0 fwd) and restores them
                        # after growing context. The growing context entries
                        # are ephemeral and get overwritten next iteration.
                        #
                        # Previously only context requests were saved, causing
                        # gen requests to load stale kv_lens from the context
                        # phase on every iteration. This meant the draft model
                        # missed KV entries for previously accepted tokens,
                        # producing incorrect attention outputs.
                        self._saved_draft_kv_lens[:batch_size].copy_(
                            attn_metadata.kv_lens_cuda[:batch_size]
                        )

                        # KV rewind matching two-model prepare_for_generation(0).
                        # Two-model formula:
                        #   kv_lens -= seq_lens - num_accepted_draft_tokens - 1
                        # For context requests, 2M sets num_accepted_draft_tokens
                        # = context_len - 1, giving rewind = 0 (keep ALL context
                        # KV entries). For gen requests, rewind removes stale
                        # draft KV entries beyond the accepted path.
                        #
                        # One-model equivalent:
                        #   Context: NO rewind (kv_lens stays at context_len)
                        #   Gen: kv_lens -= seq_lens - num_accepted_tokens
                        #        where num_accepted_tokens = n_draft_accepted + 1
                        #        equivalent to 2M: seq_lens - n_acc - 1
                        #
                        # FIX (Bug 11): Bug 10 applied rewind to ALL requests
                        # including context. But for context, num_accepted_tokens
                        # = 1 (target token only), giving rewind = context_len - 1
                        # which removes almost all context KV entries! The 2M
                        # effectively does 0 rewind for context. Only gen requests
                        # need rewind to remove stale draft KV entries.
                        if num_gens > 0:
                            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                                seq_lens[num_contexts:batch_size]
                                - num_accepted_tokens[num_contexts:batch_size]
                            )
                        attn_metadata.kv_lens_cuda[:batch_size] += K

                        # FIX (Bug 13): Sync HOST kv_lens tensors with GPU.
                        # The C++ attention kernel's plan() reads BOTH:
                        #   - sequence_length = kv_lens_cuda_runtime (GPU)
                        #   - host_past_key_value_lengths = kv_lens_runtime (CPU)
                        # Our code only modified kv_lens_cuda (GPU), leaving
                        # kv_lens_runtime (CPU) stale from the initial prepare().
                        # In the 2M, the draft model has its own attention
                        # metadata where prepare() properly sets both. In the
                        # 1M, we share the target's metadata and must manually
                        # sync the HOST tensor.
                        if hasattr(attn_metadata, "kv_lens_runtime"):
                            attn_metadata.kv_lens_runtime[:batch_size].copy_(
                                attn_metadata.kv_lens_cuda[:batch_size]
                            )
                        if hasattr(attn_metadata, "host_total_kv_lens"):
                            attn_metadata.host_total_kv_lens[0] = 0
                            attn_metadata.host_total_kv_lens[1] = (
                                attn_metadata.kv_lens_cuda[:batch_size].sum().item()
                            )

                    # Growing context uses CAUSAL attention (via packed_mask set
                    # by dt_prepare_tree_mask_and_position_offset).
                    #
                    # FIX (Bug 15): Must set use_spec_decoding=True so the C++
                    # attention kernel reads spec_decoding_generation_lengths to
                    # know how many tokens per generation request. Without this,
                    # the kernel falls back to standard generation (1 token per
                    # request), producing ZERO attention output for tokens 1+.
                    #
                    # In 2M, use_spec_decoding is already True from the draft
                    # engine's initialization. The commented-out line in 2M's
                    # prepare_for_generation (# attn_metadata.use_spec_decoding
                    # = True) is commented out because it's ALREADY True.
                    attn_metadata.use_spec_decoding = True

                else:
                    # === Steps 1+: Growing context ===
                    num_tokens_per_req = i * K

                    if original_all_rank_num_tokens is not None:
                        if spec_metadata.all_rank_num_seqs is not None:
                            attn_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_seqs

                    # Growing context: process ALL accumulated tokens
                    num_infer_tokens = batch_size * num_tokens_per_req

                    # Use accumulated hidden states from previous forward
                    # passes.  After each step, accumulated_hs is updated by
                    # overwriting existing positions with new prenorm (matching
                    # the two-model resource-manager write-back) and extending
                    # with selected parents' prenorm for the new K tokens.
                    inp_hs = accumulated_hs[:batch_size, :num_tokens_per_req, :].reshape(
                        num_infer_tokens, -1
                    )
                    inp_ids = self.dt_draft_tokens_buffer[:num_infer_tokens].to(torch.int32)
                    inp_pos = self.dt_position_ids_buffer[:num_infer_tokens]
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

                    new_draft_tokens, new_draft_scores = self.sample_dynamic(
                        logits, K, draft_model=draft_model
                    )

                    # Reshape for update: [batch_size, K, K]
                    new_draft_tokens = new_draft_tokens.reshape(batch_size, K, K)
                    new_draft_scores = new_draft_scores.reshape(batch_size, K, K)

                    previous_draft_scores = self.dt_update_draft_tokens_and_scores(
                        cur_draft_idx=i,
                        new_draft_tokens=new_draft_tokens,
                        new_draft_scores=new_draft_scores,
                        previous_draft_scores=previous_draft_scores,
                        attn_metadata=attn_metadata,
                        spec_tree_manager=spec_tree_manager,
                    )

                    # Update hidden states using write_buffer + read_map,
                    # replicating the two-model resource manager pattern:
                    #
                    # 1) Write current forward's prenorm to write_buffer
                    #    (overwrites previous values, matching 2M write indices)
                    # 2) Set read_map for new K tokens via selected_parents
                    #    (matching 2M update_hidden_states_indices)
                    # 3) Construct accumulated_hs:
                    #    - First K: step0_hs (STATIC, matching 2M pos `start`)
                    #    - Rest: gather from write_buffer via read_map
                    hs_to_save_reshaped = hidden_states_to_save.reshape(
                        batch_size, num_tokens_per_req, -1
                    )
                    sp = self._last_selected_parents  # [batch_size, K]

                    # 1) Update write_buffer with current prenorm
                    hs_write_buffer[:batch_size, :num_tokens_per_req] = hs_to_save_reshaped

                    # 2) Set read_map for new K tokens (depth i)
                    # Depth i tokens parent to depth i-1 tokens, which are
                    # at write_buffer positions [(i-1)*K : i*K].
                    parent_offset = (i - 1) * K
                    hs_read_map[:batch_size, i * K : (i + 1) * K] = parent_offset + sp

                    # 3) Construct accumulated_hs for next step
                    num_tokens_next = (i + 1) * K
                    # First K: always step0_hs (target L1 HS at last accepted)
                    new_acc = step0_hs.unsqueeze(1).expand(-1, K, -1).clone()
                    # Remaining: gather from write_buffer using read_map
                    if num_tokens_next > K:
                        read_idx = hs_read_map[:batch_size, K:num_tokens_next]
                        gathered = torch.gather(
                            hs_write_buffer[:batch_size],
                            1,
                            read_idx.unsqueeze(-1).expand(-1, -1, hs_dim),
                        )
                        new_acc = torch.cat([new_acc, gathered], dim=1)
                    accumulated_hs = new_acc

                    # Update position buffer
                    num_tokens_current = (i + 1) * K
                    prev_pos = self.dt_position_ids_buffer[: batch_size * num_tokens_per_req].view(
                        batch_size, num_tokens_per_req
                    )
                    new_pos = torch.cat([prev_pos, prev_pos[:, -K:] + 1], dim=1)
                    self.dt_position_ids_buffer[: batch_size * num_tokens_current] = (
                        new_pos.reshape(-1)
                    )

                    # Growing seq_lens and pre-increment kv_lens
                    attn_metadata._seq_lens[:batch_size].fill_(num_tokens_current)
                    attn_metadata._seq_lens_cuda[:batch_size].fill_(num_tokens_current)
                    attn_metadata.on_update()
                    attn_metadata.kv_lens_cuda[:batch_size] += K
                    # Bug 13: Sync HOST kv_lens with GPU
                    if hasattr(attn_metadata, "kv_lens_runtime"):
                        attn_metadata.kv_lens_runtime[:batch_size].copy_(
                            attn_metadata.kv_lens_cuda[:batch_size]
                        )
                    if hasattr(attn_metadata, "host_total_kv_lens"):
                        attn_metadata.host_total_kv_lens[1] = (
                            attn_metadata.kv_lens_cuda[:batch_size].sum().item()
                        )

        # Resample final tokens and build tree
        real_draft_tokens, topk_score_indices = self.dt_resampling_final_draft_tokens(batch_size)

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

            # Store tree-topology position of last accepted token for
            # gather_ids in the next draft loop's step 0.
            self._last_accepted_tree_pos = torch.zeros(num_gens, dtype=torch.long, device="cuda")
            for g_idx in range(num_gens):
                req_slot = g_idx
                n_acc = int(accept_token_num[req_slot].item())
                if n_acc > 0:
                    self._last_accepted_tree_pos[g_idx] = accept_index[req_slot, n_acc].long()
                else:
                    self._last_accepted_tree_pos[g_idx] = 0

            # Save tree-topology target tokens for draft model input.
            # accepted_tokens has positions 0..n_accepted overwritten with
            # path-ordered tokens, which misaligns with hidden_states and
            # the tree attention mask (both in tree-topology order).
            self._tree_topology_target_tokens = gen_target.to(torch.int32)

        num_accepted_tokens = self._apply_force_accepted_tokens(num_accepted_tokens, num_contexts)

        return accepted_tokens, num_accepted_tokens

    # ---- Dynamic tree helper methods ----

    def sample_dynamic(
        self, logits: torch.Tensor, max_top_k: int, draft_model=None
    ) -> torch.Tensor:
        """TopK sampling with log softmax for dynamic tree."""
        last_p = self.logsoftmax(logits)
        topk_values, topk_indices = torch.topk(last_p, k=max_top_k, dim=-1)
        # Apply draft-to-target vocab mapping if the draft model has it
        if draft_model is not None and hasattr(draft_model.model, "d2t"):
            d2t = draft_model.model.d2t.data
            topk_indices = topk_indices + d2t[topk_indices]
        return topk_indices, topk_values

    def dt_update_draft_tokens_and_scores(
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
            self.dt_draft_tokens_buffer[:num_tokens_layer0] = new_draft_tokens.reshape(-1)

            self.history_draft_tokens_buffer[:batch_size, :K] = new_draft_tokens.reshape(
                batch_size, K
            )
            self.history_score_buffer[:batch_size, :K] = new_draft_scores[:, :]

            # Process the parent buffer
            self.history_draft_tokens_parent_buffer[:batch_size, : K + 1] = (
                torch.arange(-1, K, device="cuda", dtype=torch.int32)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            self.dt_prepare_tree_mask_and_position_offset(
                cur_draft_idx, attn_metadata, spec_tree_manager, None
            )

            return_draft_scores = new_draft_scores
        else:
            new_draft_tokens = new_draft_tokens.reshape(batch_size, K * K)

            # Update scores with previous layer's scores
            new_draft_scores = new_draft_scores + previous_draft_scores.unsqueeze(2)
            new_draft_scores = new_draft_scores.reshape(batch_size, K * K)

            # Extract the real draft tokens: topK again
            topk_values, topk_indices = torch.topk(new_draft_scores, k=K, dim=-1)
            real_draft_tokens = torch.gather(new_draft_tokens, dim=1, index=topk_indices)
            num_tokens_previous_layer = cur_draft_idx * K
            num_tokens_current_layer = (cur_draft_idx + 1) * K
            old_tokens = self.dt_draft_tokens_buffer[
                : batch_size * num_tokens_previous_layer
            ].reshape(batch_size, num_tokens_previous_layer)
            self.dt_draft_tokens_buffer[: batch_size * num_tokens_current_layer] = torch.cat(
                [old_tokens, real_draft_tokens], dim=1
            ).reshape(-1)

            # Save to history buffers
            write_history_start_offset = K + (cur_draft_idx - 1) * K * K
            write_history_end_offset = write_history_start_offset + K * K
            self.history_draft_tokens_buffer[
                :batch_size, write_history_start_offset:write_history_end_offset
            ] = new_draft_tokens
            self.history_score_buffer[
                :batch_size, write_history_start_offset:write_history_end_offset
            ] = new_draft_scores

            # Update tree mask
            selected_parents = topk_indices // K
            # Store for hidden states gathering in the draft loop
            self._last_selected_parents = selected_parents
            self.dt_prepare_tree_mask_and_position_offset(
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

    def dt_resampling_final_draft_tokens(self, batch_size: int):
        """Reconstruct the tree based on history buffers."""
        topk_score_indices = torch.topk(
            self.history_score_buffer[:batch_size, :], k=self._max_total_draft_tokens, dim=-1
        ).indices
        topk_score_indices = torch.sort(topk_score_indices).values

        real_draft_tokens = torch.gather(
            self.history_draft_tokens_buffer[:batch_size, :], dim=1, index=topk_score_indices
        )

        return real_draft_tokens, topk_score_indices

    def dt_prepare_tree_mask_and_position_offset(
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
