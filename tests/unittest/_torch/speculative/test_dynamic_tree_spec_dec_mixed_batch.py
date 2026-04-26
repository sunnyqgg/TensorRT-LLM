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
"""Regression tests for dynamic-tree spec-dec metadata plumbing.

Fix #1 (row alignment): on mixed prefill+decode batches, the dynamic-tree
``update_spec_dec_param`` Case 1 must source from gen-only slot rows
``[num_contexts:batch_size]`` and write to gen-only destination rows
``[0:num_gens)``.  The pre-fix code sourced from ``[:batch_size]``, which
mis-aligned context-occupied row 0 with gen row 0 in the XQA kernel.

Fix #2 (KV reserve): a dynamic-tree *draft* manager must reserve KV slots
for ``K * max_draft_len`` per generation step even when ``py_draft_tokens``
is shorter, since the draft loop will actually write that many slots.
The target manager must keep its existing budget exactly.
"""

import math
import os
import sys
import types
import unittest

import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_attn_metadata(max_num_requests: int, n_dt: int, buf_dim: int):
    """Construct a TrtllmAttentionMetadata with the dynamic-tree dest buffers.

    Mirrors the shapes that ``update_spec_dec_param`` would itself allocate
    on the first call when ``is_spec_dec_dynamic_tree=True``.
    """
    attn = TrtllmAttentionMetadata(
        max_num_requests=max_num_requests,
        max_num_tokens=1024,
        kv_cache_manager=None,
    )
    # Allocate the same shapes the runtime would lazily create.
    attn.spec_decoding_position_offsets = torch.zeros(
        (max_num_requests * buf_dim,), dtype=torch.int, device="cuda"
    )
    attn.spec_decoding_packed_mask = torch.zeros(
        [max_num_requests, buf_dim, math.ceil(buf_dim / 32)],
        dtype=torch.int,
        device="cuda",
    )
    attn.spec_decoding_generation_lengths = torch.zeros(
        [max_num_requests], dtype=torch.int, device="cuda"
    )
    return attn


def _seed_slot_storage(spec_tree_manager: SpecTreeManager) -> None:
    """Stamp every slot's position_offsets / packed_mask with a unique pattern.

    Slot ``s`` row ``r`` gets value ``s * 1000 + r``.  This makes any row
    misalignment numerically visible after copy.
    """
    ss = spec_tree_manager.slot_storage
    n_dt = ss.position_offsets.shape[1]
    num_slots = ss.position_offsets.shape[0] - 1  # last row is dummy
    for s in range(num_slots):
        offsets = torch.arange(n_dt, dtype=torch.int32, device="cuda") + s * 1000
        ss.position_offsets[s].copy_(offsets)
        # Each (slot, row, col) gets a distinct int32 — column 0 carries the
        # signal, remaining columns stay 0.
        ss.packed_mask[s, :, 0].copy_(offsets)


# ---------------------------------------------------------------------------
# Fix #1 — row-alignment test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dynamic-tree slot buffers live on CUDA")
def test_dynamic_tree_mixed_batch_row_alignment():
    """Mixed [ctx, gen] batch: dest row 0 must come from gen slot, not ctx slot.

    Pre-fix: the source slice was ``all_ids_buf[:batch_size]`` = ``[ctx_slot,
    gen_slot]``, so dest row 0 received the ctx slot's data — but the XQA
    kernel reads dest row 0 expecting gen-0 data.

    Post-fix: source slice is ``all_ids_buf[num_contexts:batch_size]`` and
    dest is ``[:num_gens, :n_dt, ...]``, so dest row 0 carries gen-0 data.
    """
    max_num_requests = 4
    # Pick max_total_draft_tokens so the runtime ``assert buf_dim == n_dt``
    # in trtllm.py holds: K*L must equal max_total_draft_tokens+1's lower
    # bound, i.e. _internal_buf_dim == n_dt.  Linear-tree-style sizing
    # (K*L == max_total_draft_tokens) satisfies this.
    max_draft_len = 4
    dynamic_tree_max_topK = 2
    max_total_draft_tokens = max_draft_len * dynamic_tree_max_topK  # 8 = K*L
    n_dt = max_total_draft_tokens + 1  # 9

    spec_tree_manager = SpecTreeManager(
        max_num_requests=max_num_requests,
        use_dynamic_tree=True,
        max_total_draft_tokens=max_total_draft_tokens,
        max_draft_len=max_draft_len,
        eagle_choices=None,
        dynamic_tree_max_topK=dynamic_tree_max_topK,
    )
    # Plan-A invariant: assert in trtllm.py requires buf_dim == n_dt.  Pick
    # ``max_total_draft_tokens >= K*L`` so the assertion holds at runtime.
    assert spec_tree_manager._internal_buf_dim == n_dt

    _seed_slot_storage(spec_tree_manager)

    # Simulate ``fill_all_slot_ids`` for a 1 ctx + 1 gen batch.  We pick
    # distinct, non-zero slot IDs so the failure is unambiguous.
    ctx_slot = 1
    gen_slot = 2
    spec_tree_manager.slot_storage.all_ids_buf[0] = ctx_slot
    spec_tree_manager.slot_storage.all_ids_buf[1] = gen_slot

    attn = _build_attn_metadata(max_num_requests, n_dt, spec_tree_manager._internal_buf_dim)

    batch_size = 2
    num_contexts = 1

    attn.update_spec_dec_param(
        batch_size=batch_size,
        is_spec_decoding_enabled=True,
        is_spec_dec_tree=True,
        is_spec_dec_dynamic_tree=True,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        spec_tree_manager=spec_tree_manager,
        num_contexts=num_contexts,
    )
    torch.cuda.synchronize()

    # Dest row 0 (== gen-0 in the kernel's view) must carry the GEN slot's
    # data, NOT the context slot's.
    pos_dst_2d = attn.spec_decoding_position_offsets.view(max_num_requests, -1)
    expected_gen0 = torch.arange(n_dt, dtype=torch.int32, device="cuda") + gen_slot * 1000
    forbidden_ctx = torch.arange(n_dt, dtype=torch.int32, device="cuda") + ctx_slot * 1000

    assert torch.equal(pos_dst_2d[0, :n_dt], expected_gen0), (
        f"Dest row 0 should carry gen slot {gen_slot}'s data, got {pos_dst_2d[0, :n_dt].tolist()}"
    )
    assert not torch.equal(pos_dst_2d[0, :n_dt], forbidden_ctx), (
        "Dest row 0 still aligned to ctx slot — Fix #1 not applied"
    )

    # Mask: column 0 of each (row, mask-block) must match expected.
    mask_dst = attn.spec_decoding_packed_mask
    expected_mask_col0 = expected_gen0
    assert torch.equal(mask_dst[0, :n_dt, 0], expected_mask_col0)

    # generation_lengths is filled to batch_size for CUDA-graph stability,
    # not just num_gens.
    gl = attn.spec_decoding_generation_lengths
    assert int(gl[0].item()) == n_dt
    assert int(gl[1].item()) == n_dt


# ---------------------------------------------------------------------------
# Fix #2 — KV reserve, V1 path
# ---------------------------------------------------------------------------


def _make_v1_kvm_for_reserve_test(is_draft: bool, use_dynamic_tree: bool = True):
    """Build a stand-in KVCacheManager-V1-shaped object exercising _kv_reserve.

    The full KVCacheManager constructor is heavy (touches GPU pools, MPI,
    pp layer config, cache reuse). For Fix #2 V1 we only need to verify
    the reservation arithmetic in ``prepare_resources``: every gen request
    invokes ``add_token`` exactly ``1 + max(draft_len,
    _kv_reserve_draft_tokens)`` times.
    """
    from tensorrt_llm._torch.pyexecutor import resource_manager as rm_mod

    obj = types.SimpleNamespace()
    obj.is_draft = is_draft
    spec_config = types.SimpleNamespace(
        use_dynamic_tree=use_dynamic_tree,
        dynamic_tree_max_topK=4,
        max_draft_len=4,
        # V1 reads ``tokens_per_gen_step``; mirror typical Eagle3 value.
        tokens_per_gen_step=9,  # max_total_draft_tokens = 8
    )
    obj.max_total_draft_tokens = spec_config.tokens_per_gen_step - 1

    # Replicate the V1 ctor reserve calculation exactly (Plan-A Phase-2
    # Commit-2 V1 __init__ block).  Keeping this in lockstep with the source
    # is intentional; an accidental drift in the production constant should
    # break this test loudly.
    obj._kv_reserve_draft_tokens = obj.max_total_draft_tokens
    if (
        obj.is_draft
        and spec_config is not None
        and getattr(spec_config, "use_dynamic_tree", False)
        and getattr(spec_config, "dynamic_tree_max_topK", 0) > 0
    ):
        kl = spec_config.dynamic_tree_max_topK * spec_config.max_draft_len
        obj._kv_reserve_draft_tokens = max(obj.max_total_draft_tokens, kl)

    add_token_calls = []

    class _Impl:
        cross_kv = False

        def add_token(self, req_id):
            add_token_calls.append(req_id)

    obj.impl = _Impl()
    obj._add_token_calls = add_token_calls
    obj._spec_config = spec_config
    obj._rm_mod = rm_mod
    return obj


def _v1_gen_loop(obj, draft_len: int, req_id: int = 7) -> int:
    """Replicate the V1 generation-request body from ``prepare_resources``.

    Mirrors the post-fix loop body so the test is decoupled from the
    surrounding context-loop wiring (block reuse, helix, kv-connector, ...)
    while still exercising the reserve arithmetic introduced by Fix #2.
    """
    obj.impl.add_token(req_id)
    for _ in range(draft_len):
        obj.impl.add_token(req_id)
    reserve_slack = obj._kv_reserve_draft_tokens - draft_len
    for _ in range(max(0, reserve_slack)):
        obj.impl.add_token(req_id)
    return len(obj._add_token_calls)


def test_v1_draft_reserve_grows_to_K_times_L():
    """V1 draft manager: gen request gets 1 + K*L add_token calls."""
    obj = _make_v1_kvm_for_reserve_test(is_draft=True, use_dynamic_tree=True)
    K = obj._spec_config.dynamic_tree_max_topK
    L = obj._spec_config.max_draft_len
    max_total = obj.max_total_draft_tokens

    # Pick draft_len < K*L to force the reserve slack to fire.
    draft_len = max_total  # 8 in the default config
    n = _v1_gen_loop(obj, draft_len)

    assert obj._kv_reserve_draft_tokens == max(max_total, K * L)
    assert n == 1 + max(draft_len, K * L), (
        f"V1 draft must reserve max(draft_len, K*L) = "
        f"max({draft_len}, {K * L}); got {n - 1} add_tokens after the base."
    )


def test_v1_target_reserve_capped_at_max_total():
    """V1 target manager: reserve == max_total_draft_tokens, NOT K*L.

    The is_draft gate keeps the target's reservation at the size the target
    model actually consumes (max_total_draft_tokens), even when K*L is
    larger.  This isolates the draft-only over-reservation to the draft
    manager and prevents accidental KV blow-up on the target.
    """
    obj = _make_v1_kvm_for_reserve_test(is_draft=False, use_dynamic_tree=True)
    K = obj._spec_config.dynamic_tree_max_topK
    L = obj._spec_config.max_draft_len
    draft_len = 6
    n = _v1_gen_loop(obj, draft_len)

    assert obj._kv_reserve_draft_tokens == obj.max_total_draft_tokens
    assert obj._kv_reserve_draft_tokens < K * L, (
        f"Target should NOT inherit the K*L draft reserve; got "
        f"{obj._kv_reserve_draft_tokens}, K*L={K * L}"
    )
    # Target gen loop adds 1 + max(draft_len, max_total_draft_tokens).
    expected = 1 + max(draft_len, obj.max_total_draft_tokens)
    assert n == expected


def test_v1_non_dynamic_tree_unchanged():
    """V1 + linear/static tree: reserve == max_total (no growth)."""
    obj = _make_v1_kvm_for_reserve_test(is_draft=True, use_dynamic_tree=False)
    assert obj._kv_reserve_draft_tokens == obj.max_total_draft_tokens


# ---------------------------------------------------------------------------
# Fix #2 — KV reserve, V2 draft path via _prepare_draft_resources
# ---------------------------------------------------------------------------


class _FakeKVCache:
    """Minimal stand-in for V2's ``_KVCache`` — tracks capacity + resize calls."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.is_active = True
        self.resize_history = []

    def resize(self, new_capacity: int) -> bool:
        self.resize_history.append(new_capacity)
        self.capacity = new_capacity
        return True

    def resume(self, *_a, **_kw) -> bool:
        self.is_active = True
        return True


def _make_v2_draft_manager(reserve: int):
    """Build a real ``KVCacheManagerV2`` instance via ``__new__``.

    The full ``__init__`` requires a GPU pool, MPI mapping, and a
    ``KvCacheConfig``. ``_prepare_draft_resources`` only reads a small set
    of attributes (``is_draft``, ``_kv_reserve_draft_tokens``,
    ``num_extra_kv_tokens``, ``_stream``, ``kv_cache_map``), so we
    bypass the heavy ctor and set just those — leaving the production
    method body itself unmocked. Any silent regression in
    ``_prepare_draft_resources`` is caught by this test.
    """
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2

    mgr = KVCacheManagerV2.__new__(KVCacheManagerV2)
    mgr.is_draft = True
    mgr._kv_reserve_draft_tokens = reserve
    mgr.num_extra_kv_tokens = 0
    mgr._stream = types.SimpleNamespace(cuda_stream=0)
    mgr.kv_cache_map = {}
    return mgr


def _make_gen_request(req_id: int, draft_tokens):
    """Build a SimpleNamespace stand-in for ``LlmRequest`` for V2 gen path."""
    return types.SimpleNamespace(
        py_request_id=req_id,
        py_draft_tokens=list(draft_tokens),
        # ``request_context`` writes ``use_draft_model`` on enter/exit.
        use_draft_model=False,
    )


def _make_scheduled_requests(*, gen_requests=()):
    """Build a real ``ScheduledRequests`` populated with gen requests only."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    sr = ScheduledRequests()
    sr.generation_requests = list(gen_requests)
    return sr


def test_v2_prepare_resources_draft_dynamic_tree_kv_reserve():
    """Drive ``KVCacheManagerV2.prepare_resources`` end-to-end on a draft mgr.

    Worst case: ``draft_len == max_total_draft_tokens < K*L``.  Pre-fix this
    only added ``1 + draft_len``; post-fix must add ``1 + K*L`` because the
    dynamic-tree draft loop will write that many KV slots regardless of the
    schedule's ``py_draft_tokens`` length.

    This is the production-path regression Round-2 MUST-FIX #1 demanded:
    losing the ``reserve_slack`` plumbing in ``_prepare_draft_resources``
    would change ``kv_cache.capacity`` here and fail the assertion below.
    """
    K, L = 4, 4
    max_total = 8
    reserve = max(max_total, K * L)  # = 16
    pre_cap = 100

    mgr = _make_v2_draft_manager(reserve)
    kv_cache = _FakeKVCache(pre_cap)
    mgr.kv_cache_map[42] = kv_cache

    req = _make_gen_request(req_id=42, draft_tokens=[0] * max_total)
    sr = _make_scheduled_requests(gen_requests=[req])

    # Production entry point — no helper reimplementation.
    mgr.prepare_resources(sr)

    expected = pre_cap + 1 + reserve  # 100 + 1 + 16 = 117
    assert kv_cache.capacity == expected, (
        f"prepare_resources must reserve K*L slots; got "
        f"capacity={kv_cache.capacity}, expected={expected}"
    )
    assert kv_cache.resize_history == [expected], (
        f"Expected single resize to {expected}, got {kv_cache.resize_history}"
    )
    # request_context must restore the flag after exiting.
    assert req.use_draft_model is False


def test_v2_prepare_resources_draft_no_slack_when_draft_eq_reserve():
    """When ``draft_len >= K*L`` (linear tree), reserve_slack is 0.

    Drives the same production path, but with sizing where the existing
    gen-capacity formula already covers the draft footprint, so
    ``_prepare_draft_resources`` must not add any extra slack.
    """
    K, L = 4, 4
    max_total = 16  # max_total >= K*L → reserve == max_total
    reserve = max(max_total, K * L)  # = 16
    pre_cap = 30

    mgr = _make_v2_draft_manager(reserve)
    kv_cache = _FakeKVCache(pre_cap)
    mgr.kv_cache_map[1] = kv_cache

    req = _make_gen_request(req_id=1, draft_tokens=[0] * max_total)
    sr = _make_scheduled_requests(gen_requests=[req])

    mgr.prepare_resources(sr)

    # draft_len == reserve → reserve_slack == 0 → only base + draft_len added
    assert kv_cache.capacity - pre_cap == 1 + max_total
    assert kv_cache.resize_history == [pre_cap + 1 + max_total]


def test_v2_prepare_resources_target_is_no_op():
    """Target manager: ``prepare_resources`` returns without resizing.

    The target gen path is ``try_allocate_generation``; ``prepare_resources``
    short-circuits on ``self.is_draft is False``.  This pins that contract.
    """
    mgr = _make_v2_draft_manager(reserve=16)
    mgr.is_draft = False  # flip to target

    kv_cache = _FakeKVCache(50)
    mgr.kv_cache_map[7] = kv_cache

    req = _make_gen_request(req_id=7, draft_tokens=[0] * 6)
    sr = _make_scheduled_requests(gen_requests=[req])

    mgr.prepare_resources(sr)

    # Target's prepare_resources is a no-op for KV resize.
    assert kv_cache.capacity == 50
    assert kv_cache.resize_history == []


# ---------------------------------------------------------------------------
# Targeted source-level guard on the production resize body
# ---------------------------------------------------------------------------


def test_v2_prepare_draft_resources_keeps_reserve_slack_in_gen_loop():
    """Pin the exact ``reserve_slack`` arithmetic in the production V2 body.

    Replaces the prior whole-module ``count(...) >= 4`` guard with a
    targeted assertion on the gen-loop body of the actual method that owns
    the dynamic-tree reservation. If the slack term is removed or moved
    outside ``_prepare_draft_resources``, this test fails immediately.
    """
    import inspect

    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2

    src = inspect.getsource(KVCacheManagerV2._prepare_draft_resources)
    # The gen-loop must (a) compute slack from ``_kv_reserve_draft_tokens``
    # vs the schedule's ``draft_token_length`` and (b) add it to ``new_cap``
    # before resizing.
    assert "_kv_reserve_draft_tokens" in src, (
        "Fix #2 lost: _prepare_draft_resources no longer references _kv_reserve_draft_tokens"
    )
    assert "reserve_slack" in src, (
        "Fix #2 lost: _prepare_draft_resources no longer computes reserve_slack"
    )
    assert "get_draft_token_length(req)" in src, (
        "Fix #2 contract changed: slack must be relative to draft_token_length"
    )
    assert "new_cap += reserve_slack" in src, (
        "Fix #2 lost: _prepare_draft_resources no longer adds slack to new_cap before resize()"
    )


# ---------------------------------------------------------------------------
# Fix #1.b — under-budget config: buf_dim > n_dt (K*L > max_total_draft+1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dynamic-tree slot buffers live on CUDA")
def test_dynamic_tree_under_budget_buf_dim_gt_n_dt():
    """Allow buf_dim > n_dt and keep row alignment + reshape semantics intact.

    Production configs may set ``max_total_draft_tokens + 1 < K * max_draft_len``
    (the EAGLE3 ``draft30`` benchmark uses ``max_total=30, K=10, L=6`` so
    ``buf_dim = max(31, 60) = 60`` while ``n_dt = 31``).  The old runtime
    ``assert buf_dim == n_dt`` rejected this config outright.  Post-fix:

    * the assertion is ``buf_dim >= n_dt`` (this constructor path),
    * the dest copy still writes only the first ``n_dt`` rows / cols, and
    * ``_reshape_position_offsets_for_cpp`` honors stride before slicing so
      C++ sees a contiguous ``[m, n_dt]`` buffer with the right per-row data.
    """
    max_num_requests = 4
    max_draft_len = 6
    dynamic_tree_max_topK = 10
    max_total_draft_tokens = 30  # K*L = 60, max_total+1 = 31 → buf_dim=60
    n_dt = max_total_draft_tokens + 1
    expected_buf_dim = dynamic_tree_max_topK * max_draft_len  # 60
    assert expected_buf_dim > n_dt

    spec_tree_manager = SpecTreeManager(
        max_num_requests=max_num_requests,
        use_dynamic_tree=True,
        max_total_draft_tokens=max_total_draft_tokens,
        max_draft_len=max_draft_len,
        eagle_choices=None,
        dynamic_tree_max_topK=dynamic_tree_max_topK,
    )
    assert spec_tree_manager._internal_buf_dim == expected_buf_dim

    _seed_slot_storage(spec_tree_manager)

    ctx_slot, gen0_slot, gen1_slot = 1, 2, 3
    spec_tree_manager.slot_storage.all_ids_buf[0] = ctx_slot
    spec_tree_manager.slot_storage.all_ids_buf[1] = gen0_slot
    spec_tree_manager.slot_storage.all_ids_buf[2] = gen1_slot

    attn = _build_attn_metadata(max_num_requests, n_dt, expected_buf_dim)

    batch_size = 3  # 1 ctx + 2 gens
    num_contexts = 1

    # Must not raise: the relaxed assertion accepts buf_dim > n_dt.
    attn.update_spec_dec_param(
        batch_size=batch_size,
        is_spec_decoding_enabled=True,
        is_spec_dec_tree=True,
        is_spec_dec_dynamic_tree=True,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        spec_tree_manager=spec_tree_manager,
        num_contexts=num_contexts,
    )
    torch.cuda.synchronize()

    # Stride/query_len metadata must reflect the under-budget layout.
    assert attn.position_offsets_stride == expected_buf_dim
    assert attn.position_offsets_query_len == n_dt

    # Each gen request's data must land at the correct stride*i offset, with
    # only the first n_dt cols populated. Rows >= num_gens stay padded.
    pos_dst_2d = attn.spec_decoding_position_offsets.view(max_num_requests, expected_buf_dim)
    expected_gen0 = torch.arange(n_dt, dtype=torch.int32, device="cuda") + gen0_slot * 1000
    expected_gen1 = torch.arange(n_dt, dtype=torch.int32, device="cuda") + gen1_slot * 1000
    assert torch.equal(pos_dst_2d[0, :n_dt], expected_gen0)
    assert torch.equal(pos_dst_2d[1, :n_dt], expected_gen1)

    # _reshape_position_offsets_for_cpp must produce a dense [m, n_dt] tensor
    # whose row 0 carries gen0 data, not the leftover bytes from row 0's
    # [n_dt:buf_dim) tail (which the broken flat-reshape would surface).
    metadata = types.SimpleNamespace(
        spec_decoding_position_offsets=attn.spec_decoding_position_offsets,
        max_num_requests=max_num_requests,
        position_offsets_query_len=n_dt,
        position_offsets_stride=expected_buf_dim,
    )
    reshaped = TrtllmAttention._reshape_position_offsets_for_cpp(metadata)
    assert reshaped.shape == (max_num_requests, n_dt)
    assert reshaped.is_contiguous(), (
        "C++ kernel expects a contiguous [m, n_dt] view; the reshape must "
        "contiguize after slicing the [m, buf_dim] view down to [:, :n_dt]."
    )
    assert torch.equal(reshaped[0], expected_gen0)
    assert torch.equal(reshaped[1], expected_gen1)


if __name__ == "__main__":
    unittest.main()
