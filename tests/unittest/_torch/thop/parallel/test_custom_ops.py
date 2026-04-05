import pytest
import torch
import torch._library.utils as library_utils

import tensorrt_llm  # noqa: F401


def discover_namespace_ops(namespace: str, prefix: str = ""):
    """Discover custom ops in a specific namespace."""
    # C++ custom ops are lazy loaded, cannot use torch.ops.x to discover all custom ops.
    # Use schemas to discover instead.
    ops_schemas = torch._C._jit_get_all_schemas()
    ops = []

    ns_prefix = f"{namespace}::{prefix}"
    print("Discovering custom ops:")
    for schema in ops_schemas:
        if not schema.name.startswith(ns_prefix):
            continue
        op = library_utils.lookup_op(schema.name)
        ops.append(op)
        print(f"    {op._name}")
    return ops


def discover_custom_ops(namespaces):
    """Discover all custom ops in the codebase."""
    discovered_ops = []
    for ns in namespaces:
        ops = discover_namespace_ops(ns)
        print(f"Total {len(ops)} custom ops in namespace {ns}")
        discovered_ops.extend(ops)
    return discovered_ops


@pytest.fixture(scope="module", autouse=True)
def custom_ops():
    """Discover custom ops in the codebase."""
    # "auto_deploy" custom ops are not checked here.
    custom_op_namespaces = ("trtllm", )

    return discover_custom_ops(custom_op_namespaces)


# Better to add OpInfo for each custom op, and use opcheck to test the custom ops.
# Currently OpInfo for custom ops are not available in the codebase.
# As a trade-off, only fake registration is checked.
def test_register_fake(custom_ops):
    """Test custom operator fake impl registration."""

    # Custom ops that are not required to have fake impl.
    waivers = {
        "trtllm::record_stream",
        "trtllm::wait_event",
        "trtllm::record_event",
        "trtllm::set_stream",
    }

    # TODO: add fake impl for these ops in follow-up PRs.
    to_fix = {
        "trtllm::lora_grouped_gemm",
        "trtllm::lora_grouped_gemm_cuda_graph",
        "trtllm::lora_group_gemm_param_fill_row_reorder_fusion",
        "trtllm::mtp_relaxed_acceptance_op",
        "trtllm::mtp_update_hidden_states_op",
        "trtllm::mtp_prepare_drafter_inputs_op",
        "trtllm::extract_real_draft_tokens_op",
        "trtllm::selective_scan",
        "trtllm::reducescatter_list",
        "trtllm::reducescatter_list_pg",
        "trtllm::fp8_per_tensor_scale_moe_runner",
        "trtllm::migrate_to_host_accessible",
        "trtllm::mnnvl_moe_alltoallv_prepare_without_allgather",
        "trtllm::mamba_conv1d",
        "trtllm::llama4_moe_tp8ep1_min_latency",
        "trtllm::llama4_fp8_fp8_gemm_swiglu",
        "trtllm::llama4_fp8_bf16_gemm",
        "trtllm::llama4_bf16_bf16_gemm",
        "trtllm::fused_topk_softmax",
        "trtllm::fp8_batched_quantize_1x128_permute102",
        "trtllm::fp8_block_scaling_moe_gemm",
        "trtllm::fp8_block_scaling_bmm_out",
        "trtllm::fp8_block_scaling_bmm",
        "trtllm::fp4_batched_quantize",
        "trtllm::fp4_gemm_trtllmgen",
        "trtllm::fp4_bmm",
        "trtllm::fp4_fp8_gemm_trtllmgen",
        "trtllm::cuda_scaled_mm",
        "trtllm::initialize_static_lowprecision_buffers",
        "trtllm::cutlass_scaled_mm",
        "trtllm::fp8_per_tensor_scaling_tllmg_gemm",
        "trtllm::load_chunked_kv_cache_for_mla",
        "trtllm::load_paged_kv_cache_for_mla",
        "trtllm::set_paged_kv_cache_for_mla",
        "trtllm::set_chunked_kv_cache_for_mla",
        "trtllm::mla_rope_append_paged_kv_assign_q",
        "trtllm::fused_qk_norm_rope",
        "trtllm::bf16_mxe2m1_block_scale_moe_runner",
        "trtllm::e4m3_mxe2m1_block_scale_moe_runner",
        "trtllm::mxe4m3_mxe2m1_block_scale_moe_runner",
        "trtllm::mxfp8_quantize",
        "trtllm::mamba2_mtp_ssm_cache_update",
        "trtllm::build_dynamic_tree_op",
        "trtllm::verify_dynamic_tree_greedy_op",
        "trtllm::verify_dynamic_tree_greedy_out_op",
    }

    ops_missing_fake_impl = []

    for op in custom_ops:
        if op._name in waivers or op._name in to_fix:
            continue
        if not library_utils.has_fake_kernel(op):
            ops_missing_fake_impl.append(op)

    names = ", ".join(op._name for op in ops_missing_fake_impl)
    assert len(
        ops_missing_fake_impl) == 0, f"Custom ops missing fake impl: {names}"


@pytest.mark.parametrize("batch_size,num_draft_tokens,num_spec_step", [
    (1, 17, 5),
    (4, 17, 5),
    (1, 9, 3),
    (8, 33, 9),
])
def test_verify_dynamic_tree_packed_equivalence(batch_size, num_draft_tokens,
                                                num_spec_step):
    """Verify that the packed kernel produces identical outputs to the original 3-pointer kernel."""
    device = "cuda"

    # Build random but valid tree structure
    retrieve_index = torch.zeros(batch_size,
                                 num_draft_tokens,
                                 dtype=torch.int32,
                                 device=device)
    retrieve_next_token = torch.full((batch_size, num_draft_tokens),
                                     -1,
                                     dtype=torch.int32,
                                     device=device)
    retrieve_next_sibling = torch.full((batch_size, num_draft_tokens),
                                       -1,
                                       dtype=torch.int32,
                                       device=device)

    # Simple chain tree: node 0 -> node 1 -> node 2 -> ...
    for b in range(batch_size):
        for i in range(num_draft_tokens):
            retrieve_index[b, i] = i
        for i in range(num_draft_tokens - 1):
            retrieve_next_token[b, i] = i + 1

    # Random candidates and target predictions (some matching, some not)
    vocab_size = 100
    candidates = torch.randint(0,
                               vocab_size, (batch_size, num_draft_tokens),
                               dtype=torch.int64,
                               device=device)
    target_predict = torch.randint(0,
                                   vocab_size, (batch_size, num_draft_tokens),
                                   dtype=torch.int64,
                                   device=device)
    # Force some matches to exercise acceptance path
    for b in range(batch_size):
        for i in range(min(3, num_draft_tokens - 1)):
            target_predict[b, i] = candidates[b, i + 1]

    tree_valid = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Pack retrieve into [B, N, 3]
    retrieve_packed = torch.stack(
        [retrieve_index, retrieve_next_token, retrieve_next_sibling],
        dim=-1).contiguous()

    # Allocate output buffers for original kernel
    predicts_orig = torch.zeros(batch_size * num_draft_tokens,
                                dtype=torch.int64,
                                device=device)
    accept_index_orig = torch.zeros(batch_size,
                                    num_spec_step,
                                    dtype=torch.int64,
                                    device=device)
    accept_token_num_orig = torch.zeros(batch_size,
                                        dtype=torch.int64,
                                        device=device)
    accept_token_orig = torch.zeros(batch_size,
                                    num_spec_step,
                                    dtype=torch.int64,
                                    device=device)

    # Allocate output buffers for packed kernel
    predicts_packed = torch.zeros_like(predicts_orig)
    accept_index_packed = torch.zeros_like(accept_index_orig)
    accept_token_num_packed = torch.zeros_like(accept_token_num_orig)
    accept_token_packed = torch.zeros_like(accept_token_orig)

    # Run original 3-pointer kernel
    torch.ops.trtllm.verify_dynamic_tree_greedy_out_op(
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        predicts_orig,
        accept_index_orig,
        accept_token_num_orig,
        accept_token_orig,
        tree_valid,
        num_spec_step,
    )

    # Run packed kernel
    torch.ops.trtllm.verify_dynamic_tree_greedy_out_packed_op(
        candidates,
        retrieve_packed,
        target_predict,
        predicts_packed,
        accept_index_packed,
        accept_token_num_packed,
        accept_token_packed,
        tree_valid,
        num_spec_step,
    )

    # Assert equivalence
    torch.testing.assert_close(predicts_orig,
                               predicts_packed,
                               msg="predicts mismatch")
    torch.testing.assert_close(accept_index_orig,
                               accept_index_packed,
                               msg="accept_index mismatch")
    torch.testing.assert_close(accept_token_num_orig,
                               accept_token_num_packed,
                               msg="accept_token_num mismatch")
    torch.testing.assert_close(accept_token_orig,
                               accept_token_packed,
                               msg="accept_token mismatch")
