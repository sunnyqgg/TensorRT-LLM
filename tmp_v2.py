#!/usr/bin/env python3
"""
独立运行的 KV cache relocation 测试脚本（不依赖 pytest）
"""

import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import torch
from transformers import LlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_llama import LlamaForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.speculative.utils import SpecDecodingTensor
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping


@lru_cache(maxsize=1)
def get_sm_version():
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor


# ============================================================================
# 辅助函数
# ============================================================================


def llm_models_root():
    """获取 LLM 模型根目录"""
    root = Path("/home/scratch.trt_llm_data/llm-models/")

    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ.get("LLM_MODELS_ROOT"))

    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")

    if not root.exists():
        raise FileNotFoundError(
            f"找不到模型目录！请设置 LLM_MODELS_ROOT 环境变量或确保以下路径之一存在:\n"
            f"  - /home/scratch.trt_llm_data/llm-models/\n"
            f"  - /scratch.trt_llm_data/llm-models/")

    return root


@contextmanager
def default_dtype(dtype: torch.dtype):
    """临时设置默认的 torch dtype"""
    cur_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(cur_default)


# ============================================================================
# LLaMA 3.1 8B 配置
# ============================================================================
LLAMA_3_1_8B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.43.0.dev0",
    "use_cache": True,
    "vocab_size": 128256
}


def test_llama_verification_with_kv_cache_relocation():
    """
    验证 KV cache relocation 的模型输出
    """
    print("=" * 80)
    print("开始测试: LLaMA KV Cache Relocation")
    print("=" * 80)

    backend = "TRTLLM"
    metadata_cls = get_attention_backend(backend).Metadata

    config_dict = deepcopy(LLAMA_3_1_8B_CONFIG)

    llama_config = LlamaConfig.from_dict(config_dict)
    dtype = llama_config.torch_dtype
    device = torch.device('cuda')

    print(f"\n配置信息:")
    print(f"  dtype: {dtype}")
    print(f"  device: {device}")
    print(f"  backend: {backend}")

    with torch.device(device), default_dtype(dtype):
        models_path = llm_models_root()
        model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

        print(f"\n加载模型: {model_dir}")

        hf_llama = HFLlamaForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="cuda",
        ).eval()

        model_config = ModelConfig(pretrained_config=llama_config,
                                   attn_backend=backend)

        llama = LlamaForCausalLM(model_config).to(dtype).to(device)
        llama.load_weights(hf_llama.state_dict())

    print("模型加载完成!")

    num_blocks = 1
    tokens_per_block = 64
    head_dim = llama.config.hidden_size // llama.config.num_attention_heads
    num_layers = llama.config.num_hidden_layers
    num_kv_heads = llama.config.num_key_value_heads
    max_seq_len = num_blocks * tokens_per_block
    batch_size = 1

    print(f"\nKV Cache 配置:")
    print(f"  num_blocks: {num_blocks}")
    print(f"  tokens_per_block: {tokens_per_block}")
    print(f"  head_dim: {head_dim}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  max_seq_len: {max_seq_len}")

    if dtype == torch.half:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    # Context phase
    print("\n" + "=" * 80)
    print("阶段 1: Context Phase")
    print("=" * 80)

    input_ids = torch.tensor([
        128000, 32, 6369, 1990, 264, 22999, 1217, 323, 459, 21075, 11478, 18328,
        13, 578, 18328, 6835, 11190, 11, 11944, 11, 323, 48887, 11503, 311, 279,
        1217, 596, 4860, 13, 14194, 25, 22691, 36660, 3931, 2891, 25
    ],
                             dtype=torch.int,
                             device=device)

    print(f"输入 token 数: {input_ids.size(-1)}")  #36

    num_cached_tokens_per_seq = [0]
    request_ids = [900]
    token_nums = [input_ids.size(-1)]
    prompt_lens = [input_ids.size(-1)]
    requests = kv_cache_manager.add_dummy_requests(request_ids, token_nums)
    requests[0]

    attn_metadata = metadata_cls(
        seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
        num_contexts=1,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
        ),
        max_num_requests=1,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
    )

    position_ids = [torch.arange(0, input_ids.size(-1))]
    position_ids = torch.cat(position_ids).unsqueeze(0).cuda()

    with torch.inference_mode():
        attn_metadata.prepare()
        logits = llama.forward(input_ids=input_ids,
                               position_ids=position_ids,
                               attn_metadata=attn_metadata)

    print(f"Context phase 完成, logits shape: {logits.shape}")

    def run_forward(input_ids, position_ids, attn_metadata):
        # attn_metadata.prepare()
        return llama.forward(input_ids=input_ids,
                             position_ids=position_ids,
                             attn_metadata=attn_metadata,
                             return_context_logits=True)

    # Generation phase 0
    print("\n" + "=" * 80)
    print("阶段 2: Generation Phase 0 (Speculative Decoding)")
    print("=" * 80)

    gen_input_ids_0 = torch.tensor([
        22691, 11, 0, 13, 15592, 323, 315, 12, 311, 362, 220, 32, 362, 426, 330,
        358, 362, 358, 358, 362, 32, 0, 13, 32, 6369
    ],
                                   dtype=torch.int,
                                   device=device)  # 25

    print(f"生成 token 数: {gen_input_ids_0.size(-1)}")

    # spec_decoding_position_offsets = torch.tensor([
    #     0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    #     3
    # ],
    #                                               dtype=torch.int,
    #                                               device=device)
    # spec_decoding_packed_mask = torch.tensor(
    #     [
    #         1, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2051, 4099, 8195,
    #         16387, 32771, 65541, 131077, 262153, 524297, 1048593, 2097169,
    #         4194321, 8388641, 16842757
    #     ],
    #     dtype=torch.int,
    #     device=device).unsqueeze(0).unsqueeze(2)

    spec_decoding_position_offsets = torch.tensor([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0
    ],
                                                  dtype=torch.int,
                                                  device=device)
    spec_decoding_packed_mask = torch.tensor(
        [
            33554431, 33554431, 33554431, 33554431, 33554431, 33554431,
            33554431, 33554431, 33554431, 33554431, 33554431, 33554431,
            33554431, 33554431, 33554431, 33554431, 33554431, 33554431,
            33554431, 33554431, 33554431, 33554431, 33554431, 33554431, 33554431
        ],
        dtype=torch.int,
        device=device).unsqueeze(0).unsqueeze(2)

    num_cached_tokens_per_seq = [input_ids.size(-1)]
    is_spec_decoding_enabled = True
    use_spec_decoding = True
    is_spec_dec_tree = True
    is_spec_dec_dynamic_tree = True
    max_draft_tokens = gen_input_ids_0.size(-1) - 1

    attn_metadata_gen_phase_0 = metadata_cls(
        seq_lens=torch.tensor([gen_input_ids_0.size(-1)], dtype=torch.int),
        num_contexts=0,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
        ),
        max_num_requests=1,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        is_spec_decoding_enabled=is_spec_decoding_enabled,
        use_spec_decoding=use_spec_decoding,
        is_spec_dec_tree=is_spec_dec_tree,
        is_spec_dec_dynamic_tree=is_spec_dec_dynamic_tree,
    )
    spec_decoding_tensor = SpecDecodingTensor(
        position_offsets=spec_decoding_position_offsets,
        packed_mask=spec_decoding_packed_mask)

    attn_metadata_gen_phase_0.prepare()
    attn_metadata_gen_phase_0.update_spec_dec_param(
        is_spec_decoding_enabled=is_spec_decoding_enabled,
        is_spec_dec_dynamic_tree=is_spec_dec_dynamic_tree,
        is_spec_dec_tree=is_spec_dec_tree,
        max_draft_tokens=max_draft_tokens,
        spec_decoding_tensor=spec_decoding_tensor,
    )

    gen_position_ids_0 = [
        torch.full((gen_input_ids_0.size(-1), ),
                   input_ids.size(-1),
                   dtype=torch.int64)
    ]
    gen_position_ids_0 = torch.cat(gen_position_ids_0).unsqueeze(0).cuda()

    print("\n>>> 即将调用 C++ 代码进行前向计算 <<<")
    print(">>> 可以在这里设置 GDB 断点 <<<\n")

    with torch.inference_mode():
        # 这里是调用 C++ 代码的关键点
        # breakpoint()  # 在 GDB 中运行时会在这里暂停
        gen_logits_0 = run_forward(input_ids=gen_input_ids_0,
                                   position_ids=gen_position_ids_0,
                                   attn_metadata=attn_metadata_gen_phase_0)
        torch.cuda.synchronize()
        print(f"====gqq get_sm_version() is {get_sm_version()}")

        print(f"Generation phase 0 完成, logits shape: {gen_logits_0.shape}")

        # 保存 logits 和相关张量到 safetensor
        from safetensors.torch import save_file
        save_gen_logits_0 = gen_logits_0.cpu()
        save_path = "generation_phase_0_output" + str(
            get_sm_version()) + ".safetensors"
        # save_file 需要一个字典
        tensors_to_save = {"gen_logits_0": save_gen_logits_0}
        save_file(tensors_to_save, save_path)
        print(f"已保存张量到: {save_path}")
        print(f"保存的张量: gen_logits_0 shape={save_gen_logits_0.shape}")
        print(
            f"save_gen_logits_0.shape: {save_gen_logits_0.shape} and save_gen_logits_0.max(): {save_gen_logits_0.max()} and save_gen_logits_0.min(): {save_gen_logits_0.min()} and save_gen_logits_0.mean(): {save_gen_logits_0.mean()}"
        )
        save_path_sm90 = "generation_phase_0_output90.safetensors"
        sm90_ref = load_saved_tensors(save_path_sm90)
        sm90_ref_logits = sm90_ref["gen_logits_0"]
        print(f"sm90_ref_logits shape={sm90_ref_logits.shape}")
        print("come here")

    #     # KV cache 重定位
    #     print("\n执行 KV cache relocation...")
    #     request.py_num_accepted_draft_tokens = 1
    #     request.py_num_accepted_draft_tokens_indices = [1]
    #     request.py_rewind_len = gen_input_ids_0.size(
    #         -1) - request.py_num_accepted_draft_tokens - 1
    #     request.state = LlmRequestState.GENERATION_IN_PROGRESS
    #     scheduled_requests = ScheduledRequests()
    #     scheduled_requests.generation_requests = [request]
    #     kv_cache_manager.max_draft_len = gen_input_ids_0.size(-1) - 1
    #     kv_cache_manager.update_kv_cache_draft_token_location(
    #         scheduled_requests, attn_metadata_gen_phase_0,
    #         kv_cache_dtype_byte_size)
    #     if request.py_rewind_len > 0:
    #         kv_cache_manager.rewind_kv_cache(request, request.py_rewind_len)
    #         print(f"Rewind {request.py_rewind_len} tokens")

    # torch.cuda.synchronize()

    # # Generation phase 1
    # print("\n" + "=" * 80)
    # print("阶段 3: Generation Phase 1")
    # print("=" * 80)

    # gen_input_ids_1 = torch.tensor([2650, 649], dtype=torch.int, device=device)

    # print(f"生成 token 数: {gen_input_ids_1.size(-1)}")  #dense custom

    # num_cached_tokens_per_seq_1 = [
    #     input_ids.size(-1) + request.py_num_accepted_draft_tokens + 1
    # ]
    # attn_metadata_gen_phase_0.seq_lens = torch.tensor(
    #     [gen_input_ids_1.size(-1)], dtype=torch.int)
    # attn_metadata_gen_phase_0.kv_cache_params.num_cached_tokens_per_seq = num_cached_tokens_per_seq_1
    # attn_metadata_gen_phase_0.prepare()
    # attn_metadata_gen_phase_0.update_spec_dec_param(
    #     is_spec_decoding_enabled=is_spec_decoding_enabled,
    #     is_spec_dec_tree=is_spec_dec_tree if get_sm_version() < 100 else False,#False for b100
    #     is_spec_dec_dynamic_tree=False,
    #     max_draft_tokens=gen_input_ids_1.size(-1) - 1)

    # gen_position_ids_1 = [
    #     torch.full(
    #         (gen_input_ids_1.size(-1), ),
    #         input_ids.size(-1) + request.py_num_accepted_draft_tokens + 1,
    #         dtype=torch.int64)
    # ]
    # gen_position_ids_1 = torch.cat(gen_position_ids_1).unsqueeze(0).cuda()

    # with torch.inference_mode():

    #     gen_logits_1 = run_forward(input_ids=gen_input_ids_1,
    #                                position_ids=gen_position_ids_1,
    #                                attn_metadata=attn_metadata_gen_phase_0)

    # print(f"Generation phase 1 完成, logits shape: {gen_logits_1.shape}")

    # torch.cuda.synchronize()

    # # Reference generation
    # print("\n" + "=" * 80)
    # print("阶段 4: Reference Generation (Ground Truth)")
    # print("=" * 80)

    # gen_input_ids_ref = torch.tensor([22691, 0, 2650, 649],
    #                                  dtype=torch.int,
    #                                  device=device)

    # print(f"参考 token 数: {gen_input_ids_ref.size(-1)}")

    # num_cached_tokens_per_seq_ref = [input_ids.size(-1)]

    # attn_metadata_ref = metadata_cls(
    #     seq_lens=torch.tensor([gen_input_ids_ref.size(-1)], dtype=torch.int),
    #     num_contexts=0,
    #     kv_cache_params=KVCacheParams(
    #         use_cache=True,
    #         num_cached_tokens_per_seq=num_cached_tokens_per_seq_ref,
    #     ),
    #     max_num_requests=1,
    #     max_num_tokens=8192,
    #     kv_cache_manager=kv_cache_manager,
    #     request_ids=request_ids,
    #     prompt_lens=prompt_lens,
    #     is_spec_decoding_enabled=is_spec_decoding_enabled,
    #     use_spec_decoding=use_spec_decoding,
    #     is_spec_dec_tree=is_spec_dec_tree,
    #     is_spec_dec_dynamic_tree=False)
    # attn_metadata_ref.update_spec_dec_param(
    #     is_spec_decoding_enabled=is_spec_decoding_enabled,
    #     is_spec_dec_tree=is_spec_dec_tree if get_sm_version() < 100 else False,#is_spec_dec_tree
    #     is_spec_dec_dynamic_tree=False,
    #     max_draft_tokens=gen_input_ids_ref.size(-1) - 1,
    # )

    # gen_position_ids_ref = [
    #     torch.full((gen_input_ids_ref.size(-1), ),
    #                input_ids.size(-1),
    #                dtype=torch.int64)
    # ]
    # gen_position_ids_ref = torch.cat(gen_position_ids_ref).unsqueeze(0).cuda()
    # with torch.inference_mode():
    #     attn_metadata_ref.prepare()
    #     gen_logits_ref = run_forward(input_ids=gen_input_ids_ref,
    #                                  position_ids=gen_position_ids_ref,
    #                                  attn_metadata=attn_metadata_ref)

    # print(f"Reference generation 完成, logits shape: {gen_logits_ref.shape}")

    # torch.cuda.synchronize()

    # # Verification
    # print("\n" + "=" * 80)
    # print("阶段 5: 验证结果")
    # print("=" * 80)

    # try:
    #     torch.testing.assert_close(gen_logits_1[0, :],
    #                                gen_logits_ref[2, :],
    #                                atol=0.02,
    #                                rtol=0.02)
    #     print("✓ Token 0 验证通过!")
    # except AssertionError as e:
    #     print(f"✗ Token 0 验证失败: {e}")

    # try:
    #     torch.testing.assert_close(gen_logits_1[1, :],
    #                                gen_logits_ref[3, :],
    #                                atol=0.02,
    #                                rtol=0.02)
    #     print("✓ Token 1 验证通过!")
    # except AssertionError as e:
    #     print(f"✗ Token 1 验证失败: {e}")

    # kv_cache_manager.shutdown()

    # print("\n" + "=" * 80)
    # print("测试完成!")
    # print("=" * 80)


def load_saved_tensors(path="generation_phase_0_output.safetensors"):
    """
    从 safetensor 文件中加载保存的张量

    用法:
        tensors = load_saved_tensors()
        gen_logits_0 = tensors["gen_logits_0"]
        gen_input_ids_0 = tensors["gen_input_ids_0"]
    """
    from safetensors.torch import load_file

    print(f"从 {path} 加载张量...")
    tensors = load_file(path)
    print(f"加载的张量: {list(tensors.keys())}")
    for name, tensor in tensors.items():
        print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    return tensors


if __name__ == '__main__':
    print("TensorRT-LLM KV Cache Relocation 测试")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    print()

    try:
        test_llama_verification_with_kv_cache_relocation()
        print("\n✅ 所有测试通过!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
