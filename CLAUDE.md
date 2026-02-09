# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
使用中文回答
保证性能的前提下，code要尽量简单

## Repository Overview

TensorRT-LLM is NVIDIA's optimized library for Large Language Model inference on NVIDIA GPUs. The codebase is architected on PyTorch and provides both high-level Python APIs and low-level optimization capabilities for efficient LLM serving.

### Key Architecture Components

**Two Backend Systems:**
1. **PyTorch Backend** (`tensorrt_llm/_torch/`) - Native PyTorch implementation with full flexibility
2. **TensorRT Backend** (`tensorrt_llm/`) - Optimized compilation using TensorRT engines

**Core Abstractions:**
- `LLM` class in `tensorrt_llm/llmapi/llm.py` - Main entry point for users
- `BaseLlmArgs`/`TorchLlmArgs`/`TrtLlmArgs` in `tensorrt_llm/llmapi/llm_args.py` - Configuration management using Pydantic models
- `PyExecutor` in `tensorrt_llm/_torch/pyexecutor/` - PyTorch runtime execution engine
- Model implementations in `tensorrt_llm/_torch/models/` - Native PyTorch model definitions

### Critical Workflow: LLM Initialization

When a user creates an `LLM` instance:
1. `LLM.__init__()` validates kwargs and selects `llm_args_cls` based on backend
2. `llm_args_cls.from_kwargs()` constructs args (triggers Pydantic validators)
3. Pydantic `@model_validator` methods run (e.g., `validate_speculative_config` in `TorchLlmArgs`)
4. For PyTorch backend: `_build_model()` → `create_py_executor()` in `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py`
5. Executor creates model engines, KV cache managers, and speculative decoding components

**Important:** Configuration fields may be modified during validation. For example, `AutoDecodingConfig` is transformed to `NGramDecodingConfig` in `create_py_executor()` when `decoding_type == "AUTO"`.

## Speculative Decoding Architecture

Located in `tensorrt_llm/_torch/speculative/`:

### Key Components

**Tree Management (`spec_tree_manager.py`):**
- `SpecTreeManager` - Manages draft token trees (static or dynamic)
- Static tree: Pre-computed tree structure using `eagle_choices`
- Dynamic tree: Runtime tree construction using `dynamic_tree_max_topK` parameter
- Maintains attention masks, position offsets, and tree paths for verification

**Drafting Loops (`drafting_loops.py`):**
- `LinearDraftingLoopWrapper` - Sequential token generation
- `StaticTreeDraftingLoopWrapper` - Fixed tree structure (eagle_choices)
- `DynamicTreeDraftingLoopWrapper` - Runtime adaptive tree (new feature)

**Dynamic Tree Implementation:**
- Buffers: `history_draft_tokens_buffer`, `history_score_buffer`, `history_draft_tokens_parent_buffer`
- Each layer generates `dynamic_tree_max_topK` tokens per parent node
- Total draft tokens: `dynamic_tree_max_topK * max_draft_len` (default)
- Maximum history: `dynamic_tree_max_topK + dynamic_tree_max_topK^2 * (max_draft_len - 1)`
- Tree rebuilt each iteration based on top-K sampling

### Config Flow for Dynamic Tree

`Eagle3DecodingConfig` with `use_dynamic_tree=True`:
1. User sets: `max_draft_len`, `dynamic_tree_max_topK`, `max_total_draft_tokens` (optional)
2. Validation in `llm_args.py:validate_eagle_config()` computes constraints
3. `DynamicTreeDraftingLoopWrapper` instantiated in executor creation
4. Runtime: Tree structure changes per-request based on logit sampling

## Common Development Commands

### Running Examples

```bash
# Basic LLM inference with PyTorch backend
python examples/llm-api/quickstart_advanced.py \
    --model_dir /path/to/model \
    --backend pytorch \
    --max_batch_size 8 \
    --max_num_tokens 8192

# EAGLE3 dynamic tree speculative decoding
python examples/llm-api/quickstart_advanced.py \
    --model_dir /path/to/target_model \
    --draft_model_dir /path/to/draft_model \
    --spec_decode_algo EAGLE3 \
    --use_dynamic_tree \
    --dynamic_tree_max_topK 10 \
    --spec_decode_max_draft_len 4 \
    --max_total_draft_tokens 40

# AUTO speculative decoding (selects NGramDecodingConfig automatically)
python examples/llm-api/quickstart_advanced.py \
    --model_dir /path/to/model \
    --spec_decode_algo AUTO \
    --max_batch_size 8
```

### Git Workflow

This appears to be an active development branch (`yweng/add_dyanmic_tree_support`) implementing EAGLE3 dynamic tree support.

Main branch: `main`

Key recent commits (c2a7b5366..37e7bd87):
- Implementation of EAGLE3 dynamic tree feature
- Changes in `llm_args.py`, `drafting_loops.py`, `spec_tree_manager.py`
- Tree building algorithm with runtime topK selection

## Important Implementation Notes

### Pydantic Configuration Validation

All `*LlmArgs` classes use Pydantic v2 with strict validation. Key patterns:

1. **Union Types with Discriminators:**
   - `SpeculativeConfig` is a Union of multiple config types
   - No explicit discriminator field, type determined by isinstance checks
   - Be careful: Pydantic may set Union fields to None if validation fails silently

2. **Model Validators:**
   - Run in `mode="after"` - after field assignment
   - Can modify `self` fields (e.g., compute derived values)
   - Order matters: validators run in definition order

3. **Config Transformation:**
   - Some configs transform during initialization (e.g., `AutoDecodingConfig` → `NGramDecodingConfig`)
   - Check in `create_py_executor()` for runtime transformations

### Attention Backends

`attn_backend` parameter controls attention kernel implementation:
- `TRTLLM` - Custom TensorRT-LLM kernels (default, most optimized)
- `FLASHINFER` - FlashInfer attention (open source)
- `FLASHINFER_STAR_ATTENTION` - FlashInfer with star attention
- `VANILLA` - PyTorch native attention (debugging)

Constraints:
- FlashInfer backends disable `enable_block_reuse` automatically
- Star attention disables chunked context

### Speculative Decoding Token Flow

For EAGLE3 one-model mode (draft + target in single model):

1. **Prefill:** Target model processes prompt, stores hidden states
2. **Draft Generation:** Draft layers autoregressively generate tree of candidates
3. **Verification:** Target model verifies all draft tokens in parallel using tree attention
4. **Acceptance:** Longest accepted prefix becomes new context
5. **Repeat:** Draft from new position with updated KV cache

Key files:
- Verification logic: `tensorrt_llm/_torch/pyexecutor/py_executor.py:_forward_step()`
- Tree attention: `tensorrt_llm/_torch/attention_backend/trtllm.py`
- Acceptance: `tensorrt_llm/_torch/speculative/` acceptance modules

## File Organization

```
tensorrt_llm/
├── llmapi/               # High-level user-facing API
│   ├── llm.py           # LLM class (main entry point)
│   ├── llm_args.py      # Configuration classes with Pydantic validation
│   └── llm_utils.py     # Model loading, caching utilities
├── _torch/              # PyTorch backend implementation
│   ├── models/          # Native PyTorch model definitions (Llama, DeepSeek, etc.)
│   ├── pyexecutor/      # Execution engine for PyTorch backend
│   │   ├── py_executor.py         # Main executor loop
│   │   └── py_executor_creator.py # Executor factory
│   ├── speculative/     # Speculative decoding implementations
│   │   ├── drafting_loops.py      # Draft token generation loops
│   │   ├── spec_tree_manager.py   # Tree structure management
│   │   └── eagle3.py              # EAGLE3-specific logic
│   └── attention_backend/ # Attention kernel implementations
└── examples/
    └── llm-api/
        └── quickstart_advanced.py # Primary test/demo script
```

## Debugging Speculative Decoding Issues

Common issues when `speculative_config` becomes None:

1. **Check Pydantic validation:** Add debug logging in `validate_speculative_config()` model validators
2. **Verify Union type compatibility:** Ensure config class is in `SpeculativeConfig` TypeAlias
3. **Check transformation points:** Look for `spec_config =` assignments in `create_py_executor()`
4. **Backend compatibility:** Call `config.supports_backend(backend)` to verify

Example debug pattern:
```python
# Add to llm_args.py in validator
from tensorrt_llm.logger import logger
logger.info(f"[DEBUG] speculative_config = {self.speculative_config}, type = {type(self.speculative_config)}")
```

## Model Architecture Notes

Models in `tensorrt_llm/_torch/models/` follow a standard pattern:
- Inherit from `DecoderModel` base class
- Define layers in `__init__`: embeddings, decoder layers, LM head
- `forward()` method handles both prefill and decode phases
- Support for LoRA adapters, quantization, tensor parallelism

EAGLE3 models have additional draft layers that predict next tokens using hidden states from specific layers of the target model.
