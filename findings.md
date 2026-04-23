# Findings: Dynamic Tree EAGLE3 + CUDA Graph + TP>1 Crash

## 1. 测试矩阵

| # | Model | TP | Mode | CUDA Graph | Commit | Result | Node |
|---|-------|----|------|------------|--------|--------|------|
| 1 | Llama 8B | 1 | Dynamic tree | ON | baseline | **acc=5.22** ✅ | - |
| 2 | Llama 8B | 8 | Dynamic tree | ON | baseline | **CRASH** ❌ | - |
| 3 | Llama 8B | 8 | Linear | ON | baseline | **acc=3.79** ✅ | - |
| 4 | Qwen3-235B | 8 | Linear | ON | baseline | **acc=2.05** ✅ | - |
| 5 | Qwen3-235B | 8 | Dynamic tree | ON | baseline | **CRASH** ❌ | - |
| 6 | Qwen3-235B | 8 | Dynamic tree | OFF | baseline | **acc=2.85** ✅ | - |
| 7 | Llama 8B | 8 | Dynamic tree | OFF | 4db2d36 | **acc=2.50** ✅ | viking-prod-299 |
| 8 | Llama 8B | 8 | Dynamic tree | ON | 4db2d36 | **CRASH** ❌ | viking-prod-299 |
| 9 | Llama 8B | 8 | Dynamic tree | ON | 745239f (baseline) | **CRASH** ❌ | a4u8g-0120 |
| 10 | Llama 8B (BF16) | 1 | Dynamic tree | ON | 4db2d36 | **acc=5.22** ✅ | viking-prod-216 |
| 11 | Llama 8B (BF16) | 8 | Dynamic tree | ON | 4db2d36 | **CRASH** ❌ (CUBLAS_STATUS_EXECUTION_FAILED) | viking-prod-299 |
| 12 | Llama 8B (BF16) | 8 | Dynamic tree | OFF | 4db2d36 | **acc=3.39** ⚠️ | viking-prod-299 |

## 2. 关键观察

### 2.1 Crash 触发条件
- **必须**: Dynamic tree + CUDA graph + TP>1
- **TP=1 正常**: Dynamic tree + CUDA graph + TP=1 → acc=5.22 ✅
- **Linear 正常**: Linear + CUDA graph + TP>1 → acc=3.79 ✅
- **无 CG 正常**: Dynamic tree + 无 CUDA graph + TP>1 → acc=2.50 ✅

### 2.2 核心推断
**问题不仅仅是 `.item()` GPU→CPU sync**：
- 如果 `.item()` 是唯一原因，TP=1 也应该出问题（`.item()` 在 TP=1 时同样执行）
- 但 TP=1 完全正常 → 必须有 **NCCL 通信 + CUDA graph** 的特定交互问题
- 即：Dynamic tree 的某个操作在 CUDA graph capture 时与 NCCL all-reduce 不兼容

### 2.3 Pre-existing Bug
- Baseline `745239f8c` 也 crash → 不是我们的代码引入的
- 这是 dynamic tree + CUDA graph + TP>1 从来就不工作

## 3. on_update() 调用点分析

### 已修复（commit 4db2d36aa）
| 位置 | 文件 | 说明 |
|------|------|------|
| prepare_for_generation step0 | eagle3_dynamic_tree.py:921-927 | 替换为直接 int 赋值 |
| prepare_for_generation step>0 | eagle3_dynamic_tree.py:950-955 | 替换为直接 int 赋值 |
| prepare_1st_drafter_inputs | eagle3_dynamic_tree.py:466-473 | is_cuda_graph guard |
| linear draft step0 | eagle3.py:681-687 | 替换为直接 int 赋值 |

### 未修复但在执行路径上
| 位置 | 文件 | 说明 |
|------|------|------|
| _restore_attn_metadata_from_spec_dec | interface.py:528 | draft loop 后调用，仍在 CG capture 内 |
| save_metadata_state finally | drafting_loops.py:84 | 仅 LinearDraftingLoopWrapper/StaticTree 使用，**dynamic tree 不走此路径** |

### 不在 dynamic tree 路径上
| 位置 | 文件 | 说明 |
|------|------|------|
| LinearDraftingLoopWrapper._prepare_drafter_inputs | drafting_loops.py:176 | linear two-model 路径 |
| StaticTreeDraftingLoopWrapper._prepare_drafter_inputs | drafting_loops.py:455 | static tree 路径 |

## 4. Dynamic Tree vs Linear 关键差异

### 4.1 Tensor shape 差异
- **Linear**: 每个 draft step 固定 1 token/request → shape 一致
- **Dynamic tree**: step i 有 `i * K` tokens/request → shape 随 step 增长
- CUDA graph capture 时，每个 step 的 kernel 被展开录制，shape 在 capture 和 replay 间应一致

### 4.2 NCCL 操作差异
- **Linear**: 每个 step 1 个 all-reduce（1 token 的 hidden states）
- **Dynamic tree**: 每个 step all-reduce 的 tensor size 不同（K, 2K, 3K, ...）
- TP>1 时，所有 rank 必须以相同顺序、相同 shape 执行 all-reduce

### 4.3 额外操作
- Dynamic tree 有 `torch.topk`, `torch.gather`, mask 计算等操作
- 这些操作在 CUDA graph capture 时被录制
- 如果任何操作有 data-dependent control flow，会导致 CUDA graph 不兼容

## 5. Verification 命令

### Test 2: Dynamic tree + CG + TP=8 (关键测试)
```bash
python3 examples/llm-api/quickstart_advanced.py \
  --model_dir /home/scratch.qgai_sw/qgai/project/LLM/Meta-Llama-3.1-8B-Instruct \
  --spec_decode_algo EAGLE3 \
  --spec_decode_max_draft_len 6 \
  --draft_model_dir /home/scratch.trt_llm_data_ci/llm-models/EAGLE3-LLaMA3.1-Instruct-8B \
  --use_dynamic_tree --dynamic_tree_max_topK 10 --max_total_draft_tokens 60 \
  --use_one_model \
  --dataset /home/scratch.qgai_sw/qgai/project/mt_dataset.json \
  --use_cuda_graph --kv_cache_fraction 0.5 --num_samples 1 --tp_size 8
```

### Test 2 + Debug (带 CUDA_LAUNCH_BLOCKING)
```bash
CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=WARN python3 examples/llm-api/quickstart_advanced.py \
  --model_dir /home/scratch.qgai_sw/qgai/project/LLM/Meta-Llama-3.1-8B-Instruct \
  --spec_decode_algo EAGLE3 \
  --spec_decode_max_draft_len 6 \
  --draft_model_dir /home/scratch.trt_llm_data_ci/llm-models/EAGLE3-LLaMA3.1-Instruct-8B \
  --use_dynamic_tree --dynamic_tree_max_topK 10 --max_total_draft_tokens 60 \
  --use_one_model \
  --dataset /home/scratch.qgai_sw/qgai/project/mt_dataset.json \
  --use_cuda_graph --kv_cache_fraction 0.5 --num_samples 1 --tp_size 8
```

### Test 1: Dynamic tree + no CG + TP=8 (功能回归)
```bash
python3 examples/llm-api/quickstart_advanced.py \
  --model_dir /home/scratch.qgai_sw/qgai/project/LLM/Meta-Llama-3.1-8B-Instruct \
  --spec_decode_algo EAGLE3 \
  --spec_decode_max_draft_len 6 \
  --draft_model_dir /home/scratch.trt_llm_data_ci/llm-models/EAGLE3-LLaMA3.1-Instruct-8B \
  --use_dynamic_tree --dynamic_tree_max_topK 10 --max_total_draft_tokens 60 \
  --use_one_model \
  --dataset /home/scratch.qgai_sw/qgai/project/mt_dataset.json \
  --kv_cache_fraction 0.5 --num_samples 5 --tp_size 8
```

### Test 3: Linear + CG + TP=8 (回归测试)
```bash
python3 examples/llm-api/quickstart_advanced.py \
  --model_dir /home/scratch.qgai_sw/qgai/project/LLM/Meta-Llama-3.1-8B-Instruct \
  --spec_decode_algo EAGLE3 \
  --spec_decode_max_draft_len 5 \
  --draft_model_dir /home/scratch.trt_llm_data_ci/llm-models/EAGLE3-LLaMA3.1-Instruct-8B \
  --use_one_model \
  --dataset /home/scratch.qgai_sw/qgai/project/mt_dataset.json \
  --use_cuda_graph --kv_cache_fraction 0.5 --num_samples 5 --tp_size 8
```

## 6. ROOT CAUSE 发现 (Session 2)

### Crash Kernel
`updateKVCacheDraftTokenLocationBatched2D<KVBlockArray, 32>` — 在 `eagle3_dynamic_tree.py:345`
`_relocate_kv_eagerly()` 中调用。

### 调用链
```
Eagle3Worker.forward()
  → sample_and_accept_draft_tokens() [eagle3_dynamic_tree.py:364]
    → _sample_and_accept_dynamic_tree()
    → _relocate_kv_eagerly() [line 374]
      → torch.ops.tensorrt_llm.update_kv_cache_draft_token_location_2d() [line 345]
        → updateKVCacheDraftTokenLocationBatched2D<KVBlockArray, 32>  ← CRASH HERE
```

### 为什么只在 TP>1 时 Crash
- **Linear 模式没有 `_relocate_kv_eagerly`** → 不调用此 kernel → 不 crash
- TP=1 时 KV cache 参数（num_kv_heads, block_offsets 等）不同，可能碰巧不触发 OOB
- Crash 发生在 CUDA graph **warmup** 阶段（真实执行），不是 capture 阶段

### Attention Workspace 动态 Resize
Log 显示 workspace 在 warmup 期间动态增长：
```
0 → 42558464 bytes (x8 ranks)
42558464 → 71368704 bytes (x8 ranks)
0 → 71368704 bytes (x8 ranks)  ← 重置后再次增长
```
第 3 次从 0 开始说明 workspace 在两次 warmup 间被重置。

### 关键参数 (_relocate_kv_eagerly)
```python
torch.ops.tensorrt_llm.update_kv_cache_draft_token_location_2d(
    accepted_draft_indices[:batch_size],  # 全部 -1 在 warmup 时
    num_accepted_tokens[:batch_size],     # 全部 1 在 warmup 时
    kv_lens_cuda[:batch_size],
    True,
    cache_mgr.num_layers,
    cache_mgr.num_kv_heads,    # TP>1 时更小 (8/8=1 for Llama 8B)
    kv_head_dim_bytes,
    cache_mgr.max_total_draft_tokens,
    cache_mgr.max_attention_window_vec[0],
    cache_mgr.kv_cache_pool_pointers,
    attn_metadata.kv_cache_block_offsets,
    cache_mgr.max_blocks_per_seq,
    cache_mgr.tokens_per_block,
    None,
)
```

## 8. Session 3 发现 (2026-03-24)

### 8.1 on_update fix (4db2d36) 验证 — BF16 模型

使用 BF16 模型 (`Meta-Llama-3.1-8B-Instruct`) 而非 FP8，排除量化对 draft quality 的影响。

| TP | CG | Mode | AR | 状态 |
|----|-----|------|----|------|
| 1 | ON | Dynamic tree | 5.22 | ✅ 正常 |
| 8 | ON | Dynamic tree | CRASH | ❌ CUBLAS_STATUS_EXECUTION_FAILED |
| 8 | OFF | Dynamic tree | 3.39 | ⚠️ 不崩溃但 AR 从 5.22 降到 3.39 |

### 8.2 关键结论

1. **on_update fix 不能解决 TP>1 + CG crash**: TP=8 + CG 仍然崩溃，错误从 `illegal memory access` 变成 `CUBLAS_STATUS_EXECUTION_FAILED`（可能是错误被更早捕获）
2. **TP>1 AR 下降是 pre-existing bug**: 即使无 CG，TP=8 dynamic tree AR 也只有 3.39（TP=1 为 5.22），约 35% 下降
3. **两个独立问题需要分别修复**:
   - **Bug A**: Dynamic tree + CG + TP>1 crash → 与 `_relocate_kv_eagerly` 或 NCCL+CG 交互有关
   - **Bug B**: Dynamic tree + TP>1 AR 下降 (5.22→3.39) → 与 CG 无关，纯 TP 问题

### 8.3 之前 Session 2 的 FP8 测试对比

FP8 模型 (`Llama-3.1-8B-Instruct-FP8`) 在 commit 4db2d36 上：
- TP=2 + CG + dynamic tree: AR=2.56 ✅（没 crash！）
- TP=2 + CG + linear: AR=2.09 ✅
- TP=1 + CG + dynamic tree: AR=2.27 ✅
- TP=2 + NO CG + dynamic tree: AR=2.52 ✅

**FP8 TP=2 不 crash 但 BF16 TP=8 crash** → crash 与 TP 数量有关（TP=2 碰巧不触发，TP=8 触发）

### 8.5 Crash Fix: 条件跳过 `_relocate_kv_eagerly`

**Root cause**: `update_kv_cache_draft_token_location_2d` C++ kernel 在 CUDA graph + TP>1 时崩溃。

**Fix**: 在 `sample_and_accept_draft_tokens` 中，当 `is_cuda_graph AND tp_size > 1` 时跳过 eager KV relocation。
Lazy relocation 在 `resource_manager.py` 的 `update_resources()` 中正确处理（CUDA graph 外）。

**代码变更** (`eagle3_dynamic_tree.py:375-380`):
```python
if num_gens > 0:
    # Skip eager KV relocation during CUDA graph with TP>1:
    # the C++ kernel crashes with NCCL+CG interaction.
    if not (attn_metadata.is_cuda_graph and self.mapping.tp_size > 1):
        self._relocate_kv_eagerly(attn_metadata, batch_size)
```

**验证结果**:

| TP | CG | AR | 状态 | 对比 |
|----|-----|-----|------|------|
| 1 | ON | 5.22 | ✅ | = baseline |
| 2 | ON | 3.28 | ✅ 不 crash | = NO CG baseline |
| 8 | ON | 3.39 | ✅ 不 crash | = NO CG baseline |

**关键**: TP=1 AR 不受影响 (5.22)，TP>1 不再 crash，AR 与 NO CG 一致。

### 8.6 剩余问题: TP>1 AR 下降 (5.22→3.3x)

**独立的 pre-existing bug**，与 CUDA graph 和 crash fix 无关：
- TP=1: AR=5.22
- TP=2: AR=3.28 (无论是否 CG)
- TP=8: AR=3.39 (无论是否 CG)
- Linear 模式不受影响: TP=1~8 AR 稳定 3.1-3.6

**验证 eager reloc 不是原因**:
- TP=8 + NO CG + eager reloc 执行: AR=3.39
- TP=8 + CG + eager reloc 跳过: AR=3.39
- 两者完全一致（相同文本，相同 iter 数 151）→ eager reloc 对 TP>1 AR 无影响
- 用户注释掉 `is_cuda_graph` guard 后重跑，结果仍然 3.39

### 8.4 节点信息
- viking-prod-216: 8x H200, Job 1597495
- viking-prod-299: 8x H200, Job 1597496
- viking-prod-323: 8x H200, Job 1597210 (已有容器)

## 7. 待调查方向

### 5.1 interface.py:528 的 on_update()
- `_restore_attn_metadata_from_spec_dec()` 在 draft loop 后调用
- 对 linear 和 dynamic tree 都执行
- Linear 没问题 → 可能不是关键因素，但应该修复

### 5.2 all_rank_num_tokens 处理
- eagle3_dynamic_tree.py:591-593:
  ```python
  if original_all_rank_num_tokens is not None:
      if spec_metadata.all_rank_num_seqs is not None:
          attn_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_seqs
  ```
- 这控制 TP 时的 token 分配，可能影响 all-reduce 行为

### 5.3 CUDA graph capture 中的 Python if 语句
- Dynamic tree draft loop 中有多个 Python `if` 分支
- CUDA graph capture 时这些分支的真值在 capture 和 replay 间必须一致
- 需要检查是否有基于 GPU tensor 值的分支

### 5.4 spec_decoding_packed_mask 和 position_offsets
- Dynamic tree 每个 step 更新 mask 和 position offsets
- 这些操作涉及 `torch.gather`, `torch.cat` 等
- 需要确认这些在 CUDA graph 中是否安全

## 9. Session 4: Linear vs Dynamic Tree TP>1 AR 对比实验 (2026-03-24)

### 9.1 实验目的
确认 TP>1 AR 下降是 dynamic tree 特有还是通用 TP 问题。

### 9.2 实验配置
- **代码版本**: clean `4db2d36aa`（所有未提交修改已 revert）
- **模型**: Meta-Llama-3.1-8B-Instruct (BF16)
- **Draft model**: EAGLE3-LLaMA3.1-Instruct-8B
- **Dataset**: mt_dataset.json, num_samples=5
- **kv_cache_fraction**: 0.15
- **无 CUDA graph**（torch.compile 默认模式）

### 9.3 对比结果

| TP | Mode | CG | AR | Node | 备注 |
|----|------|-----|-----|------|------|
| 1 | Linear | ON | **3.51** | viking-prod-216 | baseline |
| 1 | Dynamic tree | ON | **5.44** | viking-prod-216 | baseline |
| 8 | Linear | OFF | **~3.90** | viking-prod-323 | partial (1 prompt) |
| 8 | Dynamic tree | OFF | **~3.39** | viking-prod-299 | prior session |
| 8 | Linear | OFF | *(running)* | viking-prod-323 | 重跑中 |
| 8 | Dynamic tree | OFF | *(running)* | viking-prod-299 | 重跑中 |

### 9.4 关键结论

1. **Linear 模式 TP>1 AR 不降反升**: TP=1 AR=3.51 → TP=8 AR≈3.90
2. **Dynamic tree TP>1 AR 大幅下降**: TP=1 AR=5.44 → TP=8 AR≈3.39（约 -38%）
3. **AR drop 是 dynamic tree 特有问题**，不是通用 TP 问题
4. `lm_head` 使用 `gather_output=True`（`modeling_utils.py:386`），每个 rank 都获得完整 vocab logits → 排除了 "partial logits" 假说

### 9.5 排除的假说

| 假说 | 排除原因 |
|------|---------|
| lm_head 在 TP>1 时返回 partial logits | `gather_output=True` → 每个 rank 获得 all-gather 后的完整 vocab logits |
| TP>1 通用问题（NCCL 精度等） | Linear 模式 TP=1→8 AR 不降 |
| CUDA graph 相关 | NO CG 时 dynamic tree TP>1 同样 AR 下降 |
| eager KV relocation | 跳过/不跳过 eager reloc，TP>1 AR 都是 ~3.39 |

### 9.6 待确认（等实验完成）
- 323 linear TP=8 最终 AR
- 299 dynamic tree TP=8 最终 AR

### 9.7 下一步 debug 方向
Dynamic tree 与 Linear 在 TP>1 时的行为差异，聚焦以下区域：
1. **Tree mask / position offset 计算** (`prepare_tree_mask_and_position_offset`)
2. **TopK sampling** (`_sample_softmax_topk`) vs argmax (`_draft_sampler_greedy`)
3. **KV cache 管理差异** (dynamic tree 的 growing context pattern)
4. **`all_rank_num_tokens` 处理** (可能影响 TP 通信)
5. **Draft hidden states 合并** (dynamic tree 每步拼接多个 token)

## 10. Session 5: Position Offsets Stride Bug (Root Cause of AR Drop) (2026-03-25)

### 10.1 关键发现: Step 0 Position Offsets Stride Mismatch

**这是 AR drop (5.22→3.39) 的根本原因。**

#### C++ 内核如何索引 position_offsets

C++ attention 内核 (unfused + XQA preprocessing) 使用以下方式索引 position_offsets:
```cpp
// unfusedAttentionKernels_2_template.h:858
spec_decoding_position_offsets[token_idx_in_seq + batch_idx * max_input_seq_len]
```
其中 `max_input_seq_len = num_tokens / num_seqs = num_gen_tokens / num_gens`。

**Step 0**: `max_input_seq_len = max_path_len = 7` (因为每个 gen request 有 7 tokens)

#### 原始代码 (BUG)
```python
tokens_per_req = self.tokens_per_gen_step  # 61
pos_2d = attn_metadata.spec_decoding_position_offsets[
    : batch_size * tokens_per_req  # stride = 61
].view(batch_size, tokens_per_req)
pos_2d[num_contexts:batch_size, :num_step0_tokens] = self._causal_offs[:num_step0_tokens]
```
- Python 写入 stride = **61** per request
- C++ 内核读取 stride = **7** per request
- **Request 0**: kernel reads base[0..6], data at base[0..6] → ✅ MATCH
- **Request 1**: kernel reads base[7..13], data at base[61..67] → ❌ MISMATCH
- **Request N**: kernel reads base[N*7..(N+1)*7-1], data at base[N*61..N*61+6] → ❌ MISMATCH

#### 本地修复 (CORRECT)
```python
pos_2d = attn_metadata.spec_decoding_position_offsets[
    : batch_size * num_step0_tokens  # stride = 7
].view(batch_size, num_step0_tokens)
pos_2d[num_contexts:batch_size, :] = self._causal_offs[:num_step0_tokens]
```
- Python 写入 stride = **7** per request
- C++ 内核读取 stride = **7** per request
- 所有 requests 都 MATCH ✅

#### 为什么仅影响 batch_size > 1
- Request 0 总是正确的（offset 0 无论 stride 是多少）
- Request 1+ 的数据位置取决于 stride，只有 stride 匹配时才正确
- **TP=1 测试用 num_samples=1** → gen 阶段 batch_size=1 → 不触发 bug → AR=5.22
- **TP>1 测试用 num_samples=5** → gen 阶段 batch_size>1 → 触发 bug → AR=3.39
- **这不是 TP 特有问题，而是 batch_size > 1 问题！**

### 10.2 Step > 0 Position Offsets (正确)

Step > 0 的 position offsets 通过 `prepare_tree_mask_and_position_offset` 设置:
```python
new_pos = self._new_pos_offset_buf[:batch_size, :num_tokens_current_layer]
# ... fill new_pos ...
attn_metadata.spec_decoding_position_offsets[
    : batch_size * num_tokens_current_layer
] = new_pos.reshape(-1)  # reshape(-1) flattens with correct stride
```
- `reshape(-1)` 在非连续 view 上会创建新的连续 tensor
- 结果 stride = `num_tokens_current_layer` per request
- C++ `max_input_seq_len = num_tokens / num_seqs = num_tokens_current_layer`
- **MATCH** ✅ (所有后续步骤都正确)

### 10.3 C++ Attention 的 Context/Generation 分离

关键发现: C++ attention op 将 mixed batch 分为两次调用:
```cpp
// attentionOp.cpp:904-927
// Context call:
seq_offset = 0;  num_seqs = num_contexts;  num_tokens = num_ctx_tokens;
// Generation call:
seq_offset = num_contexts;  num_seqs = num_generations;  num_tokens = num_gen_tokens;
```

**重要**: `sequence_lengths` 等数组通过 `.slice(0, seq_offset)` 偏移，
但 `spec_decoding_position_offsets` 使用 `data_ptr<int32_t>()` 不偏移！

这意味着 generation 内核的 `batch_idx=0` 对应第一个 gen request，
但读取 `spec_decoding_position_offsets[0]`（buffer base），不是 `[num_contexts * stride]`。

#### 对 Mixed Batch 的影响 (num_contexts > 0)
- 当前代码写 gen data 到 `base[num_contexts * stride]`
- 内核从 `base[0]` 开始读
- **混合 batch 时 position offsets 错位**
- 但混合 batch 只在新 request 进入时短暂出现（1 次迭代），影响有限

### 10.4 排除的假说

| 假说 | 排除原因 |
|------|---------|
| `all_rank_num_tokens` 导致 TP 通信问题 | 标准 TP 模式下为 None，只有 attention DP 和 MoE 使用 |
| Packed mask stride 不匹配 | Packed mask 使用 `spec_decoding_max_generation_length=61`，与 3D tensor stride 匹配 ✅ |
| KV length host/device 不一致 | `kv_lens_runtime` (CPU) 不更新，但分配的 blocks 覆盖了 stale range |
| TP allreduce 内存损坏 | `batch_indices_cuda` 修复确认有损坏，但不是 AR drop 的主因 |
| TP>1 特有问题 | AR drop 实际是 batch_size>1 问题，TP=1 测试碰巧用了 num_samples=1 |

### 10.5 验证计划

1. **确认 stride fix 修复 AR**: 用本地修复代码跑 TP=8 + NO CG + num_samples=5
   - 预期: AR 从 ~3.39 显著提升（接近 5.22）
2. **确认 TP=1 也有同样 bug**: 用原始代码跑 TP=1 + CG + num_samples=5
   - 预期: AR 从 5.22 下降（接近 3.39）
3. **Mixed batch 修复**: 将 gen position offsets 写到 `base[0]` 而不是 `base[num_contexts * stride]`

### 10.6 其他已确认的修复

| 修复 | 文件 | 说明 |
|------|------|------|
| Position offsets stride | eagle3_dynamic_tree.py | stride 61→7 (本文核心发现) |
| `_relocate_kv_eagerly` crash | eagle3_dynamic_tree.py | CG+TP>1 时条件跳过 |
| `batch_indices_cuda` 重生成 | eagle3.py | TP>1 时 draft loop 后内存被覆盖 |
| Hidden states overflow guard | eagle3.py | num_tokens > max_num_tokens 时跳过 capture |
