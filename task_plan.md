# Task Plan: SwapsMmaAb + Custom Mask for FMHA Spec-Dec Tree on Blackwell

## Goal
Enable SwapsMmaAb (Q8/Q16) kernels for EAGLE3 dynamic tree speculative decoding on Blackwell (SM100f), replacing the current KeepsMmaAb Q128 for models with sufficient numHeadsQPerKv.

## Why We Need This (nsys 数据来源: Qwen3-235B NVFP4 B200 TP8, V6)
**数据路径**: `/home/scratch.qgai_sw/qgai/project/qgai/spec-benchmarks/results/trtllm/nsys_res/Qwen3-235B-A22B-NVFP4_V6/`

| 指标 | Dynamic Tree (当前) | Linear (对比) |
|------|:---:|:---:|
| FMHA kernel | KeepsMmaAb **Q128** Custom | SwapsMmaAb **Q8** Causal |
| FMHA avg latency | **14.9 us/call** | **6.5 us/call** |
| FMHA total time | 1132 ms (13.2% GPU) | 490 ms (7.3% GPU) |
| Request latency | 31137 ms | 27993 ms |
| Output TPS | 65.8 tok/s | 73.2 tok/s |

**KeepsMmaAb Q128 比 SwapsMmaAb Q8 每次调用慢 2.3x**，因为:
- Q128 tile 处理 128 个 Q 位置，但 EAGLE3 spec-dec 每次只有 ~10 个 draft tokens × numHeadsQPerKv 个 heads
- 对于 numHeadsQPerKv=2 (Qwen3 TP8)：实际只有 ~20 个有效 Q 位置，Q128 tile 浪费 108/128 = **84% 计算**
- SwapsMmaAb Q8 tile 紧凑贴合，几乎无浪费

**预期收益**: FMHA 从 1132ms → ~490ms，节省 **642ms/请求** (~2% 端到端延迟)
- 对于更短的输出序列，FMHA 占比更高，收益更大
- Dynamic tree 当前比 linear 慢 10.1%，其中 FMHA 贡献了相当一部分

## Status: KeepsMmaAb Q128 VERIFIED (AR=4.51), SwapsMmaAb needs groupsTokensHeadsQ=false cubins

### 当前阻塞 (2026-04-13 latest)
committed bf16 Custom SwapsMmaAb cubins 全部是 `groupsTokensHeadsQ=true`:
- SwapsMmaAb 应该用 `groupsTokensHeadsQ=false` (FmhaAutoTuner.cpp:462-469)
- `groupsTokensHeadsQ=true` → `numGroupedHeads = min(tileSizeQ, numHeadsQPerKv)` → TMA mismatch
- `groupsTokensHeadsQ=false` → `numGroupedHeads = tileSizeQ` (forced) → 不依赖 numHeadsQPerKv
- **需要用 `/trtllm-gen-export` 重新生成 bf16 cubins with groupsTokensHeadsQ=false + -isCustomSpecDecodingGen true**

### 已完成的代码改动 (可直接用，等正确 cubins)
1. `prepareCustomMask.cu`: SwapsMmaAb 分支已添加（offset kernel + mask prep kernel + dispatch）
2. `trtllm.py`: buffer allocation 已更新为 `max(keeps_size, swaps_size)`
3. `fmhaKernels.h`: SwapsMmaAb Q8/Q16/Q32 heuristic 已实现（当前回退到 KeepsMmaAb Q128）
4. `xqaDispatcher.cpp`: mMaxSeqLenQ cap + numCtasPerSeqQ cap 已实施

## Skill Toolchain (use these, don't hand-write commands)
| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `/trtllm-dev-env` | 申请 B200/H200 GPU 节点、启动容器、执行远程命令 | 需要 GPU 时自动申请 |
| `/compile` | 在容器内编译 TRT-LLM (build_wheel.py) | 修改 C++ 代码后 |
| `/trtllm-gen-export` | 在 trtllm-gen 容器中生成 FMHA cubin、copy 到 TRT-LLM | 需要新 cubin 时 |
| `/trtllm-gen-env` | 设置 trtllm-gen 开发环境 (Docker, set_env.sh, artifacts) | 首次在新节点上生成 cubin |
| `/eagle_test llama llama70b` | 运行 EAGLE3 spec-dec 测试验证 AR | 编译完成后验证正确性 |

---

## Phase 1: Cubin Generation ✅ COMPLETE
- Added SwapsMmaAb + Custom configs to ExportCubin.py (GroupsTokensHeadsQ enabled/disabled)
- Generated 144 cubins on B200 via trtllm-gen
- Copied cubins to TRT-LLM, updated kernelMetaInfo.h (432 lines: 288 extern + 144 meta)

## Phase 2: Runtime Mask Preparation ✅ COMPLETE
- Added `prepareCustomMaskBuffersKernelForSwapsMmaAb` (simple packed layout)
- Added `computeCustomMaskOffsetsKernelForSwapsMmaAb` (separate offset kernel)
- Added `launchPrepareCustomMaskForSwapsMmaAb` launcher
- Updated `runPrepareCustomMask` dispatch with SwapsMmaAb branch
- Python buffer sizing in `trtllm.py`: `max(keeps_size, swaps_size)`

## Phase 3: Kernel Selection Heuristic ✅ COMPLETE
- User's linter added `mNumHeadsQPerKv` to KernelMetaInfo struct + hash (bits 57-61)
- Heuristic: SwapsMmaAb when numHeadsQPerKv >= tileSizeQ, else KeepsMmaAb Q128
- `selectKernelParams.mNumHeadsQPerKv = tileSizeQ` for SwapsMmaAb (match compile-time cubin)

## Phase 4: Testing & Debugging ✅ COMPLETE (identified root cause)

### KeepsMmaAb Results (correct)
| Model | numHeadsQPerKv | AR (mean/global) | Status |
|-------|:-:|:-:|:-:|
| LLaMA-8B TP1 draft60 | 4 | 4.85 / 4.93 | ✅ Above baseline (4.74) |
| LLaMA-8B TP1 draft30 | 4 | 4.50 / 4.40 | ✅ Above baseline (4.45) |
| LLaMA-70B TP2 draft60 | 6 | 2.17 / 1.39 | ✅ Correct text, new baseline |

### SwapsMmaAb Results (broken)
| Model | numHeadsQPerKv | AR (mean/global) | Root Cause |
|-------|:-:|:-:|:-:|
| LLaMA-70B TP2 d60 | 6 (cubin=8) | 1.54 / 1.35 | numGroupedHeads mismatch |
| LLaMA-70B TP2 d30 | 6 (cubin=8) | 1.06 / 1.04 | Same |
| LLaMA-70B d60 all-1 mask | 6 (cubin=8) | 1.59 / 1.32 | Mask irrelevant, kernel issue |

## Phase 5: Generate Per-numHeadsQPerKv Cubins ❌ BLOCKED — next step

### Root Cause (CONFIRMED)
Cubin compiled with `numHeadsQPerKv = tileSizeQ` (Q8 → 8).
`makeTmaShapeStrideQ`: `numGroupedHeads = min(tileSizeQ, numHeadsQPerKv)`.
- Compile: min(8, 8) = 8
- Runtime LLaMA-70B EAGLE3 TP2: min(8, **6**) = 6 → **TMA shape mismatch**

### EAGLE3 one-model head counts (critical discovery)
| Model | Base Q heads | EAGLE3 Q heads | KV heads | TP | Runtime numHeadsQPerKv |
|-------|:-:|:-:|:-:|:-:|:-:|
| LLaMA-8B | 32 | 32 (same) | 8 | 1 | **4** |
| LLaMA-70B | 64 | **48** (different!) | 8 | 2 | **6** |

### Next Steps
1. **ExportCubin.py**: Generate cubins for numHeadsQPerKv = {4, 6, 8, 12, 16} per tileSizeQ
   - Iterate over `numHeadsQPerKvList` in `generate()` function
   - Each produces a cubin with distinct `mNumHeadsQPerKv` in metadata
2. **kernelMetaInfo.h**: Populate `mNumHeadsQPerKv` field (already added by linter)
3. **Rebuild trtllm-gen**: `ninja Fmha` on B200, then ExportCubin.py
4. **copy_cubins.sh + update_meta_info.py**: Copy new cubins to TRT-LLM
5. **Compile + test**: Build TRT-LLM, run `/eagle_test llama llama70b`
6. **Verify mask layout**: If AR still low with matching numHeadsQPerKv, debug mask

---

## Causal Mask Heuristic (目标: Custom mask 最终要对齐这个规则)

### GQA 模型 (非 MLA) — `selectGqGenerationKernel()` (fmhaKernels.h:800-826)
`numTokensHeadsQ = numHeadsQPerKv * maxSeqLenQ`

| numTokensHeadsQ | tileSizeQ | kernelType | 典型场景 |
|:-:|:-:|:-:|:-:|
| ≤ 8 | 8 | **SwapsMmaAb** | numHeadsQPerKv=4, maxSeqLenQ=1 (LLaMA-8B) |
| ≤ 16 | 16 | **SwapsMmaAb** | numHeadsQPerKv=8, maxSeqLenQ=1 (LLaMA-70B) |
| ≤ 32 | 32 | **SwapsMmaAb** | numHeadsQPerKv=16, maxSeqLenQ=1 (Qwen3) |
| ≤ 64 | 64 | KeepsMmaAb | numHeadsQPerKv=32 |
| > 64 | 128 | KeepsMmaAb | numHeadsQPerKv=64+ |

**注意**: Causal mask 用 `groupsTokensHeadsQ=False` → `numGroupedHeads = tileSizeQ` (强制)
→ 不依赖 numHeadsQPerKv → 一套 cubin 适用所有模型

### MLA 模型 — `selectMlaGenerationKernel()` (fmhaKernels.h:571-618)
| numHeadsQPerKv | tileSizeQ | kernelType |
|:-:|:-:|:-:|
| ≤ 8 | 8 | SwapsMmaAb |
| ≤ 32 | 16 | SwapsMmaAb |
| > 32 | 64 | KeepsMmaAb (2CTA for 128) |

### Custom Mask (spec-dec tree) — 当前 vs 目标
**当前**: 始终 KeepsMmaAb Q128 + `groupsTokensHeadsQ=True`（安全但浪费 84% 计算）
**目标**: 和 Causal mask 一样按 `numTokensHeadsQ` 切换，但用 `groupsTokensHeadsQ=True`
**挑战**: `groupsTokensHeadsQ=True` 让 numGroupedHeads 依赖 numHeadsQPerKv → 需 per-numHeadsQPerKv cubins

#### `groupsTokensHeadsQ` 是什么？(trtllm-gen 源码验证)

**`groupsTokensHeadsQ` 不是正确性要求，而是优化选择。** Custom mask 两种配置都能正确工作：

| 配置 | Q tile 含义 | 每 CTA 处理 | 适用场景 |
|------|------------|------------|---------|
| `groupsTokensHeadsQ=false` | tileSizeQ 个 heads（同 1 个 token） | 1 token × tileSizeQ heads | SwapsMmaAb (spec-dec), maxSeqLenQ=1 |
| `groupsTokensHeadsQ=true` | tokens × heads 混合打包 | tileSizeQ/numHeadsQPerKv tokens × numHeadsQPerKv heads | KeepsMmaAb (spec-dec), 多 token 批处理 |

**trtllm-gen FmhaAutoTuner.cpp:462-469 的逻辑**:
```cpp
// SwapsMmaAb specDecoding: 每 CTA 只处理 1 个 token
mSingleTokenQPerCta = isSwapsMmaAb || (isMlaGen && clusterDimX == 2);
mGroupsTokensHeadsQ = !mSingleTokenQPerCta;
```
→ **SwapsMmaAb 天然用 `groupsTokensHeadsQ=false`**（每 CTA 1 token × tileSizeQ heads）
→ **KeepsMmaAb 用 `groupsTokensHeadsQ=true`**（多 token+heads 打包进大 tile 更高效）

**为什么 KeepsMmaAb 需要 `groupsTokensHeadsQ=true`？** Tile 利用率:
  - Q128 + false: 每 CTA 1 token × 128 heads → numHeadsQPerKv=4 时只用 4/128 = **3%** → 极度浪费
  - Q128 + true: token×head 打包 → numHeadsQPerKv=4, seqLenQ=5 → 20 位置 / 128 = **16%** → 好得多
**为什么 SwapsMmaAb 用 `groupsTokensHeadsQ=false`？**
  - Q8 + false: 每 CTA 1 token × 8 heads → numHeadsQPerKv=4 时 4/8 = **50%** → 已经很好
  - Q8 + true: 会引入 numGroupedHeads 依赖 → 我们踩的坑

**PR #8975 历史**: 当时 SM100f 没有 groupsTokensHeadsQ=true 的 Custom cubins，
所以生成了 KeepsMmaAb Q128 + groupsTokensHeadsQ=true 的 cubins。
而 SwapsMmaAb + Custom 应该用 `groupsTokensHeadsQ=false`（和 Causal mask 一样）。

**重要修正**: 这意味着 SwapsMmaAb + Custom 的 `numGroupedHeads` 问题可能不存在！
因为 `groupsTokensHeadsQ=false` 时 kernelParams.h line 296 强制 `numGroupedHeads = tileSizeQ`，
和 Causal mask 完全一样 → 不依赖 numHeadsQPerKv → **不需要 per-numHeadsQPerKv cubins！**

**已验证**: 我们之前生成的 144 个 SwapsMmaAb + Custom cubins **全部是 groupsTokensHeadsQ=true**！
这是错误的。SwapsMmaAb 应该用 `groupsTokensHeadsQ=false`。

**这就是 root cause！** groupsTokensHeadsQ=true 导致:
- `numGroupedHeads = min(tileSizeQ, numHeadsQPerKv)` → 依赖 numHeadsQPerKv → TMA mismatch
改为 groupsTokensHeadsQ=false 后:
- `numGroupedHeads = tileSizeQ` (强制) → 和 Causal mask 一样 → 不依赖 numHeadsQPerKv
- **不需要 per-numHeadsQPerKv cubins！一套 cubin 适用所有模型！**

### 修复方案 (简化版) — 已发现新问题！
1. ✅ ExportCubin.py: SwapsMmaAb + Custom 配置改为 `groupsTokensHeadsQList=[False]`
2. ✅ 重新生成 cubins (96 sm100f + 24 fp16)
3. ✅ 更新 kernelMetaInfo.h (384 lines)
4. ✅ 更新 heuristic (fmhaKernels.h): Causal mask 风格切换

5. ❌ **新问题: groupsTokensHeadsQ=false 导致 grid size 爆炸 + OOM**
   `numCtasPerSeqQ = maxSeqLenQ = 131072`（line 390: 每个 token 一个 CTA）
   vs KeepsMmaAb true: `numCtasPerSeqQ = maxSeqLenQ / 32 = 4096`
   → **32x 更多 CTA → autotuner warmup hang（GPU 100% 跑数小时）**

   **方案 D 已实施** (cap numCtasPerSeqQ in computeNumCtas):
   - 在 `computeNumCtas` 中对 `mIsSpecDecTree && !groupsTokensHeadsQ` 加 cap:
     `numCtasPerSeqQ = min(maxSeqLenQ, mPackedMaskMaxSeqLenQ)` (fallback 128)
   - 结果: 不再 hang，但进程被 **Killed (OOM)**
   - 原因: CUDA graph warmup 为 SwapsMmaAb Q8 创建太多 graph 变体
     - SwapsMmaAb Q8 + false: 每 token 一个 CTA → 不同 seqLenQ 需要不同 grid → 更多 CUDA graph 实例
     - KeepsMmaAb Q128 + true: 批处理多 token → grid 变化少 → 更少 CUDA graph 实例

6. ✅ **xqaDispatcher cap**: 在 xqaDispatcher.cpp:487 加 cap:
   `mMaxSeqLenQ = min(generation_input_length, spec_decoding_max_generation_length)`
   → 解决 warmup 和 autotuner 的 grid size 爆炸 + OOM

7. ✅ **AR 验证通过 (nograph)**:
   - LLaMA-8B TP1 draft60 nograph: **AR = 4.85 / 4.93** ✅ (和 KeepsMmaAb Q128 完全一致！)
   - SwapsMmaAb Q8 + Custom mask + groupsTokensHeadsQ=false 正确工作

8. ❌ **SwapsMmaAb Q8 Custom cubin GPU hang — ROOT CAUSE FOUND!**

   **Root cause**: ExportCubin.py line 1003 用 `-isCausalSpecDecodingGen true` 生成所有 gen cubins，
   包括 Custom mask cubins。但 Custom mask 应该用 `-isCustomSpecDecodingGen true`。

   两个 flag 在 FmhaAutoTuner.cpp 中走不同路径：
   - `isCausalSpecDecodingGen`: line 460 → `mSingleTokenQPerCta = !mGroupsTokensHeadsQ`
     - line 521: 可能把 maskType 从 Custom 改为 Causal（如果 groupsTokensHeadsQ=true + dense）
     - line 782 不走 → mCtaDim 可能是 384（3 warpgroups）
   - `isCustomSpecDecodingGen`: line 462 → `mSingleTokenQPerCta = isSwapsMmaAb || ...`
     - line 782: 强制 `mCtaDim = 512`（4 warpgroups）
     - 不会改变 maskType

   **结果**: cubin 编译为 3 warpgroups (CTA=384)，但运行时用 Custom mask 参数启动
   （可能需要 4 warpgroups = 512）→ barrier 不匹配 → **GPU hang**

   **修复**: ExportCubin.py 对 Custom mask cubins 需要传 `-isCustomSpecDecodingGen true`
   而不是 `-isCausalSpecDecodingGen true`。

   **已修复**: ExportCubin.py 的 linter 已更新为 Custom mask → `-isCustomSpecDecodingGen true`
   **已重新生成**: 96 sm100f cubins with `-isCustomSpecDecodingGen true` + `groupsTokensHeadsQ=false`
   **fmhaKernels.h 已重写**: linter 用 FmhaAutoTuner + FmhaInterface 替代手写 kernel selection
   → kernel selection / parameter setup / launch 全部走 trtllm-gen 的 AutoTuner 路径
   → 不再需要手写 heuristic / computeNumCtas / grid cap
   **构建阻塞**: 链接错误:
   1. `undefined reference: fmha::FmhaInterface::setVerbose` → libTrtLlmGen.a 版本/header 不匹配
   2. `undefined reference: Sm100aKernel_*_cubin_len` → sm100a cubin .cpp 缺失
   需要:
   - 重新 build trtllm-gen 的 libTrtLlmGen.a (包含 FmhaInterface 等)
   - 或者在 trtllm-gen container 中 `ninja TrtLlmGen` 后 copy .a 文件
   - sm100a cubin .cpp 也需要生成/复制
   **Root cause**: 用户 linter 更新了 `trtllmGen_fmha_export/FmhaInterface.h` (添加 `setVerbose`)
   但 `libTrtLlmGen.a` (从 trtllm-gen build/src/ 复制) 没有这个符号。
   Header 版本 > .a 版本。需要从**同一版本**的 trtllm-gen rebuild .a。

   **待做**:
   - 确认 trtllm-gen-new repo 中 FmhaInterface.cpp 是否有 setVerbose 实现
   - 如果没有，需要 pull 最新 trtllm-gen 代码
   - 或者 revert header 到匹配 .a 的版本

### 混合精度 (dtypeQ ≠ dtypeKv) — fmhaKernels.h:791-796
- 不支持 groupsTokensHeadsQ=True
- 始终 SwapsMmaAb, tileSizeQ = numHeadsQPerKv ≤ 8 ? 8 : 16

---

## Reference: How Existing Kernel+Mask Combos Work (MUST align with these)

### 1. SwapsMmaAb + Causal mask (non-spec-dec, maxSeqLenQ=1) ✅ WORKS
**生成**: ExportCubin.py line 957: `-numHeadsQ tileSizeQ*2 -numHeadsKv 2` → numHeadsQPerKv=tileSizeQ
**运行时 TMA**: `groupsTokensHeadsQ=False` → kernelParams.h line 296 强制 `numGroupedHeads = tileSizeQ`
**为什么不怕 mismatch**: 无论运行时 numHeadsQPerKv 多少，numGroupedHeads 始终 = tileSizeQ → 与 cubin 编译时一致
**Metadata**: `mGroupsHeadsQ=true, mGroupsTokensHeadsQ=false`, hash 中 `mNumHeadsQPerKv=0`（any）
**参考文件**: ExportCubin.py configs (Causal/Dense/SlidingOrChunkedCausal maskType)

### 2. KeepsMmaAb + Custom mask (spec-dec tree, PR #8975) ✅ WORKS
**生成**: ExportCubin.py 不设 `-numHeadsQ/-numHeadsKv` → 默认 numHeadsQ=4, numHeadsKv=4, numHeadsQPerKv=1
**运行时 TMA**: `groupsTokensHeadsQ=True` → `numGroupedHeads = min(128, numHeadsQPerKv) = numHeadsQPerKv`
**为什么不怕 mismatch**: Q128 tile 足够大，numGroupedHeads=numHeadsQPerKv (1~16)，kernel 代码不依赖 numGroupedHeads 的具体值（KeepsMmaAb 的 SMEM 布局是线性的）
**Metadata**: `mGroupsHeadsQ=true, mGroupsTokensHeadsQ=true`, hash 中 `mNumHeadsQPerKv=0`（any）
**Mask prep**: `prepareCustomMaskBuffersKernelForKeepsMmaAb` in prepareCustomMask.cu
**参考文件**: PR #8975, fmhaKernels.h 原始 spec-dec tree 选择逻辑

### 3. SwapsMmaAb + Custom mask (spec-dec tree, 我们在做的) ❌ BROKEN
**生成**: 同 Causal 的 `-numHeadsQ tileSizeQ*2 -numHeadsKv 2` → numHeadsQPerKv=tileSizeQ
**运行时 TMA**: `groupsTokensHeadsQ=True` → `numGroupedHeads = min(tileSizeQ, numHeadsQPerKv)`
**问题**: 当 numHeadsQPerKv < tileSizeQ 时 numGroupedHeads ≠ tileSizeQ → TMA shape mismatch
  - SwapsMmaAb 的 TMEM 布局严格依赖 numGroupedHeads（swapped MMA 操作数映射）
  - 不像 KeepsMmaAb 的线性 SMEM 可以容忍不同的 numGroupedHeads

**为什么 SwapsMmaAb 依赖 numGroupedHeads 而 KeepsMmaAb 不依赖**:
  关键代码: `kernelParams.h:292` → `numTokensPerCtaQ = tileSizeQ / numGroupedHeads`
  TMA tileShape = `[headDim, numGroupedHeads, 1, numTokensPerCtaQ]`
  - KeepsMmaAb Q128: `numGroupedHeads × numTokensPerCtaQ` = 1×128 = 4×32 = 8×16 = **始终 128**
    → 不管 numGroupedHeads 怎么变，TMA 加载的元素总数不变 → cubin 代码不受影响
  - SwapsMmaAb Q8: 当 numHeadsQPerKv=6 时 `numGroupedHeads=6, numTokensPerCtaQ=8/6=1(截断)`
    → TMA 加载 headDim×6×1 = **6 个 Q 位置**，但 cubin 期望 8 个 → 读到垃圾数据
  - 核心问题: tileSizeQ 太小 (8/16)，当 `tileSizeQ % numHeadsQPerKv ≠ 0` 时截断严重

**能否让 SwapsMmaAb 不依赖 numGroupedHeads？**
  - 方案 B (pad Q tensor): 在 Q 投影后插零使 numHeadsQPerKv 对齐 tileSizeQ → 太侵入，需改 attention module
  - 方案 C (改 trtllm-gen codegen): 让 kernel 把 Q tile 当 flat 1D → SwapsMmaAb MMA 操作数映射依赖 head 分组，大工程
  - 不能像 Causal 强制 numGroupedHeads=tileSizeQ：Custom 需多 token，强制会跨 KV group 读错误 head
  - **结论: 方案 A (per-numHeadsQPerKv cubins) 是当前最实际路径**

**修复方案 A**: 生成 per-numHeadsQPerKv cubins
  - 每个 tileSizeQ 生成多个 cubin，对应不同 numHeadsQPerKv 值
  - hash 中用 `mNumHeadsQPerKv`（已加 bits 57-61）区分
  - 运行时 `selectKernelParams.mNumHeadsQPerKv = numHeadsQPerKv` 选匹配 cubin
  - ExportCubin.py: `-numHeadsQ numHeadsQPerKv*K -numHeadsKv K` (保证 numHeadsQPerKv = 目标值)
  - 需要的 numHeadsQPerKv 值: {4, 6, 8, 12, 16} (覆盖 LLaMA-8B/70B EAGLE3, Qwen3 等)

## trtllm-gen Source Code
- **路径**: `/home/scratch.qgai_sw/qgai/project/qgai/trtllm-gen-new`
- **ExportCubin.py**: `kernels/Fmha/tools/ExportCubin.py` (cubin 生成配置)
- **Fmha.cpp**: `kernels/Fmha/Fmha.cpp` (test harness + mask reference)
- **KernelParams.h**: `kernels/Fmha/KernelParams.h` (TMA shape + kernel params)
- **KernelConfigBase.h**: `kernels/Fmha/KernelConfigBase.h` (mNumHeadsQ default=4)
- **Mask.h**: 内核 mask 代码生成 (`generateApplyCustomMaskCodeForSwappedAb`)

## Key Technical Insights
- All-1 mask test proved mask layout is NOT the issue (mask 布局不是问题)
- KeepsMmaAb Q128 的 TMA tile 总元素 = tileSizeQ × headDim，不随 numGroupedHeads 变化
- SwapsMmaAb Q8/Q16 的 TMA tile 元素 = numGroupedHeads × numTokensPerCtaQ × headDim，当 tileSizeQ 不能被 numGroupedHeads 整除时截断

## Errors Encountered
| Error | Resolution |
|-------|------------|
| ABI mismatch (wrong Docker image) | Use `jenkins/current_image_tags.properties → LLM_DOCKER_IMAGE` |
| xgrammar FetchContent patch failure | `bash fix_dep.sh xgrammar` |
| CMakeCache deleted by fix_torch_abi | Use `--configure_cmake` flag |
| memset 0xFF CUDA error | Buffer size overflow — can't blindly memset allocated region |
| numHeadsQPerKv=8 assumed for 70B | Actually 6 (EAGLE3 draft has 48 Q heads, not 64) |

## Files Modified
- `fmhaKernels.h` — heuristic + hash (mNumHeadsQPerKv in bits 57-61)
- `prepareCustomMask.cu` — SwapsMmaAb mask kernels
- `cubin/kernelMetaInfo.h` — 432 new lines + mNumHeadsQPerKv field
- `cubin/*.cpp` — 144 new cubin files
- `trtllm.py` — buffer sizing
- `ExportCubin.py` / `ExportCubin_custom_gqa.py` — SwapsMmaAb configs
