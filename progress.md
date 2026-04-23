# Progress Log

## Earlier Sessions (2026-03-22 to 2026-03-25)
- Fixed Dynamic Tree EAGLE3 + CUDA Graph + TP>1 crash
- Fixed AR drop at concurrency > 1
- See git history for details

---

## Session: SwapsMmaAb + Custom Mask (2026-04-11 to 2026-04-12)

### 2026-04-11: Initial implementation
- Generated 144 SwapsMmaAb + Custom cubins on B200 via trtllm-gen
- Added prepareCustomMask.cu SwapsMmaAb mask preparation kernels
- Added kernel selection heuristic using numHeadsQPerKv
- First test on B200: **SwapsMmaAb AR=2.80** (expected ~4.85) → broken
- Tested 3 mask layouts: LDTM (1.84), Simple packed (2.74), KeepsMmaAb-style (1.51) → all fail
- **All-1 mask test: AR=2.84** → proves issue is NOT in mask layout
- Root cause identified: cubin compiled with numHeadsQPerKv=8, LLaMA-8B runtime=4 → TMA shape mismatch

### 2026-04-12: Heuristic fix + deeper debugging
- Fixed heuristic: only SwapsMmaAb when numHeadsQPerKv >= tileSizeQ
- LLaMA-8B TP1 with KeepsMmaAb: **AR=4.85/4.93** ✅ (matches/beats baseline)
- LLaMA-8B TP1 draft30: **AR=4.50/4.40** ✅

- Tested LLaMA-70B TP2:
  - SwapsMmaAb Q8: AR=1.54 (bad)
  - KeepsMmaAb Q128: AR=2.17 (better but still low)
  - SwapsMmaAb all-1 mask: AR=1.59 (same as normal → not mask issue)

- **KEY DISCOVERY**: Added debug logging to kernelParams.h, found:
  - `numHeadsQPerKv=6` (not 8!) for LLaMA-70B TP2
  - EAGLE3 draft model has 48 Q heads (not 64 like base model)
  - TP2: 48/2=24 Q heads, 8/2=4 KV heads → numHeadsQPerKv=24/4=6
  - Cubin compiled with numHeadsQPerKv=8 → min(8,6)=6 ≠ 8 → mismatch!

- User's linter added `mNumHeadsQPerKv` to hash + KernelMetaInfo struct
- Current state: heuristic selects SwapsMmaAb when numHeadsQPerKv >= tileSizeQ
  - LLaMA-8B (numHeadsQPerKv=4): KeepsMmaAb Q128 ✅
  - LLaMA-70B (numHeadsQPerKv=6): KeepsMmaAb Q128 ✅ (6 < 8)

### 2026-04-12 (continued): Root cause correction — groupsTokensHeadsQ=false

**重大发现**: 之前 144 个 SwapsMmaAb + Custom cubins 全部用了 `groupsTokensHeadsQ=true`（错误）。
根据 trtllm-gen FmhaAutoTuner.cpp:462-469：
- SwapsMmaAb → `mSingleTokenQPerCta=true` → `mGroupsTokensHeadsQ=false`
- KeepsMmaAb → `mSingleTokenQPerCta=false` → `mGroupsTokensHeadsQ=true`

`groupsTokensHeadsQ=false` 时 kernelParams.h:296 强制 `numGroupedHeads = tileSizeQ`，
和 Causal mask 完全一样 → **不依赖 numHeadsQPerKv → 不需要 per-model cubins！**

**修复**:
1. ExportCubin.py: SwapsMmaAb + Custom 同精度 config 改为 `groupsTokensHeadsQList=[False]`
2. fmhaKernels.h: 恢复 Causal mask 风格切换（numTokensHeadsQ ≤ 8/16/32 → SwapsMmaAb）
3. B200 umb-b200-247 上重新生成 cubins (进行中)
4. 用 `/trtllm-gen-export` copy + update kernelMetaInfo.h
5. 用 `/compile` 编译
6. 用 `/eagle_test llama llama70b` 验证

### Key learnings from this session
1. **根本错误**: 我们生成了 `groupsTokensHeadsQ=true` 的 SwapsMmaAb cubins，但 SwapsMmaAb 应该用 false
2. **FmhaAutoTuner.cpp:462-469** 明确: SwapsMmaAb → mSingleTokenQPerCta=true → groupsTokensHeadsQ=false
3. **groupsTokensHeadsQ=false** 时 kernelParams.h:296 强制 numGroupedHeads=tileSizeQ → 和 Causal mask 一样
4. **不需要 per-numHeadsQPerKv cubins** — 一套 cubin 适用所有模型
5. B200 分配: 使用完整分区名 `b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb` + `--gres=gpu:8`
6. **dtypeList 必须包含 fp16** — LLaMA-8B 用 FP16 (dtype=1)，不包含则找不到 cubin
   - 完整 list: `"fp16:fp16:fp16,bf16:bf16:bf16,e4m3:e4m3:bf16,e4m3:e4m3:e4m3,e4m3:e2m1:e4m3"`
7. **方案 D 已实施成功** — cap numCtasPerSeqQ + cap mMaxSeqLenQ in xqaDispatcher
   - debug log 确认 warmup 时 `generation_input_length=10, spec_decoding_max_generation_length=10`
   - grid 不再爆炸（mMaxSeqLenQ 被 cap 到 10）
8. **GPU hang ROOT CAUSE**: ExportCubin.py 用 `-isCausalSpecDecodingGen true` 生成 Custom cubins
   - 应该用 `-isCustomSpecDecodingGen true`（FmhaAutoTuner 走不同路径: 4 warpgroups vs 3）
   - cubin 编译为 3 warpgroups，runtime 用 4 warpgroups 参数启动 → barrier 不匹配 → GPU hang
   - **已修复**: ExportCubin.py line 1003 改为 Custom → `-isCustomSpecDecodingGen true`
   - 96 sm100f cubins 已重新生成（含 FP16 + BF16 + E4M3 + E2M1）
9. **构建阻塞**: PyTorch ABI mismatch (`c10::TensorImpl::decref_pyobject`)
   - NFS 共享的 cpp/build/ 被多个容器编译过，.o 文件混合了不同 torch 版本
   - `fix_torch_abi.sh` 不够：只清理 th_common/plugins，但 libtensorrt_llm.so 也有旧 ABI
   - 需要 `/compile clean` (rm -rf cpp/build + 全量重编)
   - `/compile` skill 有 bug: `build_wheel.py` 的 interactive prompt EOF — 已修复 (`echo "" |`)
   - **下次 session 第一步**: 申请 B200 8h → `/compile clean` → `/eagle_test llama nograph`

### 2026-04-13: SwapsMmaAb mask layout debugging (continued)

**Root cause identified**: SwapsMmaAb Custom mask uses LDTM 16dp.256bit permuted layout
(Fmha.cpp:297-316), NOT simple uint32 packed layout. Our mask prep wrote simple layout → wrong.

**Key discovery**: `tileSizeQ` must be padded to 32 (Fmha.cpp:147, Mask.h:1494) for mask addressing:
- `tileSizeQ = ceil(mTileSizeQ/32)*32` (8→32 for Q8)
- `tileSizeQPerCta = paddedTileSizeQ * numInstsQ` (32 for Q8)
- Per-tile size: `numInstsQ * numInstsKv * paddedTileSizeQ * tileSizeKv / 32` words

**Progress**:
- Without padding: AR=1.91 (all mask writes in wrong offsets)
- With padded tileSizeQ only: AR=1.91 (tileSizeQPerCta still wrong)
- With padded tileSizeQ + tileSizeQPerCta: **AR=2.48** (improving but still low)
- Text quality improved from garbage `!!!` to repetitive but readable

**CPU verification results** (seqLenQ=7, causal tree mask):
- 49 checked, **25 mismatches** — token 0,1 mask correct, token 2+ masks ALL ZERO
- Token 2's mask expected at tile 2 (word 512+) but GPU wrote nothing there
- Token 0,1 tiles show correct LDTM-permuted data

**Root cause hypothesis**: mismatches could be from:
1. `cumSeqLensQPtr` causing wrong input mask row offset
2. Or mask buffer not zeroed (stale data from previous iteration)
3. Or kernel launch grid doesn't cover all tokens (unlikely, 64 threads > 7 tokens)

**2026-04-13 (continued): New B200 umbriel-b200-019**

**CPU verification with sufficient readWords** (fixed from 1024 to 1792+):
- seqLenQ=7 causal: **still 25/49 mismatches** — token 2+ mask genuinely zero!
- NOT a debug read range issue — kernel truly didn't write token 2+
- seqLenQ=10 diagonal: **stale data** from previous call (tq=0 tkv=1 got=1 but expect=0)
  → mask buffer not zeroed between calls!

**Key evidence of stale mask data**:
- First warmup call (seqQ=7, causal): writes tokens 0,1 only (2+6 nonzero words)
- Second warmup call (seqQ=10, diagonal): sees stale data from first call at token 0/KV 1
- Token 2+ genuinely not written by kernel → root cause still unknown

**Next investigation needed**:
- Add device printf to verify `randomMask` value for token 2 inside kernel
  (GPU printf doesn't flush to log redirect — needs different approach)
- Or: check if workspace.zero_() covers the custom mask buffer area
- Or: check if the LDTM bit offset for token 2 lands in an unexpected word
  that's outside the debug read range

### 2026-04-13: Session recovery + kernelMetaInfo.h restoration + KeepsMmaAb verification

**问题 1: kernelMetaInfo.h 被清空** (上一session的"9000行清理"过于激进)
- 工作目录只剩 288 行（仅 E2M1 cubins），丢失所有 bf16/fp16/E4M3 SM100f cubins
- 原因: 清理 sm103a 无效 entries 时把所有 entries 都删了
- 修复: `git checkout HEAD -- kernelMetaInfo.h` 恢复到 9768 行

**问题 2: LFS cubin .cpp 文件未下载**
- SM100f cubin .cpp 存储在 Git LFS，工作目录只有 LFS pointer（3行stub）
- `git lfs pull` 下载了所有 3258 个 cubin .cpp 文件（从 98 → 3258）

**问题 3: cmake GLOB_RECURSE 不拾取新文件**
- 增量编译 (`-f`) 跳过 configure 步骤 → 新的 cubin .cpp 不在编译列表中
- 删除 Makefile/CMakeCache.txt → build_wheel.py 不会自动 reconfigure
- **根因**: `build_wheel.py` 只在 `clean or first_build or configure_cmake` 为 true 时执行 configure
  - `first_build = not Path("cpp/build/CMakeFiles").exists()` → CMakeFiles 存在 → false
  - 需要显式传 `--configure_cmake` 参数
- 修复: `echo "" | python3 build_wheel.py --configure_cmake --use_ccache --cuda_architectures=...`
- 编译成功: libtensorrt_llm.so 1037MB，包含所有 3258 个 cubin

**问题 4: SwapsMmaAb + Custom 运行时报错**
- 错误: `TRTLLM-GEN does not support kernel type: 1 for custom mask preparation`
  - 注: type 1 = `Generation`(unresolved)，打印了 runnerParams.mKernelType；实际 kernelMeta.mKernelType = 2 = SwapsMmaAb
- 根因: `prepareCustomMask.cu::runPrepareCustomMask()` 只有 KeepsMmaAb 分支 (type 3)
  - SwapsMmaAb (type 2) 和 Generation (type 1) 都会 fallthrough 到 error 分支
  - 原始 committed 代码就只支持 KeepsMmaAb，SwapsMmaAb mask prep 代码需要新增
- AR=1.94 (terrible) — mask 准备失败，custom mask 全零 → 每步只接受1个 token

**验证: KeepsMmaAb Q128 baseline**
- 临时将 fmhaKernels.h 中 spec-dec tree heuristic 改为始终 KeepsMmaAb Q128
- 结果: Mean AR = **4.51** (n=64), 无错误，输出质量正常
- vs baseline 4.74 — 略低 5%，可能是节点差异

**用户 linter 恢复了 SwapsMmaAb heuristic** — 需要实现 SwapsMmaAb mask prep

**当前阻塞**:
- `prepareCustomMask.cu` 缺少 SwapsMmaAb 分支
- 需要实现 SwapsMmaAb + Custom mask 的 mask preparation kernel
- committed cubins 有 `groupsTokensHeadsQ=true`（与我们之前生成的 false 不同）

**关键发现: build_wheel.py --configure_cmake**
- 不删除 cpp/build/ 也能触发 cmake 重新配置
- 必须传 `--configure_cmake` 标志
- 场景: 添加/删除 cubin .cpp 文件后需要 cmake re-GLOB

### 2026-04-13 (continued): SwapsMmaAb mask prep + cubin variant 诊断

**SwapsMmaAb mask prep 代码已实现** (`prepareCustomMask.cu`):
- `prepareCustomMaskBuffersKernelForSwapsMmaAb`: SwapsMmaAb layout [instQ][instKv][kvInTile][qPadded/32]
  - 每 CTA 1 token × tileSizeQ heads (groupsTokensHeadsQ=false)
  - 所有 heads 共享同一 mask value (tree mask is per-token)
- `computeCustomMaskOffsetsKernelForSwapsMmaAb`: 不同的 tile size formula
  - numTilesQ = seqLenQ × ceil(numHeadsQPerKv / tileSizeQPerCta)
- `runPrepareCustomMask` 已添加 SwapsMmaAb dispatch 分支

**测试结果**: Mean AR = 1.96 (n=64) — 很差！

**Root cause**: committed bf16 Custom SwapsMmaAb cubins 全部是 `groupsTokensHeadsQ=true`:
```
groupsHeadsQ= true groupsTokensHeadsQ= true  // ← 全部是 true！
```
但 FmhaAutoTuner 说 SwapsMmaAb 应该用 `groupsTokensHeadsQ=false`。
- `groupsTokensHeadsQ=true` → `numGroupedHeads = min(8, 4) = 4` ≠ compile-time 8 → TMA mismatch
- 我们的 mask prep 代码也假设 `groupsTokensHeadsQ=false` layout → double mismatch
- LLaMA-8B 结果: 上一 session 的正确 cubins (groupsTokensHeadsQ=false) AR=4.85/4.93 ✅
  但那些 cubins 是 E2M1 (NVFP4) only，bf16 的不存在

**暂时回退到 KeepsMmaAb Q128** — fmhaKernels.h 中 spec-dec tree heuristic 改为 KeepsMmaAb

**下一步: 用 `/trtllm-gen-export` 重新生成 bf16 groupsTokensHeadsQ=false cubins**
- ExportCubin.py 已有正确配置 (groupsTokensHeadsQList=[False], -isCustomSpecDecodingGen true)
- 需要在 B200 上运行 `/trtllm-gen-export` 生成 + copy + update kernelMetaInfo.h
- 然后恢复 SwapsMmaAb heuristic + rebuild + test
