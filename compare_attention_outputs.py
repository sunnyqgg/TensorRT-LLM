#!/usr/bin/env python3
"""
对比 H100 和 B100 的 attention output tensor 相似度
支持 .safetensors 和 .bin 文件格式
"""

import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file


def load_bin_file(bin_path, shape, dtype='bfloat16', device='cpu'):
    """
    加载二进制文件为 PyTorch tensor

    Args:
        bin_path: 二进制文件路径
        shape: tensor 的 shape，如 (25, 4096)
        dtype: 数据类型，如 'bfloat16', 'float16', 'float32'
        device: 设备，'cpu' 或 'cuda'

    Returns:
        PyTorch tensor
    """
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }

    dtype_np_map = {
        'bfloat16': np.uint16,  # bfloat16 需要特殊处理
        'float16': np.float16,
        'float32': np.float32,
        'fp16': np.float16,
        'fp32': np.float32,
        'bf16': np.uint16,
    }

    dtype_map.get(dtype, torch.bfloat16)
    np_dtype = dtype_np_map.get(dtype, np.uint16)

    # 读取二进制文件
    with open(bin_path, 'rb') as f:
        data = np.fromfile(f, dtype=np_dtype)

    # 检查数据大小是否匹配
    expected_size = np.prod(shape)
    if len(data) != expected_size:
        print(f"⚠️  警告: 文件大小不匹配! 期望 {expected_size} 个元素，实际 {len(data)} 个元素")
        print(f"   尝试自动推断 shape...")
        # 尝试推断 shape
        if len(data) % 4096 == 0:
            inferred_shape = (len(data) // 4096, 4096)
            print(f"   推断 shape 为: {inferred_shape}")
            shape = inferred_shape
        else:
            print(f"   无法推断合理的 shape，使用原始 shape 并截断/填充")
            if len(data) < expected_size:
                # 填充 0
                data = np.pad(data, (0, expected_size - len(data)),
                              constant_values=0)
            else:
                # 截断
                data = data[:expected_size]

    # Reshape
    data = data.reshape(shape)

    # 转换为 PyTorch tensor
    if dtype in ['bfloat16', 'bf16']:
        # bfloat16 需要特殊处理：将 uint16 视图转换为 bfloat16
        tensor = torch.from_numpy(data).view(torch.bfloat16)
    else:
        tensor = torch.from_numpy(data)

    return tensor.to(device)


def load_safetensors_file(st_path):
    """
    加载 safetensors 文件

    Args:
        st_path: safetensors 文件路径

    Returns:
        PyTorch tensor (第一个 tensor)
    """
    data = load_file(st_path)
    if len(data) > 1:
        print(f"⚠️  警告: 文件包含多个 tensor: {list(data.keys())}，将使用第一个")
    return list(data.values())[0]


def load_tensor_file(file_path, shape=None, dtype='bfloat16', device='cpu'):
    """
    通用文件加载函数，根据扩展名自动选择加载方式

    Args:
        file_path: 文件路径
        shape: 仅对 .bin 文件需要，tensor 的 shape
        dtype: 仅对 .bin 文件需要，数据类型
        device: 设备

    Returns:
        PyTorch tensor
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.bin':
        if shape is None:
            raise ValueError(f"加载 .bin 文件需要指定 shape 参数")
        print(f"✓ 加载 .bin 文件: {file_path} (shape={shape}, dtype={dtype})")
        return load_bin_file(file_path, shape, dtype, device)
    elif ext == '.safetensors':
        print(f"✓ 加载 .safetensors 文件: {file_path}")
        tensor = load_safetensors_file(file_path)
        return tensor.to(device)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 .bin 和 .safetensors")


def print_tensor_stats(tensor, name):
    """打印 tensor 的统计信息"""
    print(f"\n{name} 统计信息:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
    print(f"  Inf count: {torch.isinf(tensor).sum().item()}")


def compute_similarity_metrics(actual, expected, name="Comparison"):
    """计算详细的相似度指标"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    # 确保在同一设备上
    if actual.device != expected.device:
        actual = actual.to(expected.device)

    # 转换为 float32 进行精确计算
    actual_f32 = actual.float()
    expected_f32 = expected.float()

    # 1. 绝对差异
    abs_diff = (actual_f32 - expected_f32).abs()
    max_abs_diff = abs_diff.max()
    max_abs_idx = abs_diff.argmax()
    max_abs_idx_2d = np.unravel_index(max_abs_idx.cpu().item(), actual.shape)

    print(f"\n📊 绝对差异:")
    print(
        f"  Greatest absolute difference: {max_abs_diff:.6f} at index {max_abs_idx_2d}"
    )
    print(f"  Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"  Median absolute difference: {abs_diff.median():.6f}")
    print(f"  Std of absolute difference: {abs_diff.std():.6f}")

    # 2. 相对差异（优化版本）
    denominator = torch.maximum(expected_f32.abs(), actual_f32.abs())
    significant_mask = denominator > 1e-3

    if significant_mask.any():
        rel_diff_significant = abs_diff[significant_mask] / (
            denominator[significant_mask] + 1e-8)
        max_rel_diff_sig = rel_diff_significant.max()
        max_rel_idx_flat = torch.where(
            significant_mask.flatten())[0][rel_diff_significant.argmax()]
        max_rel_idx_2d = np.unravel_index(max_rel_idx_flat.cpu().item(),
                                          actual.shape)

        print(f"\n📈 相对差异 (仅对显著值 |value| > 0.001):")
        print(
            f"  显著元素数量: {significant_mask.sum().item()} / {actual.numel()} ({100*significant_mask.sum().item()/actual.numel():.1f}%)"
        )
        print(
            f"  Greatest relative difference: {max_rel_diff_sig:.6f} ({max_rel_diff_sig*100:.2f}%) at index {max_rel_idx_2d}"
        )
        print(
            f"  Mean relative difference: {rel_diff_significant.mean():.6f} ({rel_diff_significant.mean()*100:.2f}%)"
        )
        print(
            f"  Median relative difference: {rel_diff_significant.median():.6f} ({rel_diff_significant.median()*100:.2f}%)"
        )
    else:
        print(f"\n📈 相对差异:")
        print(f"  所有值都太小 (< 0.001)，相对差异不适用")

    # 3. 传统相对差异（用于对比）
    rel_diff_traditional = abs_diff / (expected_f32.abs() + 1e-8)
    max_rel_diff_trad = rel_diff_traditional.max()
    max_rel_idx_trad = rel_diff_traditional.argmax()
    max_rel_idx_trad_2d = np.unravel_index(max_rel_idx_trad.cpu().item(),
                                           actual.shape)

    print(f"\n⚠️  传统相对差异 (可能被小值夸大):")
    print(f"  Greatest: {max_rel_diff_trad:.6f} at index {max_rel_idx_trad_2d}")
    print(
        f"    → 期望值: {expected[max_rel_idx_trad_2d]:.6e}, 实际值: {actual[max_rel_idx_trad_2d]:.6e}"
    )

    # 4. 余弦相似度
    actual_flat = actual_f32.flatten()
    expected_flat = expected_f32.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(actual_flat.unsqueeze(0),
                                                    expected_flat.unsqueeze(0))
    print(f"\n🎯 余弦相似度: {cos_sim.item():.8f}")

    # 5. 相关系数
    actual_centered = actual_flat - actual_flat.mean()
    expected_centered = expected_flat - expected_flat.mean()
    correlation = (actual_centered * expected_centered).sum() / (
        actual_centered.norm() * expected_centered.norm() + 1e-8)
    print(f"📐 Pearson 相关系数: {correlation.item():.8f}")

    # 6. 最大绝对差异位置的详细信息
    print(f"\n🔍 最大绝对差异位置 (index {max_abs_idx_2d}):")
    print(f"  H100 值: {expected[max_abs_idx_2d]:.6f}")
    print(f"  B100 值: {actual[max_abs_idx_2d]:.6f}")
    print(f"  绝对差异: {abs_diff[max_abs_idx_2d]:.6f}")
    if denominator[max_abs_idx_2d] > 1e-3:
        rel_at_max_abs = abs_diff[max_abs_idx_2d] / (
            denominator[max_abs_idx_2d] + 1e-8)
        print(f"  相对差异: {rel_at_max_abs:.6f} ({rel_at_max_abs*100:.2f}%)")

    if significant_mask.any():
        print(f"\n🔍 最大相对差异位置 (index {max_rel_idx_2d}, 仅显著值):")
        print(f"  H100 值: {expected[max_rel_idx_2d]:.6f}")
        print(f"  B100 值: {actual[max_rel_idx_2d]:.6f}")
        print(f"  绝对差异: {abs_diff[max_rel_idx_2d]:.6f}")
        print(f"  相对差异: {max_rel_diff_sig:.6f} ({max_rel_diff_sig*100:.2f}%)")

    # 7. 差异最大的前10个位置
    print(f"\n📋 绝对差异最大的前10个位置:")
    flat_abs_diff = abs_diff.flatten()
    top_indices = flat_abs_diff.topk(10).indices
    for i, idx in enumerate(top_indices):
        idx_2d = np.unravel_index(idx.cpu().item(), actual.shape)
        denom = torch.maximum(expected[idx_2d].abs(), actual[idx_2d].abs())
        if denom > 1e-3:
            rel_pct = (abs_diff[idx_2d] / (denom + 1e-8) * 100).item()
            print(
                f"  {i+1:2d}. Index {idx_2d}: H100={expected[idx_2d]:.6f}, B100={actual[idx_2d]:.6f}, "
                f"diff={abs_diff[idx_2d]:.6f} ({rel_pct:.1f}%)")
        else:
            print(
                f"  {i+1:2d}. Index {idx_2d}: H100={expected[idx_2d]:.6f}, B100={actual[idx_2d]:.6f}, "
                f"diff={abs_diff[idx_2d]:.6f} (值太小)")

    # 8. 误差分布统计
    print(f"\n📊 误差分布:")
    total = actual.numel()
    within_001 = (abs_diff <= 0.001).sum().item()
    within_01 = (abs_diff <= 0.01).sum().item()
    within_02 = (abs_diff <= 0.02).sum().item()
    within_05 = (abs_diff <= 0.05).sum().item()
    within_1 = (abs_diff <= 0.1).sum().item()

    print(
        f"  |diff| ≤ 0.001: {within_001:6d} / {total} ({100*within_001/total:.1f}%)"
    )
    print(
        f"  |diff| ≤ 0.01 : {within_01:6d} / {total} ({100*within_01/total:.1f}%)"
    )
    print(
        f"  |diff| ≤ 0.02 : {within_02:6d} / {total} ({100*within_02/total:.1f}%)"
    )
    print(
        f"  |diff| ≤ 0.05 : {within_05:6d} / {total} ({100*within_05/total:.1f}%)"
    )
    print(
        f"  |diff| ≤ 0.1  : {within_1:6d} / {total} ({100*within_1/total:.1f}%)"
    )

    # 9. 相对误差分布（对显著值）
    if significant_mask.any():
        print(f"\n📊 相对误差分布 (仅显著值):")
        sig_count = significant_mask.sum().item()
        rel_diff_sig = abs_diff[significant_mask] / (
            denominator[significant_mask] + 1e-8)
        within_1pct = (rel_diff_sig <= 0.01).sum().item()
        within_5pct = (rel_diff_sig <= 0.05).sum().item()
        within_10pct = (rel_diff_sig <= 0.10).sum().item()

        print(
            f"  相对误差 ≤ 1% : {within_1pct:6d} / {sig_count} ({100*within_1pct/sig_count:.1f}%)"
        )
        print(
            f"  相对误差 ≤ 5% : {within_5pct:6d} / {sig_count} ({100*within_5pct/sig_count:.1f}%)"
        )
        print(
            f"  相对误差 ≤ 10%: {within_10pct:6d} / {sig_count} ({100*within_10pct/sig_count:.1f}%)"
        )

    # 10. PyTorch assert_close 测试
    print(f"\n✅ PyTorch assert_close 测试:")
    for atol, rtol in [(0.001, 0.001), (0.01, 0.01), (0.02, 0.02),
                       (0.05, 0.05)]:
        try:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            print(f"  ✓ PASS with atol={atol}, rtol={rtol}")
        except AssertionError:
            print(f"  ✗ FAIL with atol={atol}, rtol={rtol}")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='对比两个 tensor 文件的相似度（支持 .safetensors 和 .bin 格式）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对比两个 safetensors 文件
  %(prog)s h100_attention0_output.safetensors b100_attention0_output.safetensors

  # 对比 bin 文件和 safetensors 文件
  %(prog)s fmha_output_trtllm_gen.bin b100_attention0_output.safetensors --shape 25 4096

  # 指定 dtype
  %(prog)s file1.bin file2.bin --shape 25 4096 --dtype float16
        """)

    parser.add_argument('file1', help='第一个文件路径（作为期望值/参考值）')
    parser.add_argument('file2', help='第二个文件路径（作为实际值/对比值）')
    parser.add_argument('--shape',
                        nargs='+',
                        type=int,
                        help='对于 .bin 文件，指定 tensor shape，如: --shape 25 4096')
    parser.add_argument(
        '--dtype',
        default='bfloat16',
        choices=['bfloat16', 'bf16', 'float16', 'fp16', 'float32', 'fp32'],
        help='对于 .bin 文件，指定数据类型 (默认: bfloat16)')
    parser.add_argument('--device',
                        default='cpu',
                        choices=['cpu', 'cuda'],
                        help='计算设备 (默认: cpu)')

    args = parser.parse_args()

    # 处理 shape 参数
    shape = None
    if args.shape:
        shape = tuple(args.shape)

    print("🔍 加载文件...")
    print(f"  文件1 (期望/参考): {args.file1}")
    print(f"  文件2 (实际/对比): {args.file2}")

    try:
        # 加载第一个文件
        tensor1 = load_tensor_file(args.file1,
                                   shape=shape,
                                   dtype=args.dtype,
                                   device=args.device)

        # 加载第二个文件
        tensor2 = load_tensor_file(args.file2,
                                   shape=shape,
                                   dtype=args.dtype,
                                   device=args.device)

    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 打印基本信息
    print_tensor_stats(tensor1, f"文件1: {os.path.basename(args.file1)}")
    print_tensor_stats(tensor2, f"文件2: {os.path.basename(args.file2)}")

    # 检查 shape 是否匹配
    if tensor1.shape != tensor2.shape:
        print(f"\n❌ 错误: Tensor shape 不匹配!")
        print(f"  文件1 shape: {tensor1.shape}")
        print(f"  文件2 shape: {tensor2.shape}")
        sys.exit(1)

    print(f"\n✓ Shape 匹配: {tensor1.shape}")

    # 计算相似度
    comparison_name = f"{os.path.basename(args.file2)} vs {os.path.basename(args.file1)}"
    compute_similarity_metrics(tensor2, tensor1, comparison_name)

    # 如果是 2D tensor，还可以按行分析
    if len(tensor1.shape) == 2:
        print(f"\n{'='*80}")
        print("📊 按行分析（前5行和后5行）")
        print(f"{'='*80}")

        num_rows = tensor1.shape[0]
        rows_to_check = list(range(min(5, num_rows))) + list(
            range(max(num_rows - 5, 5), num_rows))

        for row_idx in rows_to_check:
            row1 = tensor1[row_idx]
            row2 = tensor2[row_idx]

            abs_diff = (row1.float() - row2.float()).abs()
            max_diff = abs_diff.max()
            mean_diff = abs_diff.mean()

            print(
                f"  Row {row_idx:2d}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            )


if __name__ == "__main__":
    main()
