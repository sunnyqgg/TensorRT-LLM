#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 61 个单独的 key token 文件合并成一个完整的 KV cache key 文件
"""

import argparse
import os

import numpy as np


def load_single_key_token(filepath, num_kv_heads, head_dim, dtype=np.float16):
    """
    加载单个 token 的 key 数据

    Args:
        filepath: 文件路径
        num_kv_heads: KV head 数量
        head_dim: 每个 head 的维度
        dtype: 数据类型

    Returns:
        shape [num_kv_heads, head_dim] 的 numpy 数组
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    data = np.fromfile(filepath, dtype=dtype)
    expected_elements = num_kv_heads * head_dim

    if data.size != expected_elements:
        raise ValueError(f"文件 {filepath} 的元素数量不匹配: "
                         f"期望 {expected_elements} ({num_kv_heads}×{head_dim}), "
                         f"实际 {data.size}")

    return data.reshape(num_kv_heads, head_dim)


def concat_all_keys(dump_dir,
                    total_tokens,
                    num_kv_heads,
                    head_dim,
                    dtype=np.float16):
    """
    合并所有 token 的 key 数据

    Args:
        dump_dir: dump 文件所在目录
        total_tokens: token 总数（默认 61）
        num_kv_heads: KV head 数量
        head_dim: 每个 head 的维度
        dtype: 数据类型

    Returns:
        shape [total_tokens, num_kv_heads, head_dim] 的 numpy 数组
    """
    all_keys = []

    print(f"正在加载 {total_tokens} 个 token 的 key 数据...")
    print(
        f"配置: num_kv_heads={num_kv_heads}, head_dim={head_dim}, dtype={dtype}")
    print("-" * 60)

    for token_idx in range(total_tokens):
        filename = os.path.join(dump_dir, f"key_token_idx_[{token_idx}].bin")

        try:
            key_data = load_single_key_token(filename, num_kv_heads, head_dim,
                                             dtype)
            all_keys.append(key_data)

            if token_idx % 10 == 0 or token_idx == total_tokens - 1:
                print(f"  已加载 Token {token_idx:3d}: shape={key_data.shape}, "
                      f"min={key_data.min():.4f}, max={key_data.max():.4f}, "
                      f"mean={key_data.mean():.4f}")

        except Exception as e:
            print(f"  ⚠️  加载 Token {token_idx} 失败: {e}")
            raise

    # 沿着 token 维度 concat
    concatenated_keys = np.stack(all_keys, axis=0)

    print("-" * 60)
    print(f"✅ 成功合并所有 keys!")
    print(f"   最终 shape: {concatenated_keys.shape}")
    print(f"   总大小: {concatenated_keys.nbytes / 1024:.2f} KB "
          f"({concatenated_keys.nbytes / (1024*1024):.2f} MB)")
    print(
        f"   数据范围: min={concatenated_keys.min():.4f}, "
        f"max={concatenated_keys.max():.4f}, mean={concatenated_keys.mean():.4f}"
    )

    return concatenated_keys


def concat_all_values(dump_dir,
                      total_tokens,
                      num_kv_heads,
                      head_dim,
                      dtype=np.float16):
    """
    合并所有 token 的 value 数据

    Args:
        dump_dir: dump 文件所在目录
        total_tokens: token 总数（默认 61）
        num_kv_heads: KV head 数量
        head_dim: 每个 head 的维度
        dtype: 数据类型

    Returns:
        shape [total_tokens, num_kv_heads, head_dim] 的 numpy 数组
    """
    all_values = []

    print(f"正在加载 {total_tokens} 个 token 的 value 数据...")
    print(
        f"配置: num_kv_heads={num_kv_heads}, head_dim={head_dim}, dtype={dtype}")
    print("-" * 60)

    for token_idx in range(total_tokens):
        filename = os.path.join(dump_dir, f"v_key_token_idx_[{token_idx}].bin")

        try:
            value_data = load_single_key_token(filename, num_kv_heads, head_dim,
                                               dtype)
            all_values.append(value_data)

            if token_idx % 10 == 0 or token_idx == total_tokens - 1:
                print(
                    f"  已加载 Token {token_idx:3d}: shape={value_data.shape}, "
                    f"min={value_data.min():.4f}, max={value_data.max():.4f}, "
                    f"mean={value_data.mean():.4f}")

        except Exception as e:
            print(f"  ⚠️  加载 Token {token_idx} 失败: {e}")
            raise

    # 沿着 token 维度 concat
    concatenated_values = np.stack(all_values, axis=0)

    print("-" * 60)
    print(f"✅ 成功合并所有 values!")
    print(f"   最终 shape: {concatenated_values.shape}")
    print(f"   总大小: {concatenated_values.nbytes / 1024:.2f} KB "
          f"({concatenated_values.nbytes / (1024*1024):.2f} MB)")
    print(
        f"   数据范围: min={concatenated_values.min():.4f}, "
        f"max={concatenated_values.max():.4f}, mean={concatenated_values.mean():.4f}"
    )

    return concatenated_values


def save_concatenated_keys(keys, output_path):
    """
    保存合并后的 keys

    Args:
        keys: 合并后的 numpy 数组
        output_path: 输出文件路径
    """
    # 保存为 binary 文件
    keys.tofile(output_path)
    print(f"\n💾 已保存为二进制文件: {output_path}")

    # 同时保存为 .npy 格式（方便后续加载）
    npy_path = output_path.replace('.bin', '.npy')
    np.save(npy_path, keys)
    print(f"💾 已保存为 .npy 格式: {npy_path}")

    # 保存统计信息
    stats_path = output_path.replace('.bin', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Shape: {keys.shape}\n")
        f.write(f"Dtype: {keys.dtype}\n")
        f.write(f"Total elements: {keys.size}\n")
        f.write(f"Size (bytes): {keys.nbytes}\n")
        f.write(f"Min: {keys.min()}\n")
        f.write(f"Max: {keys.max()}\n")
        f.write(f"Mean: {keys.mean()}\n")
        f.write(f"Std: {keys.std()}\n")
        f.write(f"\nPer-token statistics:\n")
        for i in range(min(10, keys.shape[0])):
            f.write(f"  Token {i}: min={keys[i].min():.4f}, "
                    f"max={keys[i].max():.4f}, mean={keys[i].mean():.4f}\n")
        if keys.shape[0] > 10:
            f.write(f"  ...\n")
            f.write(f"  Token {keys.shape[0]-1}: min={keys[-1].min():.4f}, "
                    f"max={keys[-1].max():.4f}, mean={keys[-1].mean():.4f}\n")

    print(f"📊 已保存统计信息: {stats_path}")


def verify_concatenated_keys(keys, dump_dir, num_samples=3, is_value=False):
    """
    验证合并后的 keys/values 是否正确

    Args:
        keys: 合并后的 numpy 数组 [total_tokens, num_kv_heads, head_dim]
        dump_dir: 原始文件目录
        num_samples: 验证的样本数量
        is_value: 是否是 value 数据
    """
    data_type = "values" if is_value else "keys"
    print(f"\n🔍 验证合并结果 (抽查 {num_samples} 个 token {data_type})...")
    print("-" * 60)

    total_tokens = keys.shape[0]
    num_kv_heads = keys.shape[1]
    head_dim = keys.shape[2]

    # 随机选择几个 token 进行验证
    sample_indices = np.linspace(0, total_tokens - 1, num_samples, dtype=int)

    all_match = True
    for token_idx in sample_indices:
        if is_value:
            filename = os.path.join(dump_dir,
                                    f"v_key_token_idx_[{token_idx}].bin")
        else:
            filename = os.path.join(dump_dir,
                                    f"key_token_idx_[{token_idx}].bin")

        original = load_single_key_token(filename, num_kv_heads, head_dim,
                                         keys.dtype)
        concatenated = keys[token_idx]

        if np.allclose(original, concatenated, rtol=1e-5, atol=1e-8):
            print(f"  ✅ Token {token_idx}: 完全匹配")
        else:
            print(f"  ❌ Token {token_idx}: 不匹配!")
            max_diff = np.abs(original - concatenated).max()
            print(f"     最大差异: {max_diff}")
            all_match = False

    print("-" * 60)
    if all_match:
        print(f"✅ 验证通过：所有抽查的 {data_type} token 都匹配!")
    else:
        print(f"❌ 验证失败：存在不匹配的 {data_type} token!")

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description='将多个单独的 key/value token 文件合并成完整的 KV cache 文件')
    parser.add_argument('--dump_dir',
                        type=str,
                        default='./dump_data',
                        help='dump 文件所在目录 (默认: ./dump_data)')
    parser.add_argument('--total_tokens',
                        type=int,
                        default=61,
                        help='token 总数 (默认: 61)')
    parser.add_argument('--num_kv_heads',
                        type=int,
                        default=8,
                        help='KV head 数量 (默认: 8)')
    parser.add_argument('--head_dim',
                        type=int,
                        default=128,
                        help='每个 head 的维度 (默认: 128)')
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'bfloat16', 'float32', 'int8', 'uint8'],
        help='数据类型 (默认: float16)')
    parser.add_argument(
        '--output',
        type=str,
        default='./dump_data/all_keys_concatenated.bin',
        help='输出文件路径 (默认: ./dump_data/all_keys_concatenated.bin)')
    parser.add_argument('--type',
                        type=str,
                        default='both',
                        choices=['keys', 'values', 'both'],
                        help='选择合并 keys、values 还是两者都合并 (默认: both)')
    parser.add_argument('--no_verify', action='store_true', help='跳过验证步骤')

    args = parser.parse_args()

    # 数据类型映射
    dtype_map = {
        'float16': np.float16,
        'bfloat16': np.uint16,  # bfloat16 需要特殊处理，这里先用 uint16
        'float32': np.float32,
        'int8': np.int8,
        'uint8': np.uint8,
    }

    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("KV Cache 合并工具")
    print("=" * 60)

    # 检查目录是否存在
    if not os.path.exists(args.dump_dir):
        print(f"❌ 错误: 目录不存在: {args.dump_dir}")
        return

    try:
        # 根据 type 参数决定处理哪些数据
        if args.type in ['keys', 'both']:
            print("\n" + "=" * 60)
            print("处理 Keys")
            print("=" * 60)

            # 合并所有 keys
            concatenated_keys = concat_all_keys(args.dump_dir,
                                                args.total_tokens,
                                                args.num_kv_heads,
                                                args.head_dim, dtype)

            # 确定输出路径
            if args.type == 'both':
                keys_output = args.output.replace('.bin', '_keys.bin')
            else:
                keys_output = args.output

            # 保存结果
            save_concatenated_keys(concatenated_keys, keys_output)

            # 验证（如果需要）
            if not args.no_verify:
                verify_concatenated_keys(concatenated_keys,
                                         args.dump_dir,
                                         num_samples=5,
                                         is_value=False)

        if args.type in ['values', 'both']:
            print("\n" + "=" * 60)
            print("处理 Values")
            print("=" * 60)

            # 合并所有 values
            concatenated_values = concat_all_values(args.dump_dir,
                                                    args.total_tokens,
                                                    args.num_kv_heads,
                                                    args.head_dim, dtype)

            # 确定输出路径
            if args.type == 'both':
                values_output = args.output.replace('.bin', '_values.bin')
            else:
                values_output = args.output

            # 保存结果
            save_concatenated_keys(concatenated_values, values_output)

            # 验证（如果需要）
            if not args.no_verify:
                verify_concatenated_keys(concatenated_values,
                                         args.dump_dir,
                                         num_samples=5,
                                         is_value=True)

        print("\n" + "=" * 60)
        print("✅ 所有操作完成!")
        print("=" * 60)

        print("\n📝 后续使用方法:")
        print("```python")
        print("import numpy as np")
        print()

        if args.type in ['keys', 'both']:
            output_npy = keys_output.replace('.bin', '.npy')
            print(f"# 加载 Keys")
            print(f"keys = np.load('{output_npy}')")
            print(
                f"print(keys.shape)  # 应该是 ({args.total_tokens}, {args.num_kv_heads}, {args.head_dim})"
            )
            print()

        if args.type in ['values', 'both']:
            output_npy = values_output.replace('.bin', '.npy')
            print(f"# 加载 Values")
            print(f"values = np.load('{output_npy}')")
            print(
                f"print(values.shape)  # 应该是 ({args.total_tokens}, {args.num_kv_heads}, {args.head_dim})"
            )
            print()

        if args.type == 'both':
            print("# 访问特定 token 的数据")
            print(
                "token_5_key = keys[5]      # shape: (num_kv_heads, head_dim)")
            print(
                "token_5_value = values[5]  # shape: (num_kv_heads, head_dim)")

        print("```")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
