#!/usr/bin/env python3
"""
读取和验证 KV cache dump 数据的工具脚本
"""
from pathlib import Path

import numpy as np


def load_kv_token(token_idx,
                  dump_dir="./dump_data",
                  num_heads=8,
                  head_dim=128,
                  dtype=np.float16):
    """
    加载指定 token 的 K 和 V cache 数据

    Args:
        token_idx: token 索引
        dump_dir: dump 数据目录
        num_heads: KV head 数量
        head_dim: 每个 head 的维度
        dtype: 数据类型 (np.float16, np.int8, np.uint8 等)

    Returns:
        (k_data, v_data): 形状为 [num_heads, head_dim] 的 numpy 数组
    """
    k_path = Path(dump_dir) / f"key_token_idx_[{token_idx}].bin"
    v_path = Path(dump_dir) / f"v_key_token_idx_[{token_idx}].bin"

    if not k_path.exists():
        raise FileNotFoundError(f"K cache file not found: {k_path}")
    if not v_path.exists():
        raise FileNotFoundError(f"V cache file not found: {v_path}")

    # 读取数据
    k_data = np.fromfile(k_path, dtype=dtype)
    v_data = np.fromfile(v_path, dtype=dtype)

    # Reshape 为 [num_heads, head_dim]
    k_data = k_data.reshape(num_heads, head_dim)
    v_data = v_data.reshape(num_heads, head_dim)

    return k_data, v_data


def load_all_kv_tokens(total_tokens=61,
                       dump_dir="./dump_data",
                       num_heads=8,
                       head_dim=128,
                       dtype=np.float16):
    """
    加载所有 token 的 KV cache 数据

    Returns:
        (all_k, all_v): 形状为 [total_tokens, num_heads, head_dim] 的 numpy 数组
    """
    all_k = []
    all_v = []

    for token_idx in range(total_tokens):
        try:
            k_data, v_data = load_kv_token(token_idx, dump_dir, num_heads,
                                           head_dim, dtype)
            all_k.append(k_data)
            all_v.append(v_data)
            print(
                f"✓ Loaded token {token_idx:3d}: K shape={k_data.shape}, V shape={v_data.shape}"
            )
        except FileNotFoundError as e:
            print(f"✗ Token {token_idx}: {e}")
            break

    if not all_k:
        return None, None

    all_k = np.stack(all_k, axis=0)  # [total_tokens, num_heads, head_dim]
    all_v = np.stack(all_v, axis=0)

    return all_k, all_v


def load_q_buffer(dump_dir="./dump_data",
                  num_tokens=25,
                  num_heads=32,
                  head_dim=128,
                  dtype=np.float16):
    """加载 Q buffer"""
    q_path = Path(dump_dir) / "q_buffer.bin"
    if not q_path.exists():
        raise FileNotFoundError(f"Q buffer file not found: {q_path}")

    q_data = np.fromfile(q_path, dtype=dtype)
    q_data = q_data.reshape(num_tokens, num_heads, head_dim)
    return q_data


def load_custom_mask(dump_dir="./dump_data", mask_shape=(2, 128, 128)):
    """加载 custom mask"""
    mask_path = Path(dump_dir) / "custom_mask.bin"
    if not mask_path.exists():
        raise FileNotFoundError(f"Custom mask file not found: {mask_path}")

    mask_data = np.fromfile(mask_path, dtype=np.int32)
    mask_data = mask_data.reshape(mask_shape)
    return mask_data


def verify_kv_cache(dump_dir="./dump_data",
                    total_tokens=61,
                    num_heads=8,
                    head_dim=128):
    """验证 KV cache dump 数据"""
    print("=" * 80)
    print("验证 KV Cache Dump 数据")
    print("=" * 80)

    # 检查目录是否存在
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        print(f"❌ Dump 目录不存在: {dump_dir}")
        return False

    print(f"📁 Dump 目录: {dump_dir}")
    print(f"🎯 预期 token 数量: {total_tokens}")
    print(f"🎯 预期 KV head 数量: {num_heads}")
    print(f"🎯 预期每个 head 维度: {head_dim}")
    print()

    # 加载所有 KV cache
    print("正在加载 KV cache 数据...")
    all_k, all_v = load_all_kv_tokens(total_tokens,
                                      dump_dir,
                                      num_heads,
                                      head_dim,
                                      dtype=np.float16)

    if all_k is None:
        print("❌ 未找到任何 KV cache 数据")
        return False

    print()
    print(f"✅ 成功加载 {len(all_k)} 个 token 的 KV cache")
    print(f"   K cache shape: {all_k.shape}")
    print(f"   V cache shape: {all_v.shape}")

    # 统计信息
    print()
    print("📊 K Cache 统计信息:")
    print(f"   Min: {all_k.min():.6f}")
    print(f"   Max: {all_k.max():.6f}")
    print(f"   Mean: {all_k.mean():.6f}")
    print(f"   Std: {all_k.std():.6f}")

    print()
    print("📊 V Cache 统计信息:")
    print(f"   Min: {all_v.min():.6f}")
    print(f"   Max: {all_v.max():.6f}")
    print(f"   Mean: {all_v.mean():.6f}")
    print(f"   Std: {all_v.std():.6f}")

    # 尝试加载 Q buffer
    print()
    try:
        q_data = load_q_buffer(dump_dir)
        print(f"✅ Q buffer shape: {q_data.shape}")
        print(
            f"   Min: {q_data.min():.6f}, Max: {q_data.max():.6f}, Mean: {q_data.mean():.6f}"
        )
    except FileNotFoundError as e:
        print(f"⚠️  Q buffer: {e}")

    # 尝试加载 custom mask
    print()
    try:
        mask_data = load_custom_mask(dump_dir)
        print(f"✅ Custom mask shape: {mask_data.shape}")
        print(f"   Min: {mask_data.min()}, Max: {mask_data.max()}")
    except FileNotFoundError as e:
        print(f"⚠️  Custom mask: {e}")

    print()
    print("=" * 80)
    print("✅ 验证完成")
    print("=" * 80)

    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="读取和验证 KV cache dump 数据")
    parser.add_argument("--dump_dir",
                        type=str,
                        default="./dump_data",
                        help="Dump 数据目录")
    parser.add_argument("--total_tokens",
                        type=int,
                        default=61,
                        help="总 token 数量")
    parser.add_argument("--num_heads", type=int, default=8, help="KV head 数量")
    parser.add_argument("--head_dim", type=int, default=128, help="每个 head 的维度")
    parser.add_argument("--token_idx",
                        type=int,
                        default=None,
                        help="只加载指定 token (可选)")

    args = parser.parse_args()

    if args.token_idx is not None:
        # 只加载指定 token
        print(f"加载 token {args.token_idx}...")
        k_data, v_data = load_kv_token(args.token_idx, args.dump_dir,
                                       args.num_heads, args.head_dim)
        print(f"K cache shape: {k_data.shape}")
        print(f"V cache shape: {v_data.shape}")
        print(f"\nK cache (head 0, first 8 elements):\n{k_data[0, :8]}")
        print(f"\nV cache (head 0, first 8 elements):\n{v_data[0, :8]}")
    else:
        # 验证所有数据
        verify_kv_cache(args.dump_dir, args.total_tokens, args.num_heads,
                        args.head_dim)


if __name__ == "__main__":
    main()
