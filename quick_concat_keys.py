#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速合并 61 个 key token 文件 - 简化版
"""

import os

import numpy as np


def concat_keys(dump_dir="./dump_data",
                total_tokens=61,
                num_kv_heads=32,
                head_dim=128):
    """
    合并所有 key token 文件

    Returns:
        shape [total_tokens, num_kv_heads, head_dim] 的 numpy 数组
    """
    print(f"正在合并 {total_tokens} 个 key token 文件...")
    print(f"配置: num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    all_keys = []

    for token_idx in range(total_tokens):
        filename = os.path.join(dump_dir, f"key_token_idx_[{token_idx}].bin")

        # 读取单个 token 的 key
        key_data = np.fromfile(filename, dtype=np.float16)
        key_data = key_data.reshape(num_kv_heads, head_dim)

        all_keys.append(key_data)

        if token_idx % 10 == 0:
            print(f"  已加载 Token {token_idx}/{total_tokens}")

    # 合并
    concatenated = np.stack(all_keys, axis=0)

    print(f"\n✅ 合并完成!")
    print(f"   Shape: {concatenated.shape}")
    print(f"   Dtype: {concatenated.dtype}")
    print(f"   大小: {concatenated.nbytes / 1024:.2f} KB")
    print(f"   范围: [{concatenated.min():.4f}, {concatenated.max():.4f}]")

    return concatenated


def concat_values(dump_dir="./dump_data",
                  total_tokens=61,
                  num_kv_heads=32,
                  head_dim=128):
    """
    合并所有 value token 文件

    Returns:
        shape [total_tokens, num_kv_heads, head_dim] 的 numpy 数组
    """
    print(f"\n正在合并 {total_tokens} 个 value token 文件...")
    print(f"配置: num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    all_values = []

    for token_idx in range(total_tokens):
        filename = os.path.join(dump_dir, f"v_key_token_idx_[{token_idx}].bin")

        # 读取单个 token 的 value
        value_data = np.fromfile(filename, dtype=np.float16)
        value_data = value_data.reshape(num_kv_heads, head_dim)

        all_values.append(value_data)

        if token_idx % 10 == 0:
            print(f"  已加载 Token {token_idx}/{total_tokens}")

    # 合并
    concatenated = np.stack(all_values, axis=0)

    print(f"\n✅ 合并完成!")
    print(f"   Shape: {concatenated.shape}")
    print(f"   Dtype: {concatenated.dtype}")
    print(f"   大小: {concatenated.nbytes / 1024:.2f} KB")
    print(f"   范围: [{concatenated.min():.4f}, {concatenated.max():.4f}]")

    return concatenated


if __name__ == "__main__":
    print("=" * 70)
    print("KV Cache 合并工具（快速版）")
    print("=" * 70)

    # 合并 Keys
    keys = concat_keys(
        dump_dir="./dump_data",
        total_tokens=61,
        num_kv_heads=32,  # 根据你的模型调整
        head_dim=128  # 根据你的模型调整
    )

    # 保存 Keys
    keys.tofile("./dump_data/all_keys.bin")
    np.save("./dump_data/all_keys.npy", keys)
    print(f"\n💾 Keys 已保存:")
    print(f"   - ./dump_data/all_keys.bin")
    print(f"   - ./dump_data/all_keys.npy")

    # 合并 Values
    values = concat_values(dump_dir="./dump_data",
                           total_tokens=61,
                           num_kv_heads=32,
                           head_dim=128)

    # 保存 Values
    values.tofile("./dump_data/all_values.bin")
    np.save("./dump_data/all_values.npy", values)
    print(f"\n💾 Values 已保存:")
    print(f"   - ./dump_data/all_values.bin")
    print(f"   - ./dump_data/all_values.npy")

    print("\n" + "=" * 70)
    print("✅ 所有操作完成!")
    print("=" * 70)

    print("\n📝 使用示例:")
    print("```python")
    print("import numpy as np")
    print("")
    print("# 加载合并后的 keys")
    print("keys = np.load('./dump_data/all_keys.npy')")
    print(f"print(keys.shape)  # {keys.shape}")
    print("")
    print("# 加载合并后的 values")
    print("values = np.load('./dump_data/all_values.npy')")
    print(f"print(values.shape)  # {values.shape}")
    print("")
    print("# 访问特定 token 的数据")
    print("token_5_keys = keys[5]      # shape: (num_kv_heads, head_dim)")
    print("token_5_values = values[5]  # shape: (num_kv_heads, head_dim)")
    print("")
    print("# 访问特定 token 的特定 head")
    print("token_5_head_0_key = keys[5, 0]  # shape: (head_dim,)")
    print("```")
