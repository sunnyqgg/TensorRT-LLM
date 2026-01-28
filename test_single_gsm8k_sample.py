#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试单个GSM8K样本，用于复现和调试illegal memory错误."""

import os
from pathlib import Path

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import EagleDecodingConfig, KvCacheConfig, SamplingParams


def llm_models_root() -> str:
    """返回LLM_MODELS_ROOT路径."""
    root = Path("/home/scratch.trt_llm_data/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ.get("LLM_MODELS_ROOT"))

    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")

    assert root.exists(), (
        "You shall set LLM_MODELS_ROOT env or be able to access scratch.trt_llm_data to run this test"
    )

    return str(root)


def get_gsm8k_prompt():
    """返回标准的GSM8K few-shot prompt."""
    return """Question: Three years ago, Rosie purchased an art piece for $4000. \
The same art piece will be three times as much in another three years. \
How much money will the art piece have increased by?
Answer: If the art piece was bought at $4000, and its price in three years \
will be three times its buying price, it will be worth 3*$4000 = $12000 in another three years.
The price of the art piece will increase by $12000-$4000 = $<<12000-4000=8000>>8000
#### 8000

Question: If Clover goes for a 1.5-mile walk in the morning and another \
1.5-mile walk in the evening, every day, how many miles does he walk in 30 days?
Answer: He walks 1.5 miles in the morning and 1.5 miles in the evening so that's 1.5+1.5 = <<1.5+1.5=3>>3 miles
If he walks 3 miles everyday, for 30 days then he walks 3*30 = <<3*30=90>>90 miles in 30 days
#### 90

Question: Caitlin makes bracelets to sell at the farmer's market every weekend. \
Each bracelet takes twice as many small beads as it does large beads. \
If each bracelet uses 12 large beads, and Caitlin has 528 beads with equal amounts \
of large and small beads, how many bracelets can she make for this weekend?
Answer: Caitlin has 528/2 = <<528/2=264>>264 small beads.
Each bracelet needs 12 * 2 = <<12*2=24>>24 small beads.
Thus, Caitlin can make 264/24 = <<264/24=11>>11 bracelets for the farmer's market this weekend.
#### 11

Question: Santana has 7 brothers. 3 of them have birthdays in March, 1 of them has \
a birthday in October, 1 has a birthday in November, and another 2 of them were born \
in December. If Santana always buys each of her brothers a birthday present and a \
Christmas present, how many more presents does she have to buy in the second half \
of the year than the first half of the year?
Answer: Santana has 1 + 1 + 2 = <<1+1+2=4>>4 brothers with birthdays in the second half of the year.
She has 7 brothers - 4 brothers = <<7-4=3>>3 brothers with birthdays in the first half of the year.
Altogether, she has to buy 4 + 7 = <<4+7=11>>11 presents in the second half of the year.
Therefore, she has to buy 11 - 3 = <<11-3=8>>8 more presents in the second half of the year.
#### 8

Question: A class composed of 12 girls and 10 boys was sent to the library for \
their reading class. Their teacher found out that only 5/6 of the girls and 4/5 \
of the boys are reading. How many students are not reading?
Answer: There are 12 x 5/6 = <<12*5/6=10>>10 girls who are reading.
So, 12 - 10 = <<12-10=2>>2 girls are not reading.
There are 10 x 4/5 = <<10*4/5=8>>8 boys who are reading.
So, 10 - 8 = <<10-8=2>>2 boys are not reading.
Thus, there are a total of 2 girls + 2 boys = <<2+2=4>>4 students who are not reading.
#### 4

Question: Nissa hires 60 seasonal workers to play elves in her department store's \
Santa village. A third of the elves quit after children vomit on them, then 10 of \
the remaining elves quit after kids kick their shins. How many elves are left?
Answer:"""


def test_single_sample():
    """测试单个GSM8K样本."""
    # 模型路径配置（使用和测试文件相同的路径）
    model_path = f"{llm_models_root()}/Qwen3/Qwen3-8B"
    eagle_model_dir = f"{llm_models_root()}/Qwen3/qwen3_8b_eagle3"

    # KV Cache配置
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.4)

    # EAGLE配置
    spec_config = EagleDecodingConfig(
        max_draft_len=3,
        speculative_model_dir=eagle_model_dir,
        eagle3_one_model=False,
        eagle_choices=[
            [0],
            [1],
            [2],
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [2, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
        ],
    )

    # PyTorch配置
    pytorch_config = dict(disable_overlap_scheduler=True, cuda_graph_config=None)

    # Sampling参数
    sampling_params = SamplingParams(
        max_tokens=256,
        truncate_prompt_tokens=4096,
        temperature=0.0,  # 使用贪婪采样以保证可重复性
    )

    print("=" * 80)
    print("初始化LLM模型...")
    print("=" * 80)

    # 初始化LLM
    with LLM(
        model_path,
        max_batch_size=1,
        tensor_parallel_size=1,
        speculative_config=spec_config,
        kv_cache_config=kv_cache_config,
        **pytorch_config,
    ) as llm:
        # 获取prompt
        prompt = get_gsm8k_prompt()

        print("\n" + "=" * 80)
        print("测试Prompt:")
        print("=" * 80)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("=" * 80)
        print(f"Prompt长度: {len(prompt)} 字符")
        print("=" * 80)

        try:
            print("\n开始生成...")

            # 生成
            outputs = llm.generate(prompt, sampling_params=sampling_params)

            print("\n" + "=" * 80)
            print("生成成功！")
            print("=" * 80)

            # 打印结果
            for output in outputs:
                print(f"\n生成的回答:\n{output.outputs[0].text}")

        except Exception as e:
            print("\n" + "=" * 80)
            print("❌ 生成失败！遇到错误:")
            print("=" * 80)
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_single_sample()
