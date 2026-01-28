# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from tensorrt_llm import LLM
from tensorrt_llm.llmapi import CudaGraphConfig, EagleDecodingConfig, KvCacheConfig

from ..conftest import llm_models_root, parametrize_with_ids
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness

"""
class TestLlama3_3_70BInstruct(LlmapiAccuracyTestHarness):

    @skip_pre_hopper
    @pytest.mark.skip_less_mpi_world_size(8)
    @parametrize_with_ids("eagle3_one_model", [True, False])
    def test_fp8_eagle3_tp8(self, eagle3_one_model):
        MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp8"
        # model_path = f"{llm_models_root()}/llama-3.3-models/Llama-3.3-70B-Instruct"
        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.3-Instruct-70B"
        kv_cache_config = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.6)
        spec_config = EagleDecodingConfig(max_draft_len=3,
                                          speculative_model_dir=eagle_model_dir,
                                          eagle3_one_model=eagle3_one_model)
        pytorch_config = dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig(max_batch_size=1))
        with LLM(model_path,
                 max_batch_size=16,
                 tensor_parallel_size=8,
                 speculative_config=spec_config,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config) as llm:
            task = GSM8K(MODEL_NAME)
            task.evaluate(llm)


    @skip_pre_hopper
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_fp8_tp8(self):
        MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp8"
        kv_cache_config = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.6)
        pytorch_config = dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig(max_batch_size=1))
        with LLM(model_path,
                 max_batch_size=16,
                 tensor_parallel_size=8,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config) as llm:
            task = GSM8K(MODEL_NAME)
            task.evaluate(llm)
"""


class TestLlama3_1_8BInstruct_eagle3(LlmapiAccuracyTestHarness):
    @parametrize_with_ids("eagle3_one_model", [False])
    def test_llama3_1_8BInstruct_eagle3(self, eagle3_one_model):
        MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
        model_path = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
        kv_cache_config = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3)
        spec_config = EagleDecodingConfig(
            max_draft_len=4,
            speculative_model_dir=eagle_model_dir,
            eagle3_one_model=eagle3_one_model,
        )
        pytorch_config = dict(
            disable_overlap_scheduler=True, cuda_graph_config=CudaGraphConfig(max_batch_size=8)
        )
        with LLM(
            model_path,
            max_batch_size=8,
            tensor_parallel_size=1,
            speculative_config=spec_config,
            kv_cache_config=kv_cache_config,
            **pytorch_config,
        ) as llm:
            task = GSM8K(MODEL_NAME)
            task.evaluate(llm)

    def test_llama3_1_8BInstruct(self):
        MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
        model_path = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
        kv_cache_config = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3)
        pytorch_config = dict(
            disable_overlap_scheduler=True, cuda_graph_config=CudaGraphConfig(max_batch_size=8)
        )
        with LLM(
            model_path,
            max_batch_size=8,
            tensor_parallel_size=1,
            kv_cache_config=kv_cache_config,
            **pytorch_config,
        ) as llm:
            task = GSM8K(MODEL_NAME)
            task.evaluate(llm)

    def test_llama3_1_8BInstruct_eagle3_static_tree(self):
        MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
        model_path = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
        kv_cache_config = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3)
        spec_config = EagleDecodingConfig(
            max_draft_len=4,
            speculative_model_dir=eagle_model_dir,
            eagle3_one_model=False,
            use_dynamic_tree=False,
            eagle_choices=[
                [0],
                [0, 0],
                [1],
                [0, 1],
                [2],
                [0, 0, 0],
                [1, 0],
                [0, 2],
                [3],
                [0, 3],
                [4],
                [0, 4],
                [2, 0],
                [0, 5],
                [0, 0, 1],
                [5],
                [0, 6],
                [6],
                [0, 7],
                [0, 1, 0],
                [1, 1],
                [7],
                [0, 8],
                [0, 0, 2],
                [3, 0],
                [0, 9],
                [8],
                [9],
                [1, 0, 0],
                [0, 2, 0],
                [1, 2],
                [0, 0, 3],
                [4, 0],
                [2, 1],
                [0, 0, 4],
                [0, 0, 5],
                [0, 0, 0, 0],
                [0, 1, 1],
                [0, 0, 6],
                [0, 3, 0],
                [5, 0],
                [1, 3],
                [0, 0, 7],
                [0, 0, 8],
                [0, 0, 9],
                [6, 0],
                [0, 4, 0],
                [1, 4],
                [7, 0],
                [0, 1, 2],
                [2, 0, 0],
                [3, 1],
                [2, 2],
                [8, 0],
                [0, 5, 0],
                [1, 5],
                [1, 0, 1],
                [0, 2, 1],
                [9, 0],
                [0, 6, 0],
                [0, 0, 0, 1],
                [1, 6],
                [0, 7, 0],
            ],
        )
        pytorch_config = dict(disable_overlap_scheduler=True, cuda_graph_config=None)
        with LLM(
            model_path,
            max_batch_size=8,
            tensor_parallel_size=1,
            speculative_config=spec_config,
            kv_cache_config=kv_cache_config,
            **pytorch_config,
        ) as llm:
            task = GSM8K(MODEL_NAME)
            task.evaluate(llm)

    def test_llama3_1_8BInstruct_eagle3_dynmaic_tree(self):
        MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
        model_path = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
        kv_cache_config = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3)
        spec_config = EagleDecodingConfig(
            max_draft_len=4,
            speculative_model_dir=eagle_model_dir,
            eagle3_one_model=False,
            use_dynamic_tree=True,
            dynamic_tree_max_topK=10,
            max_total_draft_tokens=63,
        )
        pytorch_config = dict(disable_overlap_scheduler=True, cuda_graph_config=None)
        with LLM(
            model_path,
            max_batch_size=8,
            tensor_parallel_size=1,
            speculative_config=spec_config,
            kv_cache_config=kv_cache_config,
            **pytorch_config,
        ) as llm:
            task = GSM8K(MODEL_NAME)
            task.evaluate(llm)
