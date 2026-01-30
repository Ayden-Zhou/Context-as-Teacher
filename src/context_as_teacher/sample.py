"""
sample.py
vLLM 采样模块：Rollout 生成 responses。
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def generate_rollout(
    prompt_ids: list[list[int]],
    checkpoint: Path,
    max_new_tokens: int,
    temperature: float,
    responses_per_prompt: int,
) -> list[list[int]]:
    """使用 vLLM 为单个 batch 生成多响应。

    逻辑：对每个样本采样 responses_per_prompt 个 response，返回总大小
    为 batch_size * num_responses 的 Batch。

    Args:
        prompt_ids: prompt 的 token ids
        checkpoint: 模型 checkpoint 路径
        max_new_tokens: 生成最大新 token 数
        temperature: 采样温度
        responses_per_prompt: 每题采样的响应数

    Returns:
        list[list[int]]: 展开后的 response_ids（长度为 B * R）
    """
    llm = LLM(
        model=str(checkpoint),
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=responses_per_prompt,
        detokenize=False,
    )

    outputs = llm.generate(prompt_token_ids=prompt_ids, sampling_params=params)

    response_ids = [cand.token_ids for output in outputs for cand in output.outputs]

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return response_ids
