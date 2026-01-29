"""
sample.py
vLLM 采样模块：Rollout 生成 responses。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .dataclass import Batch


def generate_rollout(
    batch: Batch,
    checkpoint: Path,
    cfg: Any,
) -> Batch:
    """使用 vLLM 为单个 batch 生成多响应。

    逻辑：对每个样本采样 cfg.num_responses 个 response，返回总大小
    为 batch_size * num_responses 的 Batch。

    Args:
        batch: 待生成的 Batch（batch_size 个样本）
        checkpoint: 模型 checkpoint 路径
        cfg: 配置（num_responses, max_new_tokens, temperature 等）

    Returns:
        Batch: 填充了 response_ids 的 Batch（展开后）
    """
    # TODO: 实现 vLLM 生成
    # from vllm import LLM, SamplingParams
    # 
    # llm = LLM(model=str(checkpoint), trust_remote_code=True)
    # params = SamplingParams(
    #     temperature=cfg.temperature,
    #     max_tokens=cfg.max_new_tokens,
    #     detokenize=False,
    # )
    # 
    # batch.prompts = build_prompts(batch)
    # outputs = llm.generate(batch.prompts, params)
    # batch.update_from_vllm(outputs)
    # 
    # del llm
    # torch.cuda.empty_cache()
    
    return batch
