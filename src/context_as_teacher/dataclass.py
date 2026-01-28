"""数据类定义。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Batch:
    """训练批次数据载体，贯穿整个流程。

    生命周期:
    1. DataLoader → 初始化 (questions, answers, solutions)
    2. prompt.build_prompts → 填充 prompts
    3. tokenizer(prompts) → 填充 prompt_ids
    4. vLLM.generate(prompt_ids) → 填充 response_ids (直接返回 ids，无 detokenize)
    5. concat(prompt_ids, response_ids) → 填充 input_ids
    6. trainer.compute_loss → 消费
    """

    # ===== 原始数据 (from DataLoader) =====
    questions: list[str]
    answers: list[str]  # ground truth final answer
    solutions: list[str] = field(default_factory=list)  # ground truth reasoning

    # ===== Prompt 构建阶段 =====
    prompts: list[str] | None = None  # 构建的完整 prompt text

    # ===== 生成阶段 (全部为 token ids) =====
    prompt_ids: torch.Tensor | None = None  # [batch, prompt_len]
    response_ids: torch.Tensor | None = None  # [batch, response_len] vLLM 直接返回

    # ===== 训练阶段 =====
    input_ids: torch.Tensor | None = (
        None  # [batch, seq_len] = concat(prompt_ids, response_ids)
    )
    attention_mask: torch.Tensor | None = None  # [batch, seq_len]

    @property
    def batch_size(self) -> int:
        return len(self.questions)

    @property
    def prompt_len(self) -> int | None:
        return self.prompt_ids.shape[1] if self.prompt_ids is not None else None

    def is_ready_for_generate(self) -> bool:
        """检查是否可以进行 vLLM 生成。"""
        return self.prompt_ids is not None

    def is_ready_for_train(self) -> bool:
        """检查是否可以进行训练。"""
        return self.input_ids is not None and self.prompt_len is not None
