from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from trl import GKDTrainer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .memory import CachedMemory


class ContextDistillationTrainer(GKDTrainer):
    """On-Policy Context-Aware Self-Distillation Trainer.

    核心流程:
    1. Student 根据 Question 生成 Response (on-policy)
    2. Teacher 看到 [Memory, Question]，对生成的 Response 计算 logits
    3. 对齐 Response 部分的分布，最小化 Reverse KL
    """

    def __init__(
        self,
        memory: CachedMemory,
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        *args,
        **kwargs,
    ):
        super().__init__(teacher_model=None, *args, **kwargs)
        self.memory = memory
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        # Memory 缓存
        self._cached_memory_ids: torch.Tensor | None = None
        self._cached_memory_version: int = -1

    # ==================== Memory 管理 ====================

    def _get_memory_ids(self, device: torch.device) -> torch.Tensor:
        """获取 memory token ids，仅在版本变化时重新编码。"""
        if self.memory.version != self._cached_memory_version:
            text = self.memory.prompt_text
            self._cached_memory_ids = (
                self.tokenizer(
                    text, add_special_tokens=False, return_tensors="pt"
                ).input_ids[0]
                if text
                else torch.empty(0, dtype=torch.long)
            )
            self._cached_memory_version = self.memory.version
        return self._cached_memory_ids.to(device)

    # ==================== 核心训练逻辑 ====================

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """On-Policy 蒸馏损失计算。

        1. 提取 prompt (Question)
        2. Student 生成 Response
        3. Student/Teacher 分别前向传播
        4. 对齐 Response 部分，计算 Top-K Reverse KL
        """
        device = inputs["input_ids"].device

        # Step 1: 提取 prompt
        prompt_ids, prompt_mask, prompt_len = self._extract_prompt(inputs)

        # Step 2: Student 生成 response (on-policy)
        generated_ids, generated_mask = self._generate_response(
            model, prompt_ids, prompt_mask
        )
        response_len = generated_ids.shape[1] - prompt_len

        # Step 3: Student forward (with grad)
        student_outputs = model(
            input_ids=generated_ids, attention_mask=generated_mask, use_cache=False
        )
        # 只取 response 部分的 logits (预测 response tokens)
        student_logits = student_outputs.logits[
            :, prompt_len - 1 : -1, :
        ]  # [B, response_len, V]

        # Step 4: Teacher forward (with memory, no grad)
        teacher_logits = self._teacher_forward(
            model, generated_ids, generated_mask, prompt_len, device
        )

        # Step 5: Compute loss
        loss = self._compute_topk_reverse_kl(student_logits, teacher_logits)

        return (loss, student_outputs) if return_outputs else loss

    # ==================== 辅助方法 ====================

    def _extract_prompt(self, inputs) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """从 inputs 提取 prompt（labels == -100 的部分）。"""
        labels = inputs["labels"]
        # 找到第一个 label != -100 的位置（response 起始）
        prompt_len = (labels[0] == -100).sum().item()

        prompt_ids = inputs["input_ids"][:, :prompt_len]
        prompt_mask = (
            inputs["attention_mask"][:, :prompt_len]
            if "attention_mask" in inputs
            else None
        )

        return prompt_ids, prompt_mask, prompt_len

    def _generate_response(
        self, model, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Student 生成 response (on-policy sampling)。"""
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        # 构建 attention mask
        generated_mask = torch.ones_like(generated_ids)
        return generated_ids, generated_mask

    def _teacher_forward(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Teacher 前向传播：[Memory, Question, Response]。"""
        with torch.no_grad():
            memory_ids = self._get_memory_ids(device)
            mem_len = memory_ids.shape[0]

            if mem_len > 0:
                batch_size = input_ids.shape[0]
                memory_batch = memory_ids.unsqueeze(0).expand(batch_size, -1)
                teacher_input_ids = torch.cat([memory_batch, input_ids], dim=1)

                memory_mask = torch.ones(
                    batch_size, mem_len, device=device, dtype=attention_mask.dtype
                )
                teacher_attention_mask = torch.cat([memory_mask, attention_mask], dim=1)

                teacher_outputs = model(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    use_cache=False,
                )
                # 对齐：截取 response 部分的 logits
                # Teacher 的 response 从 mem_len + prompt_len - 1 开始预测
                start_idx = mem_len + prompt_len - 1
                teacher_logits = teacher_outputs.logits[:, start_idx:-1, :]
            else:
                # 无 memory 时退化
                teacher_outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
                teacher_logits = teacher_outputs.logits[:, prompt_len - 1 : -1, :]

        return teacher_logits

    def _compute_topk_reverse_kl(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """计算 Top-K Reverse KL: KL(student || teacher)。

        只在 Student 的 Top-K tokens 上计算，提高效率。
        """
        # Student Top-K
        topk_values, topk_indices = student_logits.topk(self.top_k, dim=-1)

        # 重新归一化 (在 Top-K 上)
        student_topk_log_probs = F.log_softmax(topk_values, dim=-1)
        student_topk_probs = student_topk_log_probs.exp()

        # 从 Teacher 中 gather 对应位置
        teacher_topk_logits = teacher_logits.gather(-1, topk_indices)
        teacher_topk_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)

        # KL(P||Q) = sum P * (log P - log Q)
        kl_per_token = (
            student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)
        ).sum(dim=-1)

        return kl_per_token.mean()
