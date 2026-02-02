from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ContextDistillationTrainer:
    """On-Policy Context-Aware Self-Distillation Trainer.

    单模型自蒸馏：学生 π_S(x) 和教师 π_T(m⊕x) 共享权重 θ。
    通过 Top-K Reverse KL 蒸馏教师的上下文感知能力到学生。
    """

    def __init__(
        self,
        cfg: Any,
        model_root: Path,
        latest_checkpoint: Path,
        load_checkpoint: Path,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        self.cfg = cfg
        self.model_root = model_root
        self.latest_checkpoint = latest_checkpoint
        self.load_checkpoint = load_checkpoint
        self.tokenizer = tokenizer
        self.model: PreTrainedModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def start_train(self) -> None:
        """加载模型和优化器，配置训练环境。"""
        # TF32 精度：Hopper 架构下提升 matmul 吞吐
        torch.set_float32_matmul_precision("high")

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.load_checkpoint,
                trust_remote_code=True,
            )

        attn_impl = self._resolve_attn_implementation()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.load_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=self.cfg.device,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        # 大模型启用 gradient checkpointing 节省显存
        num_params = sum(p.numel() for p in self.model.parameters())
        if num_params > 4e9:
            self.model.gradient_checkpointing_enable()

        # fused AdamW：融合 kernel 提升性能
        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                fused=True,
            )
        except RuntimeError:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                fused=False,
            )

    def update(
        self,
        global_step: int,
        prompt_ids: torch.Tensor | list[list[int]],
        response_ids: torch.Tensor | list[list[int]],
        teacher_prompt_ids: torch.Tensor | list[list[int]] | None = None,
    ) -> tuple[int, float, float]:
        """执行单步梯度更新，返回 (global_step, loss, grad_norm)。

        Args:
            global_step: 当前全局步数。
            prompt_ids: 学生 prompt token IDs。Shape: [B, prompt_len]
            response_ids: 生成的 response token IDs。Shape: [B, response_len]
            teacher_prompt_ids: 教师 prompt token IDs（含 memory）。Shape: [B, teacher_prompt_len]

        Returns:
            (更新后的 global_step, loss 值, 梯度范数)。
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("训练未启动，请先调用 start_train()")

        self.optimizer.zero_grad()

        loss = self.train_step(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            teacher_prompt_ids=teacher_prompt_ids,
        )

        loss.backward()
        # 计算梯度范数（不 clip，max_norm=inf）
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=float("inf")
        ).item()
        self.optimizer.step()

        global_step += 1
        loss_val = loss.item()
        if global_step % 10 == 0:
            print(
                f"       step {global_step}, loss: {loss_val:.4f}, grad_norm: {grad_norm:.4f}"
            )

        return global_step, loss_val, grad_norm

    def train_step(
        self,
        *,
        prompt_ids: torch.Tensor | list[list[int]],
        response_ids: torch.Tensor | list[list[int]],
        teacher_prompt_ids: torch.Tensor | list[list[int]] | None = None,
    ) -> torch.Tensor:
        """单步蒸馏训练，返回 loss（不含 backward）。

        数据流：
        - 学生: x (问题) → prompt_ids ⊕ response_ids → student_logits [grad=T]
        - 教师: m⊕x (记忆+问题) → teacher_prompt_ids ⊕ response_ids → teacher_logits [grad=F]
        - 损失: D_KL^(K)(π_S || π_T)

        Args:
            prompt_ids: 学生 prompt token IDs。Shape: [B, prompt_len]
            response_ids: 生成的 response token IDs。Shape: [B, response_len]
            teacher_prompt_ids: 教师 prompt token IDs（含 memory）。Shape: [B, teacher_prompt_len]

        Returns:
            Top-K Reverse KL 损失。Shape: scalar
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        pad_id = self.cfg.pad_token_id
        prompt_ids = self._prepare_ids(prompt_ids, pad_id)
        response_ids = self._prepare_ids(response_ids, pad_id)
        if teacher_prompt_ids is not None:
            teacher_prompt_ids = self._prepare_ids(teacher_prompt_ids, pad_id)

        prompt_len = prompt_ids.shape[1]
        response_mask = response_ids.ne(pad_id)  # [B, response_len]

        with autocast("cuda", dtype=torch.bfloat16):
            # Student forward [grad=T]: input = prompt ⊕ response
            student_input_ids = torch.cat(
                [prompt_ids, response_ids], dim=1
            )  # [B, prompt_len + response_len]
            student_outputs = self.model(input_ids=student_input_ids, use_cache=False)
            # 取 response 部分的 logits（预测下一个 token）
            student_logits = student_outputs.logits[
                :, prompt_len - 1 : -1, :
            ]  # [B, response_len, V]

            # Teacher forward [grad=F]: input = teacher_prompt ⊕ response
            if teacher_prompt_ids is None:
                # 无 memory 时，教师 logits 直接用学生 logits（detach）
                teacher_logits = student_logits.detach()  # [B, response_len, V]
            else:
                teacher_input_ids = torch.cat(
                    [teacher_prompt_ids, response_ids], dim=1
                )  # [B, teacher_prompt_len + response_len]
                teacher_prompt_len = teacher_prompt_ids.shape[1]

                with torch.no_grad():
                    teacher_outputs = self.model(
                        input_ids=teacher_input_ids, use_cache=False
                    )
                    teacher_logits = teacher_outputs.logits[
                        :, teacher_prompt_len - 1 : -1, :
                    ]  # [B, response_len, V]

            # Top-K Reverse KL: D_KL^(K)(π_S || π_T)
            loss = self.topk_reverse_kl(
                student_logits,
                teacher_logits,
                k=self.cfg.top_k,
                mask=response_mask,
            )

        return loss

    @staticmethod
    def topk_reverse_kl(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        k: int = 50,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算 Top-K Reverse KL 散度：D_KL^(K)(π_S || π_T)。

        在学生分布的 Top-K token 上计算逆向 KL，使学生逼近教师的 mode。
        Eq: D_KL^(K)(P_S || P_T) = Σ_{v∈Top-K(P_S)} P̃_S(v) · (log P̃_S(v) - log P̃_T(v))

        Args:
            student_logits: 学生模型输出 logits。Shape: [B, Seq, V]
            teacher_logits: 教师模型输出 logits。Shape: [B, Seq, V]
            k: Top-K 值，仅在前 k 个 token 上计算 KL。
            mask: 有效 token mask。Shape: [B, Seq]

        Returns:
            平均 KL 散度。Shape: scalar
        """
        # 取学生分布的 Top-K token
        topk_vals, topk_idx = student_logits.topk(k, dim=-1)  # [B, Seq, K], [B, Seq, K]

        # 在 Top-K 上重新归一化：P̃_S
        student_log_probs = F.log_softmax(topk_vals, dim=-1)  # [B, Seq, K]
        student_probs = student_log_probs.exp()  # [B, Seq, K]

        # 从教师 logits 中 gather 对应位置，重新归一化：P̃_T
        teacher_topk_logits = teacher_logits.gather(-1, topk_idx)  # [B, Seq, K]
        teacher_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)  # [B, Seq, K]

        # KL(P||Q) = Σ P · (log P - log Q)
        kl_per_token = (student_probs * (student_log_probs - teacher_log_probs)).sum(
            -1
        )  # [B, Seq]
        if mask is None:
            return kl_per_token.mean()  # scalar
        mask = mask.to(dtype=kl_per_token.dtype)  # [B, Seq]
        denom = mask.sum().clamp_min(1)
        return (kl_per_token * mask).sum() / denom  # scalar

    @staticmethod
    def _right_pad_ids(
        ids: list[list[int]],
        *,
        pad_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Right-pad ragged IDs to a dense tensor.

        Args:
            ids: Ragged token IDs。Shape: [B, *]
            pad_id: Padding token ID.
            device: 目标设备。

        Returns:
            Right-padded IDs。Shape: [B, Seq]
        """
        batch_size = len(ids)
        max_len = max((len(seq) for seq in ids), default=0)
        padded = torch.full(
            (batch_size, max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )  # [B, Seq]
        for row, seq in enumerate(ids):
            if seq:
                padded[row, : len(seq)] = torch.tensor(
                    seq,
                    dtype=torch.long,
                    device=device,
                )
        return padded

    def _prepare_ids(
        self,
        ids: torch.Tensor | list[list[int]],
        pad_id: int,
    ) -> torch.Tensor:
        """确保 token IDs 为右侧 padding 的 Tensor。"""
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        device = next(self.model.parameters()).device
        if isinstance(ids, torch.Tensor):
            return ids.to(device)
        return self._right_pad_ids(ids, pad_id=pad_id, device=device)

    @staticmethod
    def _resolve_attn_implementation() -> str:
        """选择注意力实现，优先 Flash Attention 2。"""
        if not torch.cuda.is_available():
            return "sdpa"
        major, minor = torch.cuda.get_device_capability()
        return "flash_attention_2" if (major, minor) >= (8, 0) else "sdpa"

    def finish_train(self, global_step: int, rollout_count: int) -> None:
        """保存模型并释放 GPU 资源。"""
        if self.model is None:
            return

        # 保存 checkpoint（safetensors 格式）
        self.latest_checkpoint.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.latest_checkpoint, safe_serialization=True)

        # 每 save_rollout_freq 轮保存快照
        if rollout_count % self.cfg.save_rollout_freq == 0:
            step_checkpoint = self.model_root / f"step_{global_step:06d}"
            step_checkpoint.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(step_checkpoint, safe_serialization=True)

        # 资源释放协议：del → gc.collect() → empty_cache()
        del self.model, self.optimizer
        self.model = None
        self.optimizer = None
        gc.collect()
        torch.cuda.empty_cache()

        self.load_checkpoint = self.latest_checkpoint
