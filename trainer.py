"""
trainer.py
实现 Context-Aware On-Policy Distillation (GKD) 逻辑。
"""

import torch
import torch.nn.functional as F
from transformers import Trainer

class ContextDistillationTrainer(Trainer):
    """自定义训练器：将 Context (Teacher) 的能力蒸馏到 Weights (Student)。

    核心流程 (Compute Loss):
        1. Exploration: Student 基于 Query 自主生成 Response (On-Policy)。
        2. Contextualization: Teacher 获取 Cached Memory (Prompt)。
        3. Alignment: 构造 Teacher 输入 (Prompt + Query + Response)。
        4. Distillation: 计算 Response 部分的 KL 散度。
    """

    def __init__(self, teacher_model, memory, temperature=1.0, *args, **kwargs):
        """
        Args:
            teacher_model: 冻结参数的 Teacher 模型。
            memory: CachedMemory 实例。
            temperature: 蒸馏温度系数。
            *args, **kwargs: 传递给父类 Trainer 的参数。
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.memory = memory
        self.temperature = temperature
        
        # 确保 Teacher 处于评估模式且不计算梯度
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """重写 Loss 计算逻辑。"""
        
        student_model = model
        tokenizer = self.processing_class # HF Trainer 自动注入
        device = student_model.device
        
        # inputs["input_ids"] 仅包含 Query，没有 Answer
        query_ids = inputs["input_ids"]
        
        # --- Phase 1: Exploration (Student Generates) ---
        # 使用 torch.no_grad() 避免生成过程构建计算图，节省显存
        with torch.no_grad():
            # 这里硬编码 max_new_tokens，实际可用 args 控制
            generated_ids = student_model.generate(
                query_ids,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # generated_ids: [Query, Response]
        # 我们需要定位 Response 的起始位置
        response_start_idx = query_ids.shape[1]
        
        # --- Phase 2: Teacher Context Injection ---
        # Teacher Input = [Prompt] + [Query] + [Response]
        prompt_ids = self.memory.get_prompt_ids(tokenizer, device)
        
        # 扩展 Prompt 以匹配 Batch Size
        batch_prompt = prompt_ids.repeat(query_ids.shape[0], 1)
        teacher_input_ids = torch.cat([batch_prompt, generated_ids], dim=1)

        # --- Phase 3: Forward Passes ---
        
        # Student Forward (需要梯度): Input = [Query, Response]
        student_outputs = student_model(generated_ids)
        student_logits = student_outputs.logits
        
        # Teacher Forward (无梯度): Input = [Prompt, Query, Response]
        with torch.no_grad():
            teacher_outputs = self.teacher_model(teacher_input_ids)
            teacher_logits = teacher_outputs.logits

        # --- Phase 4: Dynamic Logit Alignment ---
        # 关键点：Teacher 和 Student 的序列长度不同，必须精准对齐 Response 部分。
        # Logits[i] 预测的是 Token[i+1]。
        
        # Student Response Logits: 从 Query 结束处开始，到倒数第二个 Token
        s_logits_roi = student_logits[:, response_start_idx-1 : -1, :]
        
        # Teacher Response Logits: 从 Prompt + Query 结束处开始
        prompt_len = prompt_ids.shape[1]
        t_logits_roi = teacher_logits[:, prompt_len + response_start_idx - 1 : -1, :]
        
        # --- Phase 5: KL Divergence Loss ---
        loss = self._kd_loss(s_logits_roi, t_logits_roi)
        
        return (loss, student_outputs) if return_outputs else loss

    def _kd_loss(self, student_logits, teacher_logits):
        """计算 Token 级别的 KL 散度。"""
        s_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        t_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # batchmean: 对 Batch 求平均，对 Sequence 求和，符合 KL 定义
        loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
        return loss * (self.temperature ** 2)