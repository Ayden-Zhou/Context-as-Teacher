"""
memory.py
管理动态缓存记忆 (Cached Memory)，作为 OPRO (Optimization by PROmpting) 的接口。
"""

from typing import Dict, Any, List
import torch
from transformers import PreTrainedTokenizer

class CachedMemory:
    """管理 Teacher 模型的 System Prompt。

    该类维护一个不断进化的 Prompt (Cached Memory)。
    在训练循环中，Monitor 会调用 update() 传入反馈，
    未来可在此处接入 Meta-LLM 基于反馈优化 Prompt。

    Attributes:
        prompt_text (str): 当前的 System Prompt。
        history (List): Prompt 的演变历史。
    """

    def __init__(self, initial_prompt: str = ""):
        """初始化缓存记忆。

        Args:
            initial_prompt: 初始的 System Prompt 文本。
        """
        self.prompt_text = initial_prompt
        self.history: List[Dict[str, Any]] = []

    def get_prompt_ids(self, tokenizer: PreTrainedTokenizer, device: torch.device) -> torch.Tensor:
        """获取编码后的 Prompt Tensor，用于拼接到 Teacher 输入前。

        Args:
            tokenizer: 分词器实例。
            device: 目标设备 (CPU/GPU)。

        Returns:
            torch.Tensor: Shape [1, prompt_len]
        """
        if not self.prompt_text:
            return torch.tensor([[]], dtype=torch.long, device=device)
        
        # add_special_tokens=False 防止在 Prompt 前重复添加 BOS Token
        return tokenizer.encode(
            self.prompt_text, 
            return_tensors="pt", 
            add_special_tokens=False
        ).to(device)

    def update(self, feedback_stats: Dict[str, Any]):
        """[OPRO 接口] 根据训练反馈更新 Cached Memory。

        Args:
            feedback_stats: 包含 Loss, 错题集或 Gradient 信息的字典。
        """
        # Record history
        self.history.append({
            "prompt": self.prompt_text,
            "stats": feedback_stats
        })
        
        # TODO(User): 在此处接入 Meta-LLM (e.g., GPT-4) API。
        # 逻辑：Analysis(feedback) -> Generate New Prompt -> Update self.prompt_text
        # self.prompt_text = new_optimized_prompt
        pass