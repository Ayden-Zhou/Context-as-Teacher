from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Sequence, TypeVar
import torch

T = TypeVar("T", bound="Batch")

@dataclass
class Batch:
    """
    智能数据载体。支持 .to(), 切片, 堆叠, 以及类似字典的访问。
    """
    # ===== 1. 原始文本 (CPU, List) =====
    problems: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    memories: list[str] = field(default_factory=list)
    prompts: list[str] | None = None

    # ===== 2. Token IDs (GPU/CPU, Tensor) =====
    # [B, prompt_len] - 用于 vLLM 输入
    student_prompt_ids: list[list[int]] | torch.Tensor | None = None
    # [B, prompt_len] - Teacher 输入（含 memory）
    teacher_prompt_ids: list[list[int]] | torch.Tensor | None = None
    # [B, response_len] - vLLM 输出
    response_ids: list[list[int]] | torch.Tensor | None = None
    # [B, seq_len] - 用于 HF 训练 (Concat后)
    input_ids: torch.Tensor | None = None
    # [B, seq_len]
    attention_mask: torch.Tensor | None = None
    
    # ===== 3. 元数据 (任意附加信息) =====
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 确保所有列表长度一致
        if not self.problems:
            return
        bs = len(self.problems)
        for name in ("answers", "memories", "prompts"):
            if (val := getattr(self, name)) is not None and len(val) != bs:
                raise ValueError(f"{name} 长度 ({len(val)}) 与 questions ({bs}) 不一致")


    def __len__(self) -> int:
        if self.input_ids is not None:
            return self.input_ids.shape[0]
        return len(self.problems) if self.problems else 0

    @property
    def batch_size(self) -> int:
        return len(self)

    # ========== 采样方法 ==========

    def _get_lengths(self, length_key: str = "questions") -> list[int]:
        """获取每个样本的长度，用于 Smart Batching。
        
        优先级：attention_mask.sum() > 指定字段 > questions
        """
        # 如果有 attention_mask，用它计算真实长度（最准确）
        if self.attention_mask is not None:
            return self.attention_mask.sum(dim=1).tolist()
        
        # 尝试获取指定字段
        source = getattr(self, length_key, None)
        
        # Fallback 到 questions
        if source is None:
            source = self.problems
        
        # 计算长度
        if isinstance(source, torch.Tensor):
            # Tensor: 如果已 padding，所有长度相同，退化为 questions
            if source.shape[0] > 0 and len(set(source.shape[1] for _ in range(1))) == 1:
                source = self.problems
                return [len(s) for s in source] if source else [0] * len(self)
            return [source.shape[1]] * source.shape[0]
        elif isinstance(source, list):
            return [len(s) for s in source]
        else:
            return [0] * len(self)


    def sample_batches(
        self: T,
        batch_size: int,
        n_batches: int,
        device: str | torch.device | None = None,
        by_length: bool = False,
        length_key: str = "response_ids",
    ) -> Iterator[T]:
        """从 buffer 中采样 n_batches 个 mini-batch（有放回）。
        
        与 split 的区别：
        - split: 不重复切分，遍历一遍
        - sample_batches: 有放回采样，总共采 n_batches 次
        
        Args:
            batch_size: 每个 mini-batch 的大小
            n_batches: 采样次数
            device: 目标设备（可选）
            by_length: 是否按长度分组采样（减少 padding）
            length_key: 用于计算长度的字段
        
        Yields:
            采样后的 mini-batch
        """
        n = len(self)
        
        if not by_length:
            # 纯随机采样
            for _ in range(n_batches):
                indices = random.choices(range(n), k=batch_size)
                batch = self[indices]
                yield batch.to(device) if device else batch
        else:
            # 按长度排序后，在相邻区域内采样（长度相近 → 减少 padding）
            lengths = self._get_lengths(length_key)
            sorted_indices = sorted(range(n), key=lambda i: lengths[i])
            
            for _ in range(n_batches):
                # 随机选一个起始位置，取连续 batch_size 个
                start = random.randint(0, n - batch_size)
                indices = sorted_indices[start : start + batch_size]
                batch = self[indices]
                yield batch.to(device) if device else batch

    def to(self: T, device: str | torch.device) -> T:
        """递归将内部所有 Tensor 移动到指定设备，并返回自身 (流式接口)。"""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                setattr(self, field_name, value.to(device))
            elif isinstance(value, dict): # 处理 info 中的 tensor
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value[k] = v.to(device)
        return self

    def __getitem__(self: T, index: int | slice | torch.Tensor | list) -> T:
        """支持 batch[0], batch[0:5], batch[indices] 切片。"""
        # 统一将 int 转换为 slice，保持维度一致性
        if isinstance(index, int):
            index = slice(index, index + 1)
        
        new_data = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if v is None:
                new_data[k] = None
                continue
            
            if isinstance(v, torch.Tensor):
                # Tensor 原生支持 slice 和 list/tensor index
                new_data[k] = v[index]
            
            elif isinstance(v, list):
                if isinstance(index, slice):
                    # List 原生支持 slice
                    new_data[k] = v[index]
                else:
                    # List 不支持 Tensor/List 索引，需要手动构造
                    idx_list = index.tolist() if isinstance(index, torch.Tensor) else index
                    new_data[k] = [v[i] for i in idx_list]
            
            elif isinstance(v, dict):
                # 元数据浅拷贝，不切分
                new_data[k] = v
        
        return self.__class__(**new_data)

    def repeat_interleave(self: T, repeat: int) -> T:
        """按 batch 维度重复样本（仅重复长度等于 batch_size 的字段）。"""
        if repeat <= 1:
            return self
        base_len = len(self)
        data = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is None:
                data[field_name] = None
                continue

            if isinstance(value, torch.Tensor):
                data[field_name] = (
                    value.repeat_interleave(repeat, dim=0)
                    if value.shape[0] == base_len
                    else value
                )
            elif isinstance(value, list):
                data[field_name] = (
                    [v for v in value for _ in range(repeat)]
                    if len(value) == base_len
                    else value
                )
            elif isinstance(value, dict):
                copied = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == base_len:
                        copied[k] = v.repeat_interleave(repeat, dim=0)
                    elif isinstance(v, list) and len(v) == base_len:
                        copied[k] = [x for x in v for _ in range(repeat)]
                    else:
                        copied[k] = v
                data[field_name] = copied
        return self.__class__(**data)

