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
    questions: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    solutions: list[str] = field(default_factory=list)
    prompts: list[str] | None = None

    # ===== 2. Token IDs (GPU/CPU, Tensor) =====
    # [B, prompt_len] - 用于 vLLM 输入
    prompt_ids: torch.Tensor | None = None
    # [B, response_len] - vLLM 输出
    response_ids: torch.Tensor | None = None
    # [B, seq_len] - 用于 HF 训练 (Concat后)
    input_ids: torch.Tensor | None = None
    # [B, seq_len]
    attention_mask: torch.Tensor | None = None
    
    # ===== 3. 元数据 (任意附加信息) =====
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 确保所有列表长度一致
        if not self.questions:
            return
        bs = len(self.questions)
        for name in ("answers", "solutions", "prompts"):
            if (val := getattr(self, name)) is not None and len(val) != bs:
                raise ValueError(f"{name} 长度 ({len(val)}) 与 questions ({bs}) 不一致")


    def __len__(self) -> int:
        if self.input_ids is not None:
            return self.input_ids.shape[0]
        return len(self.questions) if self.questions else 0

    @property
    def batch_size(self) -> int:
        return len(self)
    
    @property
    def prompt_len(self) -> int | None:
        return self.prompt_ids.shape[1] if self.prompt_ids is not None else None

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
            source = self.questions
        
        # 计算长度
        if isinstance(source, torch.Tensor):
            # Tensor: 如果已 padding，所有长度相同，退化为 questions
            if source.shape[0] > 0 and len(set(source.shape[1] for _ in range(1))) == 1:
                source = self.questions
                return [len(s) for s in source] if source else [0] * len(self)
            return [source.shape[1]] * source.shape[0]
        elif isinstance(source, list):
            return [len(s) for s in source]
        else:
            return [0] * len(self)

    def sample(
        self: T,
        n: int,
        strategy: Literal["random", "length"] = "random",
        length_key: str = "questions",
    ) -> T:
        """从 Batch 中采样 n 个样本。
        
        Args:
            n: 采样数量
            strategy: 采样策略
                - "random": 随机采样
                - "length": 按长度排序后均匀采样（减少 padding）
            length_key: 用于计算长度的字段
        
        Returns:
            采样后的新 Batch
        """
        if n >= len(self):
            return self
        
        if strategy == "random":
            indices = random.sample(range(len(self)), n)
        elif strategy == "length":
            lengths = self._get_lengths(length_key)
            sorted_indices = sorted(range(len(self)), key=lambda i: lengths[i])
            step = len(self) / n
            indices = [sorted_indices[int(i * step)] for i in range(n)]
        else:
            raise ValueError(f"未知采样策略: {strategy}")
        
        return self[indices]

    def split(
        self: T,
        batch_size: int,
        shuffle: bool = True,
        by_length: bool = False,
        length_key: str = "questions",
    ) -> Iterator[T]:
        """将 Batch 切分为多个小 Batch。
        
        Args:
            batch_size: 每个小 Batch 的大小
            shuffle: 是否打乱顺序
            by_length: 是否按长度排序后再切分（减少 padding）
            length_key: 用于计算长度的字段
        
        Yields:
            切分后的小 Batch
        """
        indices = list(range(len(self)))
        
        if by_length:
            lengths = self._get_lengths(length_key)
            indices.sort(key=lambda i: lengths[i])
        
        if shuffle and not by_length:
            random.shuffle(indices)
        elif shuffle and by_length:
            # 按长度分组后，组间 shuffle（相近长度仍在一起）
            chunks = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
            random.shuffle(chunks)
            indices = [i for chunk in chunks for i in chunk]
        
        for i in range(0, len(indices), batch_size):
            yield self[indices[i:i + batch_size]]

    # ========== Tianshou 风格的核心魔法 ==========

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

    @classmethod
    def stack(cls: type[T], batch_list: Sequence[T]) -> T:
        """相当于 PyTorch 的 collate_fn。将多个 Batch 对象堆叠成一个。"""
        if not batch_list:
            return cls()
        
        first = batch_list[0]
        data = {}
        
        for field_name in first.__dataclass_fields__:
            val_example = getattr(first, field_name)
            if val_example is None:
                data[field_name] = None
                continue
            
            # 收集所有 batch 的该字段
            all_vals = [getattr(b, field_name) for b in batch_list]

            if isinstance(val_example, torch.Tensor):
                data[field_name] = torch.cat(all_vals, dim=0)
            
            elif isinstance(val_example, list):
                # List 直接相加 (extend)
                merged_list = []
                for l in all_vals:
                    merged_list.extend(l)
                data[field_name] = merged_list
            
            elif isinstance(val_example, dict):
                # Info 字典通常不合并，或者取第一个
                data[field_name] = val_example
                
        return cls(**data)

    # ========== 业务逻辑辅助 ==========

    def update_from_vllm(self, outputs: list, pad_token_id: int) -> None:
        """接收 vLLM 的 RequestOutput 列表，就地更新 response_ids。
        
        Args:
            outputs: vLLM 的 RequestOutput 列表
            pad_token_id: 填充 token ID（必须与 tokenizer 一致）
        """
        # 1. 解析 output_ids
        generated_sequences = [output.outputs[0].token_ids for output in outputs]
        
        if not generated_sequences:
            return
        
        # 2. Pad (vLLM 输出长度可能不一)
        max_len = max(len(seq) for seq in generated_sequences)
        padded_ids = [
            list(seq) + [pad_token_id] * (max_len - len(seq))
            for seq in generated_sequences
        ]
        
        # 3. 转为 Tensor
        device = self.prompt_ids.device if self.prompt_ids is not None else "cpu"
        self.response_ids = torch.tensor(padded_ids, device=device, dtype=torch.long)