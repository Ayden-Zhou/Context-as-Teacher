import random
from contextlib import ContextDecorator
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from math_verify import parse, verify
from torch.utils.data import DataLoader

from context_as_teacher.dataclass import Batch

_THINK_MARKERS = {
    "qwen": ("<think>", "</think>"),
    "gpt": (
        "<|channel|>analysis<|message|>",
        "<|end|><|start|>assistant<|channel|>final<|message|>",
    ),
}


def generate_run_id(model_name, dataset_name) -> str:
    """生成格式为 MMDD_HHMM_XXX 的运行 ID。"""
    now = datetime.now()
    timestamp = now.strftime("%m%d_%H%M")
    random_stamp = random.randint(100, 999)
    return f"{model_name}_{dataset_name}_{timestamp}_{random_stamp:03d}"


class Timer(ContextDecorator):
    """轻量计时器。

    Args:
        label: 标签，用于输出与登记。
        text: 结束时输出模板，变量包含 ``label`` 与 ``seconds``。
        start: 进入上下文时的提示文案（可选）。
        sink: 输出函数，默认使用 ``print``。
        record: 是否将耗时记录到类级注册表中。
        accumulate: 若为 True，则在注册表中对同标签进行累加，否则覆盖。

    Attributes:
        seconds: 本次上下文的耗时（退出后可用）。
    """

    registry: Dict[str, float] = {}

    def __init__(
        self,
        label: str = "timer",
        text: str = "{label} in {seconds:.2f} seconds",
        start: Optional[str] = None,
        sink=print,
        record: bool = True,
        accumulate: bool = False,
    ):
        self.label = label
        self.text = text
        self.start = start
        self.sink = sink
        self.record = record
        self.accumulate = accumulate
        self.seconds = 0.0

    def __enter__(self):
        if self.start:
            self.sink(self.start)
        self._t0 = perf_counter()
        return self

    def __exit__(self, *exc):
        secs = perf_counter() - self._t0
        self.seconds = secs
        self.sink(self.text.format(label=self.label, seconds=secs))
        if self.record:
            if self.accumulate and self.label in Timer.registry:
                Timer.registry[self.label] += secs
            else:
                Timer.registry[self.label] = secs
        return False

    @classmethod
    def get(cls, label: str, default: float = 0.0) -> float:
        """获取标签对应的耗时（秒）。

        Args:
            label: 计时标签。
            default: 当不存在记录时的默认返回值。
        Returns:
            float: 耗时（秒）。
        """
        return cls.registry.get(label, default)

    @classmethod
    def clear(cls, label: Optional[str] = None) -> None:
        """清除注册表记录。"""
        if label is None:
            cls.registry.clear()
        else:
            cls.registry.pop(label, None)


def extract_answer(text: str) -> Optional[str]:
    """从文本中提取 ``\boxed{}`` 形式的答案，并清理 LaTeX 标记。"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()

        # 清理 LaTeX \text{} 标记
        result = a.strip()
        if "\\text{" in result and "}" in result:
            while "\\text{" in result:
                start = result.find("\\text{")
                if start == -1:
                    break
                end = result.find("}", start)
                if end == -1:
                    break
                # Replace \text{content} with just content
                content = result[start + 6 : end]  # 6 is length of '\text{'
                result = result[:start] + content + result[end + 1 :]

        return result

    return None


def equal_func(answer: str, ground_truth: str) -> bool:
    """
    最简 math_verify 实现。
    输入可以是包含无关文本的原始字符串，parse 会自动提取核心数学内容。
    """
    try:
        # parse() 自动寻找 \boxed{} 或最后出现的数值/公式
        # verify() 自动处理符号等价性 (x+1 == 1+x) 和数值精度
        return verify(parse(answer), parse(ground_truth))
    except Exception:
        # 防止极端情况下解析器崩溃（例如输入了极其畸形的 LaTeX）
        return False


def split_trace(
    traces: List[str],
    model_type: str = "qwen",
) -> Iterator[Tuple[Optional[str], Optional[str]]]:
    """批量提取 trace 的思考部分和总结部分。"""
    start, end = _THINK_MARKERS[model_type]
    for trace in traces:
        if trace and start in trace and end in trace:
            yield (
                trace.split(start, 1)[1].split(end, 1)[0].strip(),
                trace.split(end, 1)[1].strip(),
            )
        else:
            yield (
                trace.strip() if trace else None,
                None,
            )  # 没有完整标记，全部当作 reasoning


def pass_at_k(num_traces: int, num_correct: int, k: int) -> float:
    """计算 pass@k 期望值。

    Args:
        num_traces: 总样本数（trace 数）
        num_correct: 正确样本数（正确 trace 数）
        k: 采样数

    Returns:
        pass@k 概率
    """
    if num_traces - num_correct < k:
        return 1.0
    # 数值稳定的计算方式: 1 - ∏_{i=0}^{k-1} (n-c-i) / (n-i)
    return 1.0 - np.prod(
        [(num_traces - num_correct - i) / (num_traces - i) for i in range(k)]
    )


# ==================== 数据加载与环境设置 ====================


def collate_fn(batch: list[dict]) -> Batch:
    """将 HuggingFace Dataset 的 batch 转换为 Batch 对象"""
    return Batch(
        problems=[d["problem"] for d in batch],
        answers=[d.get("answer", "") for d in batch],
        memories=[d.get("solution", "") for d in batch],
    )


def create_dataloader(
    *,
    data_path: str | Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """创建无限循环的 DataLoader"""
    ds = load_dataset("json", data_files=str(data_path))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )


def infinite_dataloader(dataloader: DataLoader):
    """无限循环的数据迭代器"""
    while True:
        yield from dataloader


def set_global_seed(seed: int) -> None:
    """设置全局随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
