"""Prompt builders for Deep Think with Confidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import AutoTokenizer

if TYPE_CHECKING:
    from .dataclass import Batch


# ==================== 训练用接口 ====================


def build_prompts(batch: Batch, model_name: str) -> list[str]:
    """为 Batch 中的所有 questions 构建 prompts。
    
    Args:
        batch: 包含 questions 的 Batch
        model_name: 模型名称（用于加载 tokenizer）
    
    Returns:
        list[str]: 构建好的 prompts
    """
    # TODO: 实现批量 prompt 构建
    pass


# ==================== 原有函数 ====================


def prepare_prompt(
    model_path: str,
    instruction: str,
) -> str:
    """Prepare prompt for Qwen models."""

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    messages = [{"role": "user", "content": instruction}]

    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return full_prompt


def reflect_prompt(
    model_path: str,
    failed_samples: list[str],
    successful_samples: list[str],
    problem: str,
) -> str:
    """Prepare prompt for memory by building instruction and applying template."""
    fmt = lambda items: "\n\n".join(
        f"[Sample {i + 1}]\n{s}\n" for i, s in enumerate(items)
    )
    failed_text = fmt(failed_samples)
    successful_text = (
        fmt(successful_samples) if successful_samples else "None available."
    )

    instruction = f"""
        You are a strict math tutor. I provided an assistant to solve the problem for me:

        <Problem>
        {problem}
        </Problem>

        <Wrong_Answers>
        {failed_text}
        </Wrong_Answers>

        <Correct_Answers>
        {successful_text}
        </Correct_Answers>
        
        
        Your task is to write an instruction for the assistant to solve the problem correctly. Put your final instruction within \\boxed{"{}"}.
        """

    return prepare_prompt(model_path, instruction)


def sample_with_memory_prompt(model_path: str, question: str, memory) -> str:
    instruction = (
        question
        + "\n\n"
        + "To sovle this question, you should follow the following rules: "
        + memory
        + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    return prepare_prompt(model_path, instruction)


def sample_prompt(model_path: str, question: str) -> str:
    instruction = (
        question
        + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    return prepare_prompt(model_path, instruction)


if __name__ == "__main__":
    print(reflect_prompt("models/qwen3-1.7b", ["1+1=3"], ["1+1=2"], "1+1=?"))
