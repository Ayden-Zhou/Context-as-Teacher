"""Prompt builders for Deep Think with Confidence."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from transformers import AutoTokenizer

if TYPE_CHECKING:
    from .dataclass import Batch


# ==================== 缓存工具 ====================


@cache
def get_tokenizer(model_path: str) -> AutoTokenizer:
    """缓存 tokenizer 加载（模型路径相同时复用）"""
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


@cache
def get_template_ids(model_path: str) -> tuple[list[int], list[int]]:
    """缓存 chat template 的前缀和后缀 token ids。
    
    通过占位符拆分模板，避免每次重复 tokenize 固定部分。
    """
    tokenizer = get_tokenizer(model_path)
    placeholder = "<<<PLACEHOLDER>>>"
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": placeholder}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix, suffix = full.split(placeholder)
    return (
        tokenizer.encode(prefix, add_special_tokens=False),
        tokenizer.encode(suffix, add_special_tokens=False),
    )


# ==================== 训练用接口 ====================


INSTRUCTION_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MEMORY_INTRO = "\n\nTo sovle this question, you should follow the following rules: "


@cache
def get_instruction_ids(model_path: str) -> list[int]:
    tokenizer = get_tokenizer(model_path)
    return tokenizer.encode(INSTRUCTION_SUFFIX, add_special_tokens=False)


@cache
def get_memory_intro_ids(model_path: str) -> list[int]:
    tokenizer = get_tokenizer(model_path)
    return tokenizer.encode(MEMORY_INTRO, add_special_tokens=False)


def build_prompt_ids(
    problems: list[str],
    memories: list[str],
    model_path: str,
    responses_per_prompt: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """为 Batch 构建 student/teacher prompt_ids。
    
    仅对动态内容 tokenize；模板前后缀、指令后缀与 memory 前缀均缓存。
    
    Args:
        problems: 包含 problems 的 list
        memories: 包含 memories 的 list
        model_path: 模型路径（用于加载 tokenizer）
        responses_per_prompt: 每题采样的响应数，用于扩展 prompt_ids
    
    Returns:
        tuple[list[list[int]], list[list[int]]]: (student_prompt_ids, teacher_prompt_ids)
    """
    tokenizer = get_tokenizer(model_path)
    prefix_ids, suffix_ids = get_template_ids(model_path)
    instruction_ids = get_instruction_ids(model_path)
    memory_intro_ids = get_memory_intro_ids(model_path)

    student_ids = [
        prefix_ids
        + tokenizer.encode(q, add_special_tokens=False)
        + instruction_ids
        + suffix_ids
        for q in problems
    ]

    teacher_ids = [
        prefix_ids
        + tokenizer.encode(q, add_special_tokens=False)
        + memory_intro_ids
        + tokenizer.encode(m, add_special_tokens=False)
        + instruction_ids
        + suffix_ids
        for q, m in zip(problems, memories)
    ]

    student_ids = [p for p in student_ids for _ in range(responses_per_prompt)]
    teacher_ids = [p for p in teacher_ids for _ in range(responses_per_prompt)]

    return student_ids, teacher_ids


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


