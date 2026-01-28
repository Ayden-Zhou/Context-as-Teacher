"""
下载并统一格式化数据集。

统一格式:
  - problem: 题目内容
  - solution: 完整解答
  - answer: 最终答案
"""

import json
import os
import re

os.environ["HTTP_PROXY"] = "http://iscs1411:14111234@202.120.40.41:9081"
os.environ["HTTPS_PROXY"] = "http://iscs1411:14111234@202.120.40.41:9081"

from datasets import load_dataset

OUTPUT_DIR = "data/dataset"


# ======================== 数据集格式转换 ========================
# 只处理核心字段 (problem, solution, answer)，其他字段直接保留


def transform_gsm8k(row: dict) -> dict:
    """GSM8K: question -> problem, 原answer -> solution, #### 后为 answer"""
    raw_answer = row["answer"]
    short = raw_answer.split("####")[-1].strip() if "####" in raw_answer else ""
    return {**row, "problem": row["question"], "solution": raw_answer, "answer": short}


def transform_math(row: dict) -> dict:
    """MATH: \\boxed{} 内为 answer"""
    solution = row["solution"]
    match = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", solution)
    short = match.group(1) if match else ""
    return {**row, "answer": short}


def transform_matharena(row: dict) -> dict:
    """MathArena: 无完整解答，answer 就是最终答案"""
    return {**row}


def transform_omnimath(row: dict) -> dict:
    """Omni-MATH: 已有 problem, solution, answer"""
    return {**row}


def transform_math500(row: dict) -> dict:
    """MATH-500: 已有 problem, solution, answer"""
    return {**row}


def transform_gpqa(row: dict) -> dict:
    """GPQA: Question -> problem, Correct Answer -> answer"""
    return {**row, "problem": row["Question"], "answer": row["Correct Answer"]}


# ======================== 数据集配置 ========================
# 格式: (repo, subset, transform, splits)
# - repo: 数据集仓库名称（如 "openai/gsm8k"）
# - subset: 数据集的配置/子集名称（如 "main"），没有则为 None
# - transform: 格式转换函数
# - splits: 数据划分列表（如 ["train", "test"]）

DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", transform_gsm8k, ["train", "test"]),
    "math": ("chiayewken/competition_math", None, transform_math, ["train", "test"]),
    "math_500": ("HuggingFaceH4/MATH-500", None, transform_math500, ["test"]),
    "omni_math": ("KbsdJames/Omni-MATH", None, transform_omnimath, ["test"]),
    "gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond", transform_gpqa, ["train"]),
    "gpqa_main": ("Idavidrein/gpqa", "gpqa_main", transform_gpqa, ["train"]),
    "gpqa_extended": ("Idavidrein/gpqa", "gpqa_extended", transform_gpqa, ["train"]),
    "aime_2024": ("math-ai/aime24", None, transform_matharena, ["train"]),
    "aime_2025": ("opencompass/AIME2025", None, transform_matharena, ["train"]),
    "brumo_2025": ("MathArena/brumo_2025", None, transform_matharena, ["train"]),
    "hmmt_feb_2025": ("MathArena/hmmt_feb_2025", None, transform_matharena, ["train"]),
    "hmmt_nov_2025": ("MathArena/hmmt_nov_2025", None, transform_matharena, ["train"]),
}


# ======================== 下载入口 ========================


def download(dataset_name: str = "gsm8k", raw: bool = False):
    """下载指定数据集并保存为 JSONL。

    Args:
        dataset_name: 数据集名称
        raw: 是否保存原始数据（不进行格式转换）
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"未知数据集: {dataset_name}，可选: {list(DATASETS.keys())}")

    repo, subset, transform, splits = DATASETS[dataset_name]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in splits:
        suffix = "_raw" if raw else ""
        output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{split}{suffix}.jsonl")
        print(f"正在下载 {dataset_name} ({split}){'（原始数据）' if raw else ''}...")

        ds = (
            load_dataset(repo, subset, split=split)
            if subset
            else load_dataset(repo, split=split)
        )
        print(f"下载成功！共 {len(ds)} 条数据")
        if raw:
            print(f"字段: {ds.column_names}")

        with open(output_path, "w", encoding="utf-8") as f:
            for row in ds:
                data = dict(row) if raw else transform(dict(row))
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"✅ 完成: {os.path.abspath(output_path)}\n")


if __name__ == "__main__":
    import fire

    fire.Fire(download)
