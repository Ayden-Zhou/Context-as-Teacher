"""
下载并统一格式化数据集。

统一格式:
  - question: 题目内容
  - answer: 包含过程的答案
  - short_answer: 最终答案
"""
import json
import os
import re

os.environ["HTTP_PROXY"] = "http://iscs1411:14111234@202.120.40.41:9081"
os.environ["HTTPS_PROXY"] = "http://iscs1411:14111234@202.120.40.41:9081"

from datasets import load_dataset

OUTPUT_DIR = "data/dataset"


# ======================== 数据集格式转换 ========================

def normalize_gsm8k(row: dict) -> dict:
    """GSM8K: answer 中 #### 后为 short_answer"""
    answer = row["answer"]
    short = answer.split("####")[-1].strip() if "####" in answer else ""
    return {"question": row["question"], "answer": answer, "short_answer": short}


def normalize_math(row: dict) -> dict:
    """MATH: solution 中 \\boxed{} 内为 short_answer"""
    solution = row["solution"]
    # 匹配 \boxed{...}，支持嵌套大括号
    match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', solution)
    short = match.group(1) if match else ""
    return {
        "question": row["problem"],
        "answer": solution,
        "short_answer": short,
        "level": row.get("level", ""),
        "type": row.get("type", ""),
    }


# ======================== 数据集配置 ========================

DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", normalize_gsm8k),
    "math": ("chiayewken/competition_math", None, normalize_math),
}


# ======================== 下载入口 ========================

def download(dataset_name: str="gsm8k"):
    """下载指定数据集并保存为统一格式的 JSONL。"""
    if dataset_name not in DATASETS:
        raise ValueError(f"未知数据集: {dataset_name}，可选: {list(DATASETS.keys())}")
    
    repo, subset, normalize = DATASETS[dataset_name]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for split in ["train", "test"]:
        output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{split}.jsonl")
        print(f"正在下载 {dataset_name} ({split})...")
        
        ds = load_dataset(repo, subset, split=split) if subset else load_dataset(repo, split=split)
        print(f"下载成功！共 {len(ds)} 条数据")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in ds:
                f.write(json.dumps(normalize(dict(row)), ensure_ascii=False) + "\n")
        
        print(f"✅ 完成: {os.path.abspath(output_path)}\n")


if __name__ == "__main__":
    import fire
    fire.Fire(download)

