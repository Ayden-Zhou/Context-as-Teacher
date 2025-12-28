import json
import os

os.environ["HTTP_PROXY"] = "http://iscs1411:14111234@202.120.40.41:9081"
os.environ["HTTPS_PROXY"] = "http://iscs1411:14111234@202.120.40.41:9081"

from datasets import load_dataset

OUTPUT_DIR = "data/dataset"


def download_gsm8k():
    """下载 GSM8K 数据集并保存为 JSONL 格式。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for split in ["train", "test"]:
        output_path = os.path.join(OUTPUT_DIR, f"gsm8k_{split}.jsonl")
        print(f"正在下载 gsm8k ({split})...")
        
        ds = load_dataset("openai/gsm8k", "main", split=split)
        print(f"下载成功！共 {len(ds)} 条数据")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in ds:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        
        print(f"✅ 完成: {os.path.abspath(output_path)}\n")


def download_competition_math():
    """下载 competition_math 数据集并保存为 JSONL 格式。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for split in ["train", "test"]:
        output_path = os.path.join(OUTPUT_DIR, f"competition_math_{split}.jsonl")
        print(f"正在下载 competition_math ({split})...")
        
        ds = load_dataset("chiayewken/competition_math", split=split)
        print(f"下载成功！共 {len(ds)} 条数据")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in ds:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        
        print(f"✅ 完成: {os.path.abspath(output_path)}\n")


if __name__ == "__main__":
    download_competition_math()

