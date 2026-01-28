"""
测试数据集的逐题 pass@1 分析：
1. 对每道题生成 8 个 trace
2. 统计每题的 pass@1
3. 按 pass@1 从低到高排序并保存为 CSV
4. 打印 pass@1 最低的前 10 题

用法:
    python test_pass@1.py [dataset_name]

    dataset_name 可选值:
    - gsm8k (默认)
    - math
"""

import json
import sys
from pathlib import Path

import fire
import pandas as pd
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt import sample_prompt

from utils import equal_func, extract_answer, pass_at_k

# 配置
MODEL_PATH = "models/qwen3-1.7b"
NUM_TRACES = 8

# 数据集配置
DATASETS = {
    "gsm8k": {
        "path": "data/dataset/gsm8k_test.jsonl",
        "output": "data/gsm8k_pass@1.csv",
    },
    "math": {
        "path": "data/dataset/math_test.jsonl",
        "output": "data/math_pass@1.csv",
    },
}


def main(dataset_name: str = "gsm8k"):
    # 0. 获取数据集配置
    print(f"测试{dataset_name}数据集")
    if dataset_name not in DATASETS:
        print(f"❌ 错误: 未知数据集 '{dataset_name}'")
        print(f"   可选数据集: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    config = DATASETS[dataset_name]
    dataset_path = config["path"]
    output_csv = config["output"]

    print(f"📚 数据集: {dataset_name}")
    print(f"📂 路径: {dataset_path}")
    print(f"💾 输出: {output_csv}")
    print()

    # 1. 加载数据集
    data = [json.loads(line) for line in open(dataset_path)]

    # 2. 初始化模型
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.95,
        max_model_len=8096,
        enable_prefix_caching=True,
    )
    params = SamplingParams(n=NUM_TRACES, temperature=0.6, top_p=0.95, max_tokens=6144)

    # 3. 批量生成
    prompts = [sample_prompt(MODEL_PATH, d["problem"]) for d in data]
    print(f"🚀 生成 {len(data)} 题 × {NUM_TRACES} traces...")
    outputs = llm.generate(prompts, params)

    # 4. 统计 pass@1
    results = []
    for i, (item, output) in enumerate(zip(data, outputs)):
        traces = [o.text for o in output.outputs]
        gold = item["short_answer"]  # 直接使用 short_answer 字段
        correct = sum(equal_func(extract_answer(t) or "", gold) for t in traces)
        p1 = pass_at_k(NUM_TRACES, correct, 1)

        results.append(
            {
                "idx": i + 1,
                "question": item["problem"][:30],
                "correct": correct,
                "total": NUM_TRACES,
                "pass@1": p1,
            }
        )

    # 5. 转为 DataFrame 并排序
    df = pd.DataFrame(results).sort_values("pass@1")

    # 6. 保存 CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"💾 已保存: {output_csv}")

    # 7. 打印 pass@1 最低的前 10 题
    print("\n" + "=" * 80)
    print("📉 pass@1 最低的前 10 题:")
    print("=" * 80)
    for _, row in df.head(10).iterrows():
        print(
            f"[{row['idx']:4d}] pass@1={row['pass@1']:.1%} ({row['correct']}/{row['total']}) - {row['question'][:80]}"
        )

    # 8. 统计平均 pass@1
    avg_pass_at_1 = df["pass@1"].mean()
    print(f"\n📊 平均 pass@1: {avg_pass_at_1:.2%}")


if __name__ == "__main__":
    fire.Fire(main)
