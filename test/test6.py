"""
测试 GSM8K 数据集的逐题正确率分析：
1. 对每道题生成 8 个 trace
2. 统计每题的正确率
3. 按正确率从低到高排序并保存为 CSV
4. 打印正确率最低的前 10 题
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt import sample_prompt
from utils import extract_answer, equal_func

# 配置
MODEL_PATH = "models/qwen3-1.7b"
DATASET_PATH = "data/dataset/gsm8k_test.jsonl"
OUTPUT_CSV = "data/accuracy.csv"
NUM_TRACES = 8

def extract_gold(answer: str) -> str:
    """提取 GSM8K 标准答案（#### 后的内容）"""
    return (m.group(1).strip() if (m := re.search(r'####\s*(.*)', answer)) else "")

def main():
    # 1. 加载数据集
    data = [json.loads(line) for line in open(DATASET_PATH)]
    
    # 2. 初始化模型
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enable_prefix_caching=True,
    )
    params = SamplingParams(n=NUM_TRACES, temperature=0.7, top_p=0.95, max_tokens=2048)
    
    # 3. 批量生成
    prompts = [sample_prompt(MODEL_PATH, d['question']) for d in data]
    print(f"🚀 生成 {len(data)} 题 × {NUM_TRACES} traces...")
    outputs = llm.generate(prompts, params)
    
    # 4. 统计正确率
    results = []
    for i, (item, output) in enumerate(zip(data, outputs)):
        traces = [o.text for o in output.outputs]
        gold = extract_gold(item['answer'])
        correct = sum(equal_func(extract_answer(t) or "", gold) for t in traces)
        accuracy = correct / NUM_TRACES
        
        results.append({
            'idx': i,
            'question': item['question'][:80],  # 截断问题文本
            'correct': correct,
            'total': NUM_TRACES,
            'accuracy': accuracy,
        })
    
    # 5. 转为 DataFrame 并排序
    df = pd.DataFrame(results).sort_values('accuracy')
    
    # 6. 保存 CSV
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 已保存: {OUTPUT_CSV}")
    
    # 7. 打印最低的前 10 题
    print("\n" + "="*80)
    print("📉 正确率最低的前 10 题:")
    print("="*80)
    for _, row in df.head(10).iterrows():
        print(f"[{row['idx']:4d}] {row['accuracy']:.1%} ({row['correct']}/{row['total']}) - {row['question']}")
    
    # 8. 统计总体正确率
    overall_acc = df['accuracy'].mean()
    print(f"\n📊 总体正确率: {overall_acc:.2%}")

if __name__ == "__main__":
    main()

