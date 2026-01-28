"""
极简版 Context as Teacher：sample → 反思 → 再 sample
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt import reflect_prompt, sample_prompt, sample_with_reflection_prompt
from vllm import LLM, SamplingParams

from utils import equal_func, extract_answer, split_trace

# 配置
MODEL_PATH = "models/qwen3-1.7b"
PROBLEM = "John has 3 boxes.  Each box is 5 inches by 6 inches by 4 inches.  The walls are 1 inch thick.  What is the total inner volume of all 3 boxes?"
GROUND_TRUTH = "72"
N = 32


def sample_and_classify(llm, prompt, params, gt, debug=False):
    """Sample N 个答案并分类为正确/错误"""
    from collections import Counter

    traces = [o.text for o in llm.generate([prompt], params)[0].outputs]
    correct, wrong = [], []
    answers = []  # 收集所有答案用于调试
    for t in traces:
        ans = extract_answer(t) or ""
        answers.append(ans)
        (correct if equal_func(ans, gt) else wrong).append(t)
    if debug:
        print(f"🔍 答案统计: {dict(Counter(answers))}")
        print(f"🎯 正确答案: {gt}")
    return correct, wrong


def get_summary(traces):
    """提取 summary 部分"""
    return [s for _, s in split_trace(traces) if s]


def main():
    llm = LLM(model=MODEL_PATH, gpu_memory_utilization=0.95, max_model_len=8096)
    params = SamplingParams(n=N, temperature=0.7, top_p=0.95, max_tokens=8096)
    prompt = sample_prompt(MODEL_PATH, PROBLEM)

    # 第一次 sample
    correct, wrong = sample_and_classify(llm, prompt, params, GROUND_TRUTH, debug=True)
    print(f"📊 第一次: ✅ {len(correct)}/{N}  ❌ {len(wrong)}/{N}")

    if not wrong:
        print("🎉 全部正确！")
        return

    # 没有正确样本时再 sample 一次
    if not correct:
        print("⚠️ 无正确样本，再次 sample...")
        c2, _ = sample_and_classify(llm, prompt, params, GROUND_TRUTH, debug=True)
        correct = c2
        if not correct:
            print("❌ 仍无正确样本，无法反思")
            return

    # 反思：用1个正确样本
    ref_prompt = reflect_prompt(
        MODEL_PATH, get_summary(wrong), get_summary(correct[:1]), PROBLEM
    )
    reflection = (
        llm.generate([ref_prompt], SamplingParams(max_tokens=4096))[0].outputs[0].text
    )
    _, rules = next(split_trace([reflection]))

    print(f"📋 规则: {rules}")

    if not rules:
        print("⚠️ 未提取到规则")
        return

    # 用反思规则重新 sample
    new_prompt = sample_with_reflection_prompt(MODEL_PATH, PROBLEM, " ".join(rules))
    new_correct, new_wrong = sample_and_classify(llm, new_prompt, params, GROUND_TRUTH)

    print(f"📊 反思后: ✅ {len(new_correct)}/{N}  ❌ {len(new_wrong)}/{N}")
    print(f"📈 提升: {len(new_correct) - len(correct)}")


if __name__ == "__main__":
    main()
