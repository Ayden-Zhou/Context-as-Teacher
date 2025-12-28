"""
完整流程测试：Context as Teacher 反思机制
1. 使用 vLLM 生成 8 个答案
2. 验证答案正确性并分类（correct/wrong）
3. 提取 summary 部分（</think> 后的内容）
4. 生成反思 prompt 并调用 LLM 反思
5. 提取反思中的规则（\\boxed{} 内容）
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM, SamplingParams
from prompt import sample_prompt, reflect_prompt, sample_with_reflection_prompt
from utils import extract_answer, equal_func, extract_think_contents

# 配置
MODEL_PATH = "models/qwen3-1.7b"
PROBLEM = "Adrien's total salary was 30 percent higher than Lylah's. Four years later, his salary had increased, and he was earning 40% more than what he was making four years ago. If Adrien's and Lylah's salary increased simultaneously, and Adrien earned $40000 four years ago, calculate the total salary the two were receiving four years later?"
GROUND_TRUTH = "95200"

def extract_boxed(text: str) -> list[str]:
    """提取所有 \\boxed{} 中的内容"""
    return re.findall(r'\\boxed\{([^}]+)\}', text)

def main():
    # 1. 初始化模型
    llm = LLM(model=MODEL_PATH, gpu_memory_utilization=0.9, max_model_len=8096)
    params = SamplingParams(n=8, temperature=0.6, top_p=0.95, max_tokens=8096)
    
    # 2. 生成 8 个答案
    prompt = sample_prompt(MODEL_PATH, PROBLEM)
    outputs = llm.generate([prompt], params)[0]
    traces = [o.text for o in outputs.outputs]
    
    # 3. 验证并分类
    correct, wrong = [], []
    for trace in traces:
        ans = extract_answer(trace)
        (correct if equal_func(ans or "", GROUND_TRUTH) else wrong).append(trace)
    
    print(f"✅ 正确: {len(correct)} / 8,  ❌ 错误: {len(wrong)} / 8\n")
    
    # 4. 提取 summary 部分（</think> 后的内容）
    summaries = {
        "correct": [s for _, s in extract_think_contents(correct) if s],
        "wrong": [s for _, s in extract_think_contents(wrong) if s],
    }

    
    # 5. 如有错误，生成反思
    if wrong:
        ref_prompt = reflect_prompt(MODEL_PATH, summaries["wrong"], summaries["correct"], PROBLEM)
        print("="*100+f"\n{ref_prompt}\n"+"="*100)
        # 增加 max_tokens 以确保模型有足够空间输出完整的反思和规则
        reflection = llm.generate([ref_prompt], SamplingParams(max_tokens=4096))[0].outputs[0].text
        
        # 6. 提取规则
        commands = extract_boxed(reflection)
        
        print("=" * 50)
        print("🔍 反思结果:")
        print(reflection)
        print("=" * 50)
        print("📋 提取的规则:", commands)
        
        # 7. 利用反思规则重新 sample
        if commands:
            reflection_text = " ".join(commands)
            print("\n" + "=" * 50)
            print("🔄 利用反思规则重新生成答案...")
            print("=" * 50)
            
            # 生成带反思规则的 prompt
            new_prompt = sample_with_reflection_prompt(MODEL_PATH, PROBLEM, reflection_text)
            
            # 重新生成 8 个答案
            new_outputs = llm.generate([new_prompt], params)[0]
            new_traces = [o.text for o in new_outputs.outputs]
            
            # 验证新答案
            new_correct, new_wrong = [], []
            for trace in new_traces:
                ans = extract_answer(trace)
                (new_correct if equal_func(ans or "", GROUND_TRUTH) else new_wrong).append(trace)
            
            print(f"\n📊 反思后结果: ✅ 正确: {len(new_correct)} / 8,  ❌ 错误: {len(new_wrong)} / 8")
            print(f"📈 提升: {len(new_correct) - len(correct)} 个答案从错误变为正确")
        else:
            print("⚠️ 未能提取到规则，跳过重新 sample")
    else:
        print("🎉 所有答案都正确，无需反思！")


if __name__ == "__main__":
    main()

