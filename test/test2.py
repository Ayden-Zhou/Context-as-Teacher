"""
测试 EleutherAI lm-evaluation-harness 的评估逻辑：
- GSM8K 标准：基于数值的浮点比较（处理逗号、####分隔符）
- MATH 标准：基于字符串的精确匹配（提取 \\boxed{}）
- 展示 Harness 默认逻辑的局限性（不支持数学等价性）
"""
import re
import math

class HarnessMathEval:
    """
    完全复刻 EleutherAI/lm-evaluation-harness 的核心清洗与评估逻辑。
    涵盖了 GSM8K (数值推理) 和 MATH (代数/符号推理) 两种标准模式。
    """

    @staticmethod
    def _clean_gsm8k(text: str) -> str:
        """
        [GSM8K 标准]
        逻辑来源: lm_eval/tasks/gsm8k/utils.py
        规则: 提取 '####' 后的内容，去除逗号，保留数值。
        """
        if "####" in text:
            text = text.split("####")[-1]
        
        # 移除逗号 (例如 1,000 -> 1000) 并去除空白
        return text.replace(",", "").strip()

    @staticmethod
    def _clean_math_boxed(text: str) -> str:
        """
        [MATH 标准]
        逻辑来源: lm_eval/tasks/hendrycks_math/utils.py (简化版)
        规则: 提取 LaTeX 中的 \boxed{...} 内容。
        """
        # 非贪婪匹配最后一个 boxed
        # Harness 实际上有一个极其复杂的 remove_boxed 脚本，这里取最核心的逻辑
        matches = re.findall(r'\\boxed\{(.*?)\}', text)
        if matches:
            return matches[-1]  # 通常取最后一个
        return text

    @staticmethod
    def is_equiv_gsm8k(pred: str, gold: str) -> bool:
        """
        GSM8K 的比较逻辑是基于 [数值] 的 (Float Equivalence)。
        """
        clean_pred = HarnessMathEval._clean_gsm8k(pred)
        clean_gold = HarnessMathEval._clean_gsm8k(gold)

        try:
            # 尝试转为浮点数比较
            pred_float = float(clean_pred)
            gold_float = float(clean_gold)
            return math.isclose(pred_float, gold_float, rel_tol=1e-5)
        except ValueError:
            # 如果无法转为数字，退化为字符串比较
            return clean_pred == clean_gold

    @staticmethod
    def is_equiv_math(pred: str, gold: str) -> bool:
        """
        MATH 的比较逻辑是基于 [字符串] 的 (Exact Match)。
        注意：Harness 对 MATH 的处理其实不如 math_verify 智能，它主要依赖严格的格式。
        """
        clean_pred = HarnessMathEval._clean_math_boxed(pred)
        clean_gold = HarnessMathEval._clean_math_boxed(gold)

        # 标准化：去除所有空白字符
        norm_pred = "".join(clean_pred.split())
        norm_gold = "".join(clean_gold.split())
        
        return norm_pred == norm_gold

# ================= 演示 =================

def main():
    print("=== EleutherAI Harness 核心逻辑演示 ===\n")

    evaluator = HarnessMathEval()

    # Case 1: GSM8K 风格 (Reasoning 标准)
    # 特点：答案在 #### 之后，允许数值格式差异 (1,000 vs 1000)
    pred_gsm = "计算过程略... 答案是 #### 1,000.00"
    gold_gsm = "1000"
    
    print(f"Case 1 [GSM8K]: {pred_gsm} vs {gold_gsm}")
    passed = evaluator.is_equiv_gsm8k(pred_gsm, gold_gsm)
    print(f"Result: {'✅ Pass' if passed else '❌ Fail'}")
    print("-" * 30)

    # Case 2: MATH 风格 (LaTeX 格式)
    # 特点：提取 Boxed，忽略空格
    pred_math = "The value is \\boxed{ x + y }"
    gold_math = "\\boxed{x+y}"
    
    print(f"Case 2 [MATH]:  {pred_math} vs {gold_math}")
    passed = evaluator.is_equiv_math(pred_math, gold_math)
    print(f"Result: {'✅ Pass' if passed else '❌ Fail'}")
    print("-" * 30)

    # Case 3: 错误演示 (Harness 的局限性)
    # Harness 的默认逻辑很“死”，不懂数学等价 (1/2 != 0.5)
    # 这就是为什么之前推荐 math_verify 的原因，但这是 Harness 的真实表现。
    pred_fail = "\\boxed{0.5}"
    gold_fail = "\\boxed{1/2}"
    
    print(f"Case 3 [Limit]: {pred_fail} vs {gold_fail}")
    passed = evaluator.is_equiv_math(pred_fail, gold_fail)
    print(f"Result: {'✅ Pass' if passed else '❌ Fail (Harness默认行为)'}")

if __name__ == "__main__":
    main()