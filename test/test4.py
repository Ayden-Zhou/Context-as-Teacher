"""
对比测试两种答案清洗方案：
- Math Verify：封装库，自动处理多种格式
- Latex2Sympy + Custom：自定义正则预处理 + SymPy 归一化
测试等价表达式组的归一化能力（千位数、分数、代数式、根号等）
"""

import re
from typing import List

from latex2sympy2_extended import latex2sympy
from math_verify import parse as mv_parse
from sympy import nsimplify, simplify, srepr


# ==========================================
# 选手 A: Math Verify (封装库)
# ==========================================
def clean_with_math_verify(text: str) -> str:
    """直接使用 math_verify 的 parse 结果"""
    try:
        # mv_parse 返回一个 list，包含它认为可能的解析结果
        candidates = mv_parse(text)
        if not candidates:
            return "❌无法解析"

        # 获取第一个候选对象的内部指纹 (通常是 sympy 对象的某种表示)
        return str(candidates[0])
    except Exception as e:
        return f"❌报错: {type(e).__name__}"


# ==========================================
# 选手 B: Latex2Sympy + Custom Rules (自定义增强版)
# ==========================================
def clean_with_custom(text: str) -> str:
    """latex2sympy2 配合针对性正则清洗"""
    from sympy import nsimplify

    # 1. 预处理 (Pre-processing)
    # 修复隐式乘法: 0.5x -> 0.5*x, 2\sqrt -> 2*\sqrt
    text = re.sub(r"(\d)([a-zA-Z\\])", r"\1*\2", text)
    # 移除 GSM8K 常见的干扰符
    text = text.replace("$", "").replace(",", "")

    try:
        # 2. 解析 (Parsing)
        sym = latex2sympy(text)

        # 3. 规范化 (Canonicalization)
        # nsimplify: 将浮点数转为有理数 (0.5 -> 1/2)，统一代数表示
        sym = nsimplify(sym, rational=True)
        sym = simplify(sym)

        # 纯数值：统一转 float
        if sym.is_number:
            return f"{float(sym):.6g}"

        # 代数式：用 srepr 获取结构指纹（最稳健）
        return srepr(sym)

    except Exception:
        return "STR:" + text.strip()


# ==========================================
# 选手 C: Math Verify + SymPy 归一化
# ==========================================
def clean_answer(answer: List[str]) -> List[str]:
    """将数学答案规范化为统一表示，使等价答案产生相同输出。"""
    try:
        return str(nsimplify(simplify(mv_parse(answer)[0]), rational=True))
    except Exception:
        return answer


# ==========================================
# 🧪 测试用例集 (模拟 HMMT/GSM8K/BRUMO)
# ==========================================
# 等价表达式组：每组内的写法应该归一化为相同结果
equivalent_groups = [
    ("1,000", "1000", "$1,000"),  # 千位数
    ("1/2", "0.5", r"\frac{1}{2}"),  # 二分之一
    ("x + 1", "1 + x"),  # 代数式顺序
    ("0.5x", r"\frac{1}{2}x", r"\frac{x}{2}"),  # 半x
    (r"\sqrt{2}/2", r"1/\sqrt{2}", r"\frac{\sqrt{2}}{2}"),  # 根号二分之一
]


def test_clean_func(clean_func, name: str) -> tuple[int, int]:
    """测试清洗函数能否将等价表达式归一化"""
    print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
    passed = failed = 0

    for group in equivalent_groups:
        results = [clean_func(expr) for expr in group]
        ok = len(set(results)) == 1
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)

        status = "✅" if ok else "❌"
        print(f"{status} {group}")
        print("   " + " | ".join(f"{e!r}→{r}" for e, r in zip(group, results)))

    print(f"\n通过: {passed}/{passed + failed}")
    return passed, failed


if __name__ == "__main__":
    test_clean_func(clean_with_math_verify, "Math Verify")
    test_clean_func(clean_with_custom, "Latex2Sympy + Custom")
    test_clean_func(clean_answer, "Math Verify + SymPy 归一化")
