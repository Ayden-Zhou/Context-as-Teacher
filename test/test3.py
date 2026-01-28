"""
测试 latex2sympy2 库的核心功能：
- LaTeX 到 SymPy 对象的转换
- 代数化简与符号运算
- 微积分表达式解析
- 语义等价性判定（通过 simplify(a-b)==0）
"""

from latex2sympy2 import latex2sympy
from sympy import simplify


def demonstrate(latex_str: str, desc: str):
    """辅助函数：展示从 LaTeX 文本到 SymPy 对象的转化过程"""
    sym_obj = latex2sympy(latex_str)
    print(f"🔹 {desc}")
    print(f"   输入 LaTeX: {latex_str!r}")
    print(f"   解析对象:   {sym_obj}  (类型: {type(sym_obj).__name__})")
    return sym_obj


def main():
    print("=== latex2sympy2 核心功能全景演示 ===\n")

    # 1. 基础算术与数值计算 (Arithmetic)
    # ------------------------------------------------
    # 自动处理 LaTeX 分数、根号，并能转为 Python float
    expr1 = demonstrate(r"\frac{1}{2} + \sqrt{4}", "基础算术")
    print(f"   数值求值:   {float(expr1)}")  # 0.5 + 2.0 = 2.5
    print("-" * 40)

    # 2. 代数化简 (Algebraic Simplification)
    # ------------------------------------------------
    # 这是字符串匹配做不到的：识别 x + x = 2x
    expr2 = demonstrate(r"\frac{x}{2} + \frac{x}{2}", "代数自动合并")
    # 注意：latex2sympy 转换后已经是 SymPy 对象，直接调用 simplify
    simplified = simplify(expr2)
    print(f"   化简结果:   {simplified}")  # 输出: x
    print("-" * 40)

    # 3. 复杂微积分 (Calculus)
    # ------------------------------------------------
    # 解析积分、极限等高级数学符号
    # 例子：x 的积分是 x^2/2
    expr3 = demonstrate(r"\int x dx", "积分解析")
    # SymPy 对象甚至可以继续进行数学运算，例如求导 verify 回去
    derivative = expr3.diff()
    print(f"   对结果求导: {derivative}")  # 输出: x
    print("-" * 40)

    # 4. 语义等价性判定 (Semantic Equivalence)
    # ------------------------------------------------
    # 核心场景：判断 "1+x" 和 "x+1" 是否是同一个答案
    latex_a = r"1 + x"
    latex_b = r"x + 1"

    obj_a = latex2sympy(latex_a)
    obj_b = latex2sympy(latex_b)

    print(f"🔹 等价性验证 ('{latex_a}' vs '{latex_b}')")
    # 方法 A: 严格结构比较 (可能失败，取决于内部树结构)
    print(f"   直接相等(==): {obj_a == obj_b}")
    # 方法 B: 数学减法验证 (最稳健的方式: a - b = 0 ?)
    is_equal = simplify(obj_a - obj_b) == 0
    print(f"   数学等价(a-b=0): {'✅ 是' if is_equal else '❌ 否'}")


if __name__ == "__main__":
    main()
