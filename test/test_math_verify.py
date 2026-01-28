"""
测试 math_verify 库的核心功能：
- 自动提取 \\boxed{} 内容
- 数值/符号表达式的等价性判断
- 集合/多解的无序匹配
- 复杂数学表达式的归一化
"""

from math_verify import parse, verify


def check(prediction: str, gold: str, description: str):
    """一个通用的验证辅助函数，打印测试结果"""
    # 1. 提取与解析 (Parsing)
    # parse() 会返回一组可能的答案候选（candidates），通常包含提取出的数值和 SymPy 对象
    parsed_pred = parse(prediction)
    parsed_gold = parse(gold)

    # 2. 验证 (Verification)
    # verify() 会自动比较两个解析结果集合，只要有一个匹配即返回 True
    result = verify(parsed_pred, parsed_gold)

    status = "✅ 通过" if result else "❌ 失败"
    print(f"[{status}] {description}")
    # 为了保持输出整洁，这里只打印原始字符串，因为 parsed 对象可能比较复杂
    print(f"   预测: {prediction!r}")
    print(f"   标准: {gold!r}")
    print("-" * 40)


def main():
    print("=== Math Verify 核心功能极简演示 (Fixed) ===\n")

    # 场景 1: 基础提取 (Extraction)
    # 能够忽略推理过程(Cot)，只抓取 \boxed{} 中的内容
    check(
        prediction="经过计算，结果是 \\boxed{1/2}。",
        gold="0.5",
        description="自动提取 Boxed 内容并比较数值 (1/2 vs 0.5)",
    )

    # 场景 2: 格式归一化 (Normalization)
    # 处理 LaTeX 写法差异 (\frac vs /)
    check(
        prediction="\\boxed{\\frac{x}{2}}",
        gold="0.5x",
        description="符号表达式等价性 (\\frac{x}{2} vs 0.5x)",
    )

    # 场景 3: 集合与顺序无关性 (Set/Tuple handling)
    # 答案是多个解，顺序不同不应判错
    check(
        prediction="根是 \\boxed{3, -1}",
        gold="-1, 3",
        description="多解/集合的无序匹配 (3, -1 vs -1, 3)",
    )

    # 场景 4: 复杂表达式 (Complex Expressions)
    # 涉及根号、三角函数等的等价变换
    check(
        prediction="\\boxed{\\sqrt{2}/2}",
        gold="1/\\sqrt{2}",
        description="复杂数学常数等价 (sqrt(2)/2 vs 1/sqrt(2))",
    )


if __name__ == "__main__":
    main()
