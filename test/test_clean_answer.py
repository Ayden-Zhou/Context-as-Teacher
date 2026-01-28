from typing import List

from math_verify import parse, verify


def clean_answers(answers: List[str]) -> List[str]:
    """清洗答案列表，等价答案产生相同输出，方便聚类投票。"""
    groups = []  # [(parsed, canonical_str)]
    results = []

    for ans in answers:
        parsed = parse(ans)
        canonical = None

        # 找等价的已有组
        for rep_parsed, rep_str in groups:
            if parsed and rep_parsed and verify(rep_parsed, parsed):
                canonical = rep_str
                break

        # 没找到则创建新组
        if canonical is None:
            canonical = str(parsed[0]) if parsed else ans.strip()
            groups.append((parsed, canonical))

        results.append(canonical)

    return results


if __name__ == "__main__":
    # 测试用例1: 等价的数学表达式
    test_answers_1 = [
        "1/2",
        "0.5",
        "2/4",
        "3/6",
        "1/3",
        "0.333",
    ]
    print("测试用例1 - 等价分数:")
    print(f"输入: {test_answers_1}")
    cleaned_1 = clean_answers(test_answers_1)
    print(f"输出: {cleaned_1}")
    print()

    # 测试用例2: 整数答案
    test_answers_2 = [
        "42",
        "42.0",
        "84/2",
        "43",
        "42",
    ]
    print("测试用例2 - 整数答案:")
    print(f"输入: {test_answers_2}")
    cleaned_2 = clean_answers(test_answers_2)
    print(f"输出: {cleaned_2}")
    print()

    # 测试用例3: 混合答案
    test_answers_3 = [
        "\\frac{1}{2}",
        "0.5",
        "\\frac{2}{3}",
        "1/2",
        "invalid answer",
        "0.666",
    ]
    print("测试用例3 - LaTeX格式混合:")
    print(f"输入: {test_answers_3}")
    cleaned_3 = clean_answers(test_answers_3)
    print(f"输出: {cleaned_3}")
    print()

    # 测试用例4: 统计聚类效果
    test_answers_4 = ["1/2", "0.5", "2/4", "1/3", "0.5", "1/2", "1/3"]
    print("测试用例4 - 聚类统计:")
    print(f"输入: {test_answers_4}")
    cleaned_4 = clean_answers(test_answers_4)
    print(f"输出: {cleaned_4}")

    # 统计每个答案出现次数
    from collections import Counter

    counter = Counter(cleaned_4)
    print(f"聚类结果: {dict(counter)}")
    print(f"最多的答案: {counter.most_common(1)}")
