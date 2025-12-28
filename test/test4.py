"""
对比测试两种答案清洗方案：
- Math Verify：封装库，自动处理多种格式
- Latex2Sympy + Custom：自定义正则预处理 + SymPy 归一化
测试等价表达式组的归一化能力（千位数、分数、代数式、根号等）
"""
import re
from math_verify import parse as mv_parse
from latex2sympy2_extended import latex2sympy
from sympy import simplify, srepr

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
    # 1. 预处理 (Pre-processing) - 解决 math_verify 的痛点
    # 修复隐式乘法: 0.5x -> 0.5*x, 2\sqrt -> 2*\sqrt
    text = re.sub(r'(\d)([a-zA-Z\\])', r'\1*\2', text)
    
    # 移除 GSM8K 常见的干扰符
    text = text.replace("$", "").replace(",", "")

    try:
        # 2. 解析 (Parsing)
        sym = latex2sympy(text)
        
        # 3. 规范化 (Canonicalization)
        # 统一化简 (处理 x+1 vs 1+x)
        simple_sym = simplify(sym)
        
        # 统一数值类型 (处理 1/2 vs 0.5)
        # 策略：如果是纯数，统一转为 float 字符串并保留一定精度，方便聚类
        if simple_sym.is_number:
            # 这里的逻辑可以根据你的聚类严格程度调整
            return f"{float(simple_sym):.6g}" 
        
        # 代数式则返回其内部结构字符串 (srepr 是最稳健的指纹)
        return str(simple_sym).replace(" ", "")
        
    except Exception:
        # 兜底：如果不是数学公式，进行简单的字符串清洗
        return "STR:" + text.strip()

# ==========================================
# 🧪 测试用例集 (模拟 HMMT/GSM8K/BRUMO)
# ==========================================
# 等价表达式组：每组内的写法应该归一化为相同结果
equivalent_groups = [
    ("1,000", "1000", "$1,000"),                          # 千位数
    ("1/2", "0.5", r"\frac{1}{2}"),                       # 二分之一
    ("x + 1", "1 + x"),                                   # 代数式顺序
    ("0.5x", r"\frac{1}{2}x", r"\frac{x}{2}"),            # 半x
    (r"\sqrt{2}/2", r"1/\sqrt{2}", r"\frac{\sqrt{2}}{2}"),# 根号二分之一
]

def test_clean_func(clean_func, name: str) -> tuple[int, int]:
    """测试清洗函数能否将等价表达式归一化"""
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    passed = failed = 0
    
    for group in equivalent_groups:
        results = [clean_func(expr) for expr in group]
        ok = len(set(results)) == 1
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        
        status = "✅" if ok else "❌"
        print(f"{status} {group}")
        if not ok:
            print("   " + " | ".join(f"{e!r}→{r}" for e, r in zip(group, results)))
    
    print(f"\n通过: {passed}/{passed + failed}")
    return passed, failed

if __name__ == "__main__":
    test_clean_func(clean_with_math_verify, "Math Verify")
    test_clean_func(clean_with_custom, "Latex2Sympy + Custom")