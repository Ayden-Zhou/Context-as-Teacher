"""
memory.py
管理动态缓存记忆 (Cached Memory)。

当前为极简版本：memory 直接使用数据集中的 solution。
后续会扩展为基于 OPRO (Optimization by PROmpting) 的动态优化接口。
"""


class CachedMemory:
    """管理 Teacher 模型的记忆/提示。

    极简版本：memory 就是数据集中的 solution。
    后续可扩展为动态优化的 Prompt。

    Attributes:
        solution (str): 当前问题的参考答案（来自数据集）。
    """

    def __init__(self, solution: str = ""):
        """初始化 CachedMemory。

        Args:
            solution: 数据集中的参考答案。
        """
        self.solution = solution

    def get_memory(self) -> str:
        """获取当前的 memory 内容。

        Returns:
            当前的 solution 字符串。
        """
        return self.solution

    def set_memory(self, solution: str) -> None:
        """设置新的 memory 内容。

        Args:
            solution: 新的参考答案。
        """
        self.solution = solution
