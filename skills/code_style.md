
# 0. 代码规范

**SYSTEM NOTE**: This document is optimized for AI Coding Agents. Content must be strictly technical, dense, and devoid of conversational filler or formatting flourishes. Maintain maximum information density.

**SYSTEM NOTE**: Focus on **algorithmic clarity** and **iteration speed**. Avoid "defensive programming". Code should look like executable pseudocode.

## 0.1 Tensors

- Shape Comments: Critical. Any operation that changes tensor dimensions must include a trailing comment.
Format: # [B, Seq, H] or # [Batch, Heads, Seq, Head_Dim]

- Vectorization: Explicit loops over tensor dimensions are strictly forbidden. Use torch.einsum, vmap, or batched operators.

- In-Place Operations: Use in-place operations (e.g., x.add_(), x += y) within optimized loops (such as samplers or optimizers) to save VRAM, unless gradient tracking (Autograd) requires otherwise.

- Device Awareness:Do not hardcode .to("cuda"). Use tensor.to(device). Create tensors directly on the target device: torch.ones(..., device=device) to avoid CPU-GPU synchronization overhead.

## 0.2 Function Signatures & Arguments

- Concise Calls: Primary data flows must use keyword arguments.

- Type Hints: Interfaces must include type annotations. Use Tensor, int, float, Batch.

## 0.3 Variable Naming (Semantic Density First)

- Avoid Excessive Abbreviation: Variable names should describe "content" rather than "type".

  - Bad: data, res, temp, obj
  - Good: student_logits, loss_mask, rollout_buffer

## 0.4 Modern Python Idioms

- Fail Fast:

  - Do not write try-except blocks to mask errors.
  - Do not write defensive asserts (unless debugging).
  - Let native PyTorch RuntimeErrors throw directly to maintain - - the "pseudocode" feel of the code logic.

## 0.5 注释与文档 (Documentation)

Google Style Docstrings：
    所有公共类 (Class) 和主要函数 (Function) 必须包含 Docstring。
    **强制要求**：在 `Args` 和 `Returns` 中必须明确标注张量的 **Shape**。
    示例：
    """
    计算缩放点积注意力 (Scaled Dot-Product Attention)。

    Args:
        q: 查询张量 (Query)。Shape: [Batch, Heads, Seq_Q, Dim]
        k: 键张量 (Key)。Shape: [Batch, Heads, Seq_K, Dim]

    Returns:
        注意力输出。Shape: [Batch, Heads, Seq_Q, Dim]
    """

代码即论文 (Code as Paper)：
    行内注释应解释“为什么” (Why) 而非“做了什么” (What)。
    对于关键算法步骤，必须在注释中引用论文中的**公式编号**或**变量符号** (如 `# Eq. 2: Softmax(QK^T / sqrt(d))`)。

## 0.6 CLI & Entry Points

- **CLI Library**: 使用 `fire` 库来完成命令行启动。禁止手动解析 `sys.argv` 或使用 `argparse`。

## 0.7 Version Control (Role: Logger)

**权限**：禁止执行 `git` 命令。仅允许向 `commit_logs.md` 追加内容。
**语言**：使用中文

**日志记录**：

- **操作**：每个逻辑变更追加一条记录。
- **格式**：`[{HH:MM}] {文件}: {操作}`。
- **约束**：严禁删除日志。

**质量门禁** (`src/dev_tools.py`)：

- **逻辑**：严格执行。零 `F` 级错误（如 F821/F401）。流水线将拒绝存在逻辑漏洞的代码。
- **风格**：忽略。流水线会自动执行 `ruff format`（自动修复导入和布局）。

**工作流**：修改代码 -> 追加日志 -> （外部执行）`just push`。
