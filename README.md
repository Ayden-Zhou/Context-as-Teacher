# context-as-tearcher

A research project

## Quick Start

```bash
# 安装依赖
just setup

# 格式化代码
just fmt

# 运行检查
just check
```


## Cached Memory can Accelerate RL Training

|  ---| 内部 | 外部 |
| --- | --- | --- |
| 短 | 人的短期记忆/ai的context | - |
| 长 | 人的长期记忆/ai的参数记忆 | Storage/notes 适应ai的存储系统 |

### Context as Teacher: 从短期记忆 (Context) 到长期权重 (Weights) 的蒸馏

> TL;DR: 我们提出了一种全新的训练范式，将 Prompt Context 视为模型的 "Cached Memory"（缓存记忆）。与其让模型在强化学习（RL）中盲目探索，不如先利用 OPRO 在上下文窗口中构建高效的解题策略（短期记忆），然后通过 On-Policy Distillation 将这些策略“内化”到模型的权重中（长期记忆）。
> 



## 🚀 背景与动机 (Motivation)

在提升大模型（LLM）的推理能力时，传统的强化学习方法（如 PPO, GRPO）面临着 **信号稀疏** 和 **探索效率低** 的双重挑战。模型往往需要数万步的试错才能摸索出正确的推理路径。

我们的核心洞察如下：

1. **Context is Fast (上下文即时生效):** 通过优化 Prompt (In-Context Learning)，我们可以瞬间修正模型的行为。
2. **Weights are Permanent (权重固化能力):** 为了降低推理成本和延迟，我们需要将这种能力固化在模型参数中。

**我们的方案：**
引入一个由 **OPRO** 驱动的 **动态教师 (Dynamic Teacher)**。教师维护着一个包含成功推理模式的“缓存记忆”（不断进化的 Prompt）。随着训练的进行，教师通过更新记忆变得越来越聪明，从而为学生模型提供比简单的标量奖励（Scalar Rewards）更丰富、更稠密的监督信号。



## 🧠 核心理念 (Conceptual Framework)

我们将训练过程建模为一个 **记忆巩固 (Memory Consolidation)** 的过程，即从**易变的上下文 (Volatile Context)** 到**持久的参数 (Persistent Weights)** 的知识迁移。

- **Teacher (短期记忆/显式推理):**
    - **载体：** 依赖 **Context Window**。
    - **机制：** Teacher 并不通过参数更新变强，而是通过 **"Cached Memory"**（由 OPRO 动态维护的 System Prompt）来存储最新的推理策略和纠错规则。
    - **特点：** 这种记忆是**显式的 (Explicit)** 且 **更新极快**（OPRO 可以即时修改 Prompt），能迅速适应新出现的错误类型，但属于“易失性记忆”。
- **Student (长期记忆/隐式内化):**
    - **载体：** 依赖 **Model Weights**。
    - **机制：** Student 的目标不是依赖 Prompt，而是将 Teacher 在 Context 中展示的规律，通过 On-Policy Distillation **“刻录”** 进自己的神经网络参数中。
    - **特点：** 这种记忆是**隐式的 (Implicit)** 且 **持久的**。训练的最终目标是清空 Context（不再需要复杂的 Prompt），让能力完全内化为模型的直觉。

## 🛠️ 方法详解 (Methodology)

### 1. 构建缓存记忆 (Cached Memory Construction via OPRO)

不同于传统的静态蒸馏，我们的 Teacher 是**进化**的。我们利用 **OPRO (Optimization by PROmpting)** 来维护一个文本形式的“错题本与经验库”：

- 系统持续监控当前策略的“困难负样本”（Hard Negatives）。
- Meta-LLM 分析错误原因，并更新 **Cached Memory**（System Prompt），加入新的规则（例如：*“在计算概率时，优先检查独立性条件”*）。
- **效果：** Prompt 随着训练自动生长，形成一种针对模型弱点的动态课程学习（Curriculum Learning）。

### 2. 训练循环 (Reflective GKD Loop)

我们采用改进版的 **Generalized Knowledge Distillation (GKD)** 循环：

1. **探索 (Exploration):** 学生模型 自主生成回答 (On-policy)。
2. **记忆检索 (Memory Retrieval):** 教师从 **Cached Memory**中获取这道题目对应的 memory (Prompt )。
3. **反思性监督 (Reflective Supervision):** 教师在的辅助下，生成新的推理路径 。
4. **提升判断** 查看教师和学生的正确率，如果教师高，那么按照 on-poicy distillation构造loss函数，如果学生高，那么按照GRPO构造loss函数。
5. **更新 (Update):** 使用loss函数进行更新。
---

## 📊 核心特性 (Key Features)

- **零开销推理 (Zero-Overhead Inference)**
通过训练，我们将 Teacher 在 Context 中昂贵的“显式推理能力”完全内化为 Student 的“隐式权重”。推理阶段，Student 无需依赖复杂的 Prompt 即可复现 Teacher 的表现。
- **敏捷纠错 (Rapid Error Correction)**
利用 Context 的易变性优势。当 Student 犯错时，OPRO 瞬间更新 Teacher 的 Cached Memory（修改 Prompt），立刻在下一个 Batch 提供修正后的指导，避免了传统 RL 梯度更新的滞后性。
- **高带宽监督 (Dense Supervision)**
打破 RL 稀疏奖励（Scalar Reward）的瓶颈。Teacher 利用 Cached Memory 生成高质量的 Token 级分布，让 Student 通过 KL 散度全带宽地“下载”解题逻辑，大幅提升样本效率。

## Author

Zhou Yunfan <zhou.yunfan@sjtu.edu.cn>
