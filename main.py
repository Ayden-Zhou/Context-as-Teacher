"""
main.py
项目入口：加载模型，初始化 Memory，启动 Context-to-Weights 蒸馏。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from memory import CachedMemory
from trainer import ContextDistillationTrainer

# 假设使用一个小模型进行演示，实际项目可替换为 Mistral-7B 或 Llama-3
MODEL_NAME = "gpt2" 

def main():
    # 1. 环境与设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing project on {device}...")

    # 2. 加载模型与分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Decoder-only 模型通常需要设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Student: 待训练模型 (Weights are malleable)
    student_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    
    # Teacher: 冻结模型 (Weights are permanent, Context is volatile)
    # 注意：Teacher 和 Student 可以是同一个基座，Teacher 的优势完全来自于 Memory
    teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    # 3. 初始化 Cached Memory (System 2 Context)
    # 初始 Prompt 包含简单的思维链指引
    initial_prompt = (
        "System: You are a rigorous reasoning engine. "
        "Think step-by-step before answering. Verify your assumptions."
    )
    memory = CachedMemory(initial_prompt=initial_prompt)

    # 4. 准备极简数据 (Queries Only)
    # On-Policy 训练不需要 Ground Truth Answer
    raw_queries = [
        "What is 12 * 12?",
        "Explain the theory of relativity in one sentence.",
        "Write a Python function to reverse a list.",
        "What is the capital of France?"
    ]
    
    # 转换为 Dataset 格式 (List[Dict])
    # padding="max_length" 方便 batch 处理
    dataset = []
    for q in raw_queries:
        enc = tokenizer(q, return_tensors="pt", padding="max_length", max_length=16, truncation=True)
        # squeeze: [1, seq_len] -> [seq_len] 以适配 Trainer 的 Collator
        dataset.append({"input_ids": enc["input_ids"].squeeze(0)})

    # 5. 设置训练参数
    args = TrainingArguments(
        output_dir="./output_cached_rl",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=1e-5,
        logging_steps=1,
        remove_unused_columns=False, # 关键：防止 Trainer 移除 input_ids 以外的字段
        report_to="none",
        save_strategy="no"
    )

    # 6. 启动自定义 Trainer
    trainer = ContextDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        memory=memory,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer, # 传入 tokenizer 供 generate 使用
        temperature=1.0
    )

    print(f"🧠 Current Teacher Context: '{memory.prompt_text}'")
    print("🔥 Starting Distillation Loop...")
    
    trainer.train()

    # 7. (模拟) OPRO 更新闭环
    # 在实际训练中，这应该在一个 callback 中调用
    print("\n[Simulating OPRO Update]")
    feedback_mock = {"loss": 0.35, "bad_cases": ["..."]}
    memory.update(feedback_mock)
    print("✅ Memory updated based on feedback. Ready for next cycle.")

if __name__ == "__main__":
    main()