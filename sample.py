"""vLLM sampling utilities."""

from vllm import LLM, SamplingParams
from prompt import sample_prompt

def sample(
    prompts: list[str],
    llm: LLM,
    num_traces: int = 512,
    top_k: int = 20,
    top_p: float = 0.95,
    temperature: float = 0.6,
    max_tokens: int = 32000,
) -> list[list[str]]:
    """Sample multiple traces per prompt using vLLM.

    Args:
        prompts: List of input prompts.
        llm: vLLM LLM instance.
        num_traces: Number of samples per prompt (n).
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        List of lists, where each inner list contains `num_traces` completions for a prompt.
    """
    params = SamplingParams(
        n=num_traces,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    outputs = llm.generate(prompts, params)

    return [[o.text for o in output.outputs] for output in outputs]

if __name__ == "__main__":
    import json, re
    from pathlib import Path
    from utils import Timer, extract_answer, equal_func
    
    extract = lambda p, t: (m.group(1).strip() if (m := re.search(p, t)) else None)
    
    model_path, dataset_path = "models/qwen3-1.7b", "data/dataset/gsm8k_test.jsonl"
    result_path = Path("data/results.json")
    
    data = [json.loads(line) for line in open(dataset_path)]
    
    # 检查是否有缓存结果
    if result_path.exists():
        print(f"📂 读取缓存: {result_path}")
        cached = json.load(open(result_path))
        outputs, stats = cached["outputs"], cached.get("stats", {})
        print(f"已缓存 {len(outputs)} 条结果")
    else:
        llm = LLM(model=model_path, enable_prefix_caching=True, enable_chunked_prefill=True, gpu_memory_utilization=0.95,
                  tensor_parallel_size=1, max_num_seqs=512, max_model_len=3384)
        prompts = [sample_prompt(model_path, d['question']) for d in data]
        
        with Timer("生成", text="{label}: {seconds:.1f}s") as t:
            raw = llm.generate(prompts, SamplingParams(n=1, top_k=20, top_p=0.95, temperature=0.6, max_tokens=3000))
        
        tokens = sum(len(o.token_ids) for r in raw for o in r.outputs)
        outputs = [[o.text for o in r.outputs] for r in raw]
        stats = {"tokens": tokens, "time": t.seconds, "speed": tokens / t.seconds}
        
        # 保存结果
        result_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"outputs": outputs, "stats": stats}, open(result_path, "w"), ensure_ascii=False)
        print(f"💾 结果已保存: {result_path}")
    
    # 评估
    errors = [i for i, (d, o) in enumerate(zip(data, outputs)) 
              if not equal_func(extract_answer(o[0]) or "", extract(r'####\s*(.*)', d['answer']) or "")]
    
    error_details = [{"idx": i, "q": data[i]['question'][:50], 
                      "pred": extract_answer(outputs[i][0]), 
                      "gold": extract(r'####\s*(.*)', data[i]['answer'])} for i in errors[:10]]
    
    print(f"错误: {len(errors)}/{len(data)} ({100*len(errors)/len(data):.1f}%)")
    if stats.get("speed"): print(f"速度: {stats['speed']:.0f} tok/s")
    print(f"前10个错误样本: {json.dumps(error_details, ensure_ascii=False, indent=2)}")