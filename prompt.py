"""Prompt builders for Deep Think with Confidence."""

from transformers import AutoTokenizer

# ============= PROMPT PREPARATION FUNCTIONS =============


def prepare_prompt(
    model_path: str,
    instruction: str,
) -> str:
    """Prepare prompt for Qwen models."""

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    messages = [{"role": "user", "content": instruction}]

    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return full_prompt

def reflect_prompt(model_path: str, failed_samples: list[str], successful_samples: list[str], problem: str) -> str:
    """Prepare prompt for reflection by building instruction and applying template."""
    fmt = lambda items: "\n\n".join(f"[Sample {i+1}]\n{s}" for i, s in enumerate(items))
    failed_text = fmt(failed_samples)
    successful_text = fmt(successful_samples) if successful_samples else "None available."

    instruction = f"""

        Given a **Problem**, some **Wrong Answers**, and (optionally) **Correct Answers**.Reflect why the wrong answers failed compared to the correct ones. Write ONE short commands within \\boxed{{}}.

        **Your Task:**
        1.  **Compare**: Look at the Wrong Answers. Why are they wrong? Did they skip a step? Did they use the wrong formula?
        2.  **Reflect**: Briefly explain the specific mistake.
        3.  **Rule**: Write several simple rules to prevent this mistake in the future. The rules must be short and tell the student exactly what to do.

        **Input Data:**

        <Problem>
        {problem}
        </Problem>

        <Wrong_Answers>
        {failed_text}
        </Wrong_Answers>

        <Correct_Answers>
        {successful_text}
        </Correct_Answers>"""

    return prepare_prompt(model_path, instruction)

def sample_with_reflection_prompt(model_path: str, question: str, reflection) -> str:
    instruction = question + "\n\n" + "To sovle this question, you should follow the following rules: " + reflection + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    return prepare_prompt(model_path, instruction)



def sample_prompt(model_path: str, question: str) -> str:
    instruction = question + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    return prepare_prompt(model_path, instruction)
