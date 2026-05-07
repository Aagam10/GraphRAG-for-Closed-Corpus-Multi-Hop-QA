"""
Answer generation: takes a question + linearized subgraph context → LLM → answer.
"""

from openai import OpenAI
import config

client = config.get_llm_client()
_GENERATION_PROMPT = (config.PROMPTS_DIR / "generation.txt").read_text(encoding="utf-8")


def generate_answer(
    question: str,
    context: str,
    model: str = config.GENERATION_MODEL,
) -> str:
    """
    Generate an answer using the LLM with the provided graph context.

    Args:
        question: the user's question
        context: linearized subgraph facts
        model: LLM model to use
    Returns:
        answer string
    """
    prompt = _GENERATION_PROMPT.replace("{question}", question).replace("{context}", context)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw_answer = response.choices[0].message.content.strip()

        # Extract only the final answer after "Final answer:" marker (last occurrence wins)
        for marker in ["final answer:", "the answer is:", "answer:"]:
            idx = raw_answer.lower().rfind(marker)
            if idx != -1:
                extracted = raw_answer[idx + len(marker):].strip()
                # Take only the first line of the extracted answer
                first_line = extracted.split("\n")[0].strip().rstrip(".")
                if first_line:
                    return first_line
        # Fallback: return just the last non-empty line
        lines = [l.strip() for l in raw_answer.split("\n") if l.strip()]
        return lines[-1].rstrip(".") if lines else raw_answer
    except Exception as e:
        print(f"Generation error: {e}")
        return ""


if __name__ == "__main__":
    context = """Fact 1: [albert einstein] --(born_in)--> [ulm]
Fact 2: [ulm] --(located_in)--> [germany]
Fact 3: [albert einstein] --(developed)--> [theory of relativity]"""

    answer = generate_answer(
        "What country is the birthplace of the developer of the theory of relativity?",
        context,
    )
    print(f"Answer: {answer}")
