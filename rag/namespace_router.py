# rag/namespace_router.py

from ai71 import AI71
import re, ast
from collections import Counter

client = AI71()


def extract_boxed(text: str) -> list[str] | None:
    """
    Extracts a list of namespaces from a string of the form \boxed{…}.
    """
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if not m:
        return None
    content = m.group(1)
    try:
        return ast.literal_eval(content)
    except Exception:
        return [x.strip() for x in content.split(",") if x.strip()]


def choose_namespaces(
    question: str, available: list[str], votes: int = 4, top_n: int = 2
) -> list[str]:
    """
    Uses the AI to vote on relevant namespaces, then filters them
    against `available`. If none survive, returns ["default"].
    """
    system_prompt = (
        "You are a helpful assistant that classifies questions into relevant topic namespaces. "
        "Only select from the list provided."
    )
    choices_str = ", ".join(available)
    ballots: list[str] = []

    for _ in range(votes):
        resp = (
            client.chat.completions.create(
                model="tiiuae/falcon-180B-chat",
                max_tokens=128,
                temperature=0.6,
                top_k=95,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Q: {question}\n\n"
                            f"Available namespaces:\n{choices_str}\n\n"
                            "Step 1: Identify what the question is about.\n"
                            "Step 2: Choose only the most relevant namespaces.\n"
                            "Step 3: Return final result in \\boxed{}."
                        ),
                    },
                ],
            )
            .choices[0]
            .message.content
        )
        ns = extract_boxed(resp)
        if ns:
            ballots.extend(ns)

    if not ballots:
        return ["default"]

    #  top_n namespaces
    counts = Counter(ballots)
    selected = [ns for ns, _ in counts.most_common(top_n)]

    # **Filter** to only those actually available
    valid = [ns for ns in selected if ns in available]
    if not valid:
        return ["default"]

    return valid
