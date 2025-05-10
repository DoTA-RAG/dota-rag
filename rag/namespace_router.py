# rag/namespace_router.py

from openai import AsyncOpenAI
import re, ast
from collections import Counter
import os
import asyncio

client = AsyncOpenAI(
    base_url="https://api.ai71.ai/v1/",
    api_key=os.environ['AI71_API_KEY']
)

def extract_boxed(text: str, self_reflect: bool = False) -> list[str] | None:
    """
    Extracts a list of namespaces from a string of the form \boxed{…}.
    """
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if not m:
        return None
    content = m.group(1)
    
    # Extract the content after using self-reflection (string)
    if self_reflect:
        return content

    try:
        return ast.literal_eval(content)
    except Exception:
        return [x.strip() for x in content.split(",") if x.strip()]


async def choose_namespaces(
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

    responses = await client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=128,
        temperature=0.6,
        n=votes,
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
    
    for response in responses.choices:
        ns = extract_boxed(response.message.content)
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
