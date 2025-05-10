#!/usr/bin/env python
"""
Asynchronous faithfulness / relevance evaluator for Falcon-3-10B-Instruct
served on api.ai71.ai.

All CLI flags are identical to the original synchronous script, plus an
optional --max_concurrency to tune parallelism.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter  # still required elsewhere
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm   # tqdm knows how to work with asyncio

import prompt_faithfulness
import prompt_relevance

# ───────────────────────────────────────────────────────────────────────────────
#  1.  Environment + async client
# ───────────────────────────────────────────────────────────────────────────────
load_dotenv()
AI71_API_KEY = os.getenv("AI71_API_KEY")
if not AI71_API_KEY:
    raise RuntimeError("Set AI71_API_KEY in your environment or in a .env file.")

from openai import AsyncOpenAI, RateLimitError, APIError

client = AsyncOpenAI(
    base_url="https://api.ai71.ai/v1/",
    api_key=AI71_API_KEY,
)

MODEL_ID         = "tiiuae/falcon3-10b-instruct"
MAX_TOKENS       = 16_384
TEMPERATURE      = 0.6
N_COMPLETIONS    = 1
MAX_RETRIES      = 5
INITIAL_DELAY    = 10          # seconds, for 1st retry

# ───────────────────────────────────────────────────────────────────────────────
#  2.  LLM call helpers
# ───────────────────────────────────────────────────────────────────────────────
def _score_regex() -> re.Pattern[str]:
    # compiled once, reused
    return re.compile(r"\\boxed\{([-\d\.]+)")

_SCORE_RE = _score_regex()

def extract_score(text: str) -> str | None:
    """Return the value inside \boxed{ … } if present."""
    m = _SCORE_RE.search(text)
    return m.group(1) if m else None


async def evaluate_example(
    metric_module,
    prompt_arg: str,
    reference_answer: str,
    generated_answer: str,
) -> str:
    """
    Fire one judgment request and return the raw LLM response text.

      • For faithfulness → prompt_arg == passages/context
      • For relevance   → prompt_arg == user query
    """
    if metric_module == prompt_faithfulness:
        user_content = metric_module.DEFAULT_USER_TEMPLATE.format(
            context=prompt_arg,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
        )
    elif metric_module == prompt_relevance:
        user_content = metric_module.DEFAULT_USER_TEMPLATE.format(
            query=prompt_arg,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
        )
    else:
        raise ValueError("Unsupported metric module")

    messages = [
        {"role": "system", "content": metric_module.DEFAULT_SYSTEM_TEMPLATE},
        {"role": "user",   "content": user_content},
    ]

    delay = INITIAL_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model       = MODEL_ID,
                max_tokens  = MAX_TOKENS,
                temperature = TEMPERATURE,
                n           = N_COMPLETIONS,
                messages    = messages,
            )
            return resp.choices[0].message.content
        except (RateLimitError, APIError) as err:
            if hasattr(err, "status_code") and err.status_code == 429:
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
            else:
                raise
    # should never reach here
    raise RuntimeError("Unexpected retry loop exit.")


# ───────────────────────────────────────────────────────────────────────────────
#  3.  Row-level async worker
# ───────────────────────────────────────────────────────────────────────────────
async def process_row_async(
    idx: int,
    row: pd.Series,
    eval_metric: Literal["faithfulness", "relevance", "both"],
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Evaluate one JSONL row for the requested metric(s).
    """
    query            = row["question"]
    # context          = "\n\n".join(p["passage"] for p in row["passages"])
    passages_raw = row.get("passages")
    if isinstance(passages_raw, list):
        context = "\n\n".join(str(p.get("passage", "")) for p in passages_raw)
    else:
        context = ""               # no context available

    reference_answer = row["gold"]
    generated_answer = row["answer"]

    faithfulness_score = relevance_score = None
    f_correct = r_correct = 0

    async with semaphore:
        if eval_metric in {"faithfulness", "both"}:
            txt = await evaluate_example(
                prompt_faithfulness, context, reference_answer, generated_answer
            )
            faithfulness_score = extract_score(txt)

        if eval_metric in {"relevance", "both"}:
            txt = await evaluate_example(
                prompt_relevance, query, reference_answer, generated_answer
            )
            relevance_score = extract_score(txt)

    # identical “is_correct” logic
    if faithfulness_score is not None:
        try:
            f_val = float(faithfulness_score)
        except ValueError:
            f_val = 0.0
        f_correct = 1 if f_val >= 0.8 else 0.5 if f_val > 0.0 else 0

    if relevance_score is not None:
        try:
            r_val = float(relevance_score)
        except ValueError:
            r_val = 0.0
        r_correct = 1 if r_val > 1.0 else 0.5 if r_val > 0.0 else 0

    return {
        "idx": idx,
        "faithfulness": faithfulness_score,
        "relevance": relevance_score,
        "faithfulness_correct": f_correct,
        "relevance_correct": r_correct,
    }


# ───────────────────────────────────────────────────────────────────────────────
#  4.  Async main
# ───────────────────────────────────────────────────────────────────────────────
async def run_async(
    input_file: str,
    eval_metric: Literal["faithfulness", "relevance", "both"],
    max_concurrency: int,
) -> None:
    df = pd.read_json(input_file, lines=True)

    sem = asyncio.Semaphore(max_concurrency)

    tasks = [
        asyncio.create_task(process_row_async(i, row, eval_metric, sem))
        for i, row in df.iterrows()
    ]

    results = []
    cumulative_f = cumulative_r = 0

    # tqdm over the *completion* of tasks, not their creation
    async for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Evaluating",
    ):
        res = await coro
        results.append(res)

        # live tally
        if eval_metric in {"faithfulness", "both"}:
            cumulative_f += res["faithfulness_correct"]
            pct = cumulative_f / len(results) * 100
            print(
                f"[{len(results):>4}] Faithfulness {cumulative_f}/{len(results)} "
                f"({pct:.2f}%), score={res['faithfulness']}"
            )
        if eval_metric in {"relevance", "both"}:
            cumulative_r += res["relevance_correct"]
            pct = cumulative_r / len(results) * 100
            print(
                f"[{len(results):>4}] Relevance    {cumulative_r}/{len(results)} "
                f"({pct:.2f}%), score={res['relevance']}"
            )

    # save output
    out_dir = "data/out"
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(input_file).split(".")[0].replace("-result", "")
    out_path = f"{out_dir}/{stem}-eval.jsonl"

    pd.DataFrame(sorted(results, key=lambda r: r["idx"])).to_json(
        out_path, orient="records", lines=True, force_ascii=False
    )
    print("\n" + "#" * 16 + "\nEvaluation completed.")
    print(f"Saved results → {out_path}")


# ───────────────────────────────────────────────────────────────────────────────
#  5.  Entry-point
# ───────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Async LLM evaluator (AI-71).")
    parser.add_argument(
        "--eval_name",
        choices=["faithfulness", "relevance", "both"],
        default="both",
        help="Metric to run (default: both).",
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the JSONL containing the answers to grade.",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=20,
        help="Maximum number of simultaneous LLM requests (default: 20).",
    )
    args = parser.parse_args()

    asyncio.run(
        run_async(
            input_file      = args.input_file,
            eval_metric     = args.eval_name,
            max_concurrency = args.max_concurrency,
        )
    )


if __name__ == "__main__":
    main()