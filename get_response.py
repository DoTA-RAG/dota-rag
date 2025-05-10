"""Batch RAG runner

Reads a JSONL file with questions, runs them through the RAG pipeline concurrently
with exponential‑backoff retries, and writes a result JSONL.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

# 3rd‑party SDK
from ai71.exceptions import APIError  # type: ignore
from rag.rag_pipeline import run_rag_pipeline  # type: ignore

# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry / backoff helper
# ---------------------------------------------------------------------------


async def call_with_backoff(
    fn, *args, retries: int = 5, base: float = 2.0, **kwargs
):
    """Call *fn* with exponential backoff on rate‑limit errors (HTTP 429).

    Raises the last exception if *retries* are exhausted.
    """

    for attempt in range(1, retries + 1):
        try:
            return await fn(*args, **kwargs)
        except APIError as exc:
            msg = str(exc)
            if "429" not in msg and "Rate limit exceeded" not in msg:
                raise  # non‑rate‑limit APIError – propagate immediately

            if attempt == retries:
                raise

            wait = base ** attempt + random.random()
            log.warning("Rate‑limit hit (attempt %d/%d). Sleeping %.1fs", attempt, retries, wait)
            await asyncio.sleep(wait)
        except Exception:
            # Unexpected error – re‑raise without swallow; caller decides how to record.
            raise


# ---------------------------------------------------------------------------
# Per‑row processing
# ---------------------------------------------------------------------------


async def process_question(
    idx: int,
    question: str,
    mode: str,
    sem: asyncio.Semaphore,
) -> tuple[int, Optional[Dict[str, Any]]]:
    """Run a single question through the RAG pipeline under *sem* concurrency control."""

    async with sem:
        try:
            result = await call_with_backoff(run_rag_pipeline, question, mode=mode)
            return idx, result
        except Exception as exc:  # noqa: BLE001
            log.error("[%d] %s", idx, exc)
            return idx, None


# ---------------------------------------------------------------------------
# CLI & driver
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch‑process a JSONL question set.")
    p.add_argument(
        "--input-file",
        "-i",
        default="data/testset/testset-50q.jsonl",
        help="Path to input JSONL containing a 'question' field.",
    )
    p.add_argument(
        "--mode",
        "-m",
        default="routing_v2_no_refine",
        help="Pipeline execution mode (passed through to run_rag_pipeline).",
    )
    p.add_argument(
        "--out-dir",
        "-o",
        default="data/out",
        help="Directory to write <input>-result.jsonl.",
    )
    p.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=5,
        help="Maximum concurrent requests to the RAG pipeline.",
    )
    return p.parse_args(argv)


async def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    src_path = Path(args.input_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src_path.stem}-result.jsonl"

    # Load dataset
    df = pd.read_json(src_path, lines=True)
    if "question" not in df.columns:
        raise ValueError("Input file must contain a 'question' column")

    # Initialise result columns
    for col in ("answer", "final_prompt", "passages"):
        df[col] = None

    # Build tasks
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(process_question(idx, question, args.mode, sem))
        for idx, question in df["question"].items()
        if isinstance(question, str) and question.strip()
    ]

    # Gather with progress bar
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        idx, result = await fut
        if result:
            df.at[idx, "answer"] = result.get("answer")
            df.at[idx, "final_prompt"] = result.get("final_prompt")
            df.at[idx, "passages"] = result.get("passages")

        # Periodic checkpoint
        if idx % 25 == 0:
            df.to_json(out_path, orient="records", lines=True, force_ascii=False)

    # Final save
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    asyncio.run(main())
