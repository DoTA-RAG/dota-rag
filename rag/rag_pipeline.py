"""
rag_pipeline.py
────────────────
Notes:
- this dont have reasoning before the final answers.
- it still hallu and create a duplicated query calls.
"""

from __future__ import annotations

import os, json
from typing import Any, List, Tuple

from dotenv import load_dotenv
import openai

from .pinecone_utils import query_pinecone, list_namespaces

# ─── 0. ENV & CLIENT ----------------------------------------------------
load_dotenv()

api_key = os.getenv("AI71_API_KEY")
if not api_key:
    raise RuntimeError("AI71_API_KEY missing")

base_url = os.getenv("AI71_BASE_URL", "https://api.ai71.ai/v1/")
client = openai.OpenAI(api_key=api_key, base_url=base_url)

MODEL = "tiiuae/falcon3-10b-instruct"   # one model for everything

# -----------------------------------------------------------------------
# (1) Helper: parse a JSON tool call from assistant content
# -----------------------------------------------------------------------
def _find_first_json_blob(text: str) -> str | None:
    """
    Returns the first substring that looks like a complete JSON dict.
    Uses a simple brace-counter so it works with pretty-printed blocks.
    """
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def extract_call(msg_content: str | None) -> Tuple[str, dict] | None:
    """
    Extracts (tool_name, arguments) from the assistant's plain-text reply.
    Expected formats (both valid JSON):
        {"tool": "search_pinecone", "arguments": {...}}
        {"name": "search_pinecone", "arguments": {...}}
    Returns None when no valid JSON blob is found.
    """
    if not msg_content:
        return None
    blob_text = _find_first_json_blob(msg_content)
    if not blob_text:
        return None

    # Try strict JSON first; if it fails, strip trailing commas and retry
    try:
        obj = json.loads(blob_text)
    except json.JSONDecodeError:
        # Remove any trailing commas before } or ]
        import re

        fixed = re.sub(r",\s*([}\]])", r"\1", blob_text)
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError:
            return None

    # Normalise field name
    tool = obj.get("tool") or obj.get("name")
    if not isinstance(tool, str):
        return None
    args = obj.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    return tool, args


# -----------------------------------------------------------------------
# (2) Main pipeline
# -----------------------------------------------------------------------
def run_rag_pipeline(question: str, question_id: str | None = None) -> dict:
    """Runs the full RAG loop and returns the JSON result block."""
    all_ns = ",".join([f'"{ns}"' for ns in list_namespaces()])

    system_prompt = f"""\
You are **DoTA-RAG**, a retrieval-augmented assistant that answers faithfully.

TOOLS
• search_pinecone(query: str, namespaces: List[str])
    – Retrieves passages from the specified Pinecone available namespaces.
    - In query, you should rewrite each query to to get the detail for multi-hop questions.
• finish_answer(answer: str)
    – Sends the final answer back to the user.

WORKFLOW
1. Please think step by step, out loud.
2. Produce a JSON object **exactly** like this for search pinecone:
`{{"tool":"search_pinecone","arguments":{{"query":"...","namespaces":["ns1","ns2"]}}}}`
No extra keys, no trailing commas.
Use one tools at a time; selected between search_pinecone and finish_answer.
3. Repeat *search_pinecone* function until the answer and context is refined (maximum 3 times); 
   each search need to rewrite query to refined the statement. (You should not call the same query)
4. When the information is enough, please end with **finish_answer** function. Hide chain-of-thought.
Produce a JSON object **exactly** like this for finish_answer 
`{{"tool":"finish_answer","arguments":{{"answer":"..."}}}}`

RULES
- Use one tools at a time

AVAILABLE NAMESPACES
[{all_ns}]
"""

    messages: List[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    trace: List[str] = []
    passages: List[dict] = []
    final_answer = "(no answer produced)"

    for _ in range(10):  # safety loop
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            # max_tokens=1024,
            temperature=0.3,
            top_p=0.4,
        )
        msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": msg.content or ""})
        trace.append(msg.content or "")

        call = extract_call(msg.content)
        if call is None:
            # Nudge the model
            messages.append(
                {"role": "system", "content": "Please call search_pinecone or finish_answer with a valid JSON object."}
            )
            continue

        fn, args = call

        # ---- search_pinecone ------------------------------------------
        if fn == "search_pinecone":
            q = args.get("query", "")
            ns = args.get("namespaces", ["default"])
            results = query_pinecone(q, top_k=5, namespaces=ns)
            
            snippets = []
            for m in results.get("matches", []):
                txt = m["metadata"].get("text", "")
                if txt:
                    snippets.append(txt)
                    passages.append(
                        {
                            "doc_id": m["metadata"].get("doc_id", m.get("id", "")),
                            "passage": txt,
                        }
                    )

            # Echo the snippets back so the model can see them
            messages.append(
                {"role": "system", "name": f"Called function: {fn}", "content": json.dumps({"passages": snippets})}
            )
            # messages.append(
            #     {"role": "user", "content": "please think step by step about the passages. If it don't have enough infomation please rewrite the sub-query and call the `search_pinecone` function again."}
            # )
            trace.append(f"search_pinecone → {len(snippets)} passages")
            continue

        # ---- finish_answer --------------------------------------------
        if fn == "finish_answer":
            final_answer = args.get("answer", "").strip()
            messages.append(
                {"role": "system", "name": fn, "content": json.dumps({"status": "ok"})}
            )
            trace.append("finish_answer → DONE")
            break

        # ---- unknown tool ---------------------------------------------
        messages.append(
            {"role": "system", "content": f"Unknown function `{fn}`. Use only the defined tools."}
        )
        print("trace:\n", trace)

    # Deduplicate & cap supporting passages
    unique_passages = {p["passage"]: p for p in passages}
    supporting_passages = list(unique_passages.values())[:10]

    # breakpoint()

    return {
        "question_id": question_id or "",
        "question": question,
        "answer": final_answer,
        "supporting_passages": supporting_passages,
        "full_prompt": messages,
        "reasoning_trace": "\n".join(trace),
    }


__all__ = ["run_rag_pipeline"]