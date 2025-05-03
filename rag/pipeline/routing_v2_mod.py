# lower final temp
# add question refine token
# 500 -> 100 -> 20
# 

from ai71 import AI71
from ..pinecone_utils import (
    query_pinecone,              # unchanged
    aggregate_pinecone_context,  # unchanged
    list_namespaces,             # unchanged
)
from ..namespace_router import choose_namespaces
from ai71.exceptions import APIError

# NEW imports for the two extra pruning stages
from pinecone import Pinecone
from rank_bm25 import BM25Okapi        # pip install rank_bm25
import os
import numpy as np

client = AI71()
pc      = Pinecone(os.environ["PINECONE_API_KEY"])          # reuse your env var


def _bm25_prune(question: str, candidates: list, keep: int = 50) -> list:
    """Reduce `candidates` to BM-25 top-`keep`."""
    corpus          = [c["metadata"]["text"] for c in candidates]
    bm25   = BM25Okapi([doc.lower().split() for doc in corpus])
    scores = bm25.get_scores(question.lower().split())
    top_ix = np.argsort(scores)[::-1][:keep]
    return [candidates[i] for i in top_ix]          # ⚡ no dict-merge

def _cohere_rerank(question: str, docs, top_n: int = 10):
    resp = pc.inference.rerank(
        model="cohere-rerank-3.5",
        query=question,
        documents=[d.metadata.get("text", "") for d in docs],
        top_n=top_n,
        return_documents=False,
    )
    return [docs[r.index] for r in resp.data]       # ⚡ objects intact


def run_rag_pipeline(question: str) -> dict:          # ↓ return type unchanged
    # ---------- original Step 1  (question-refine) ----------
    initial_resp = client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=1024,
        temperature=0.2,
        top_p=0.1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": (f"Please refine the following support question for optimal "
                         f"information retrieval: {question}. Provide a step-by-step "
                         f"breakdown—minimal drafts, 5 words per step.")},
        ],
    )
    query_context = (
        f"Question: {question}\n\n### Planning to solve problem:"
        + initial_resp.choices[0].message.content
    )

    # ---------- original Step 2  (namespace routing) ----------
    ns_to_use = choose_namespaces(question, list_namespaces(), votes=4, top_n=2)

    # ---------- ★ NEW Step 3a  Dense semantic search (50) ----------
    dense_hits = query_pinecone(
        query_context,
        top_k=500,                 # was 10
        namespaces=ns_to_use,
    ).get("matches", [])

    # ---------- ★ NEW Step 3b  Sparse BM-25 prune (→ 20) ----------
    sparse_hits = _bm25_prune(question, dense_hits, keep=100)

    # ---------- ★ NEW Step 3c  Cohere-3.5 re-rank (→ 10) ----------
    top_hits    = _cohere_rerank(question, sparse_hits, top_n=20)
    
    # ---------- original Step 4  (aggregate & answer) ----------
    # aggregated_context = aggregate_pinecone_context(top_hits)
    aggregated_context = "\n\n".join(m["metadata"].get("text", "") for m in top_hits)

    final_prompt = f"You are a helpful assistant.\n\n### Context: {aggregated_context}"
    final_resp   = client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=8192,
        temperature=0.2,
        top_p=0.95,
        messages=[
            {"role": "system", "content": final_prompt},
            {"role": "user",   "content": question},
        ],
    )
    final_answer = final_resp.choices[0].message.content

    passages = [
        {
            "passage": m["metadata"].get("text", ""),
            "doc_IDs": [m.get("id", "").split("::")[0].replace("doc-", "")],
        }
        for m in top_hits
    ]

    return {
        "question": question,
        "passages": passages,
        "final_prompt": final_prompt,
        "answer": final_answer,
    }
