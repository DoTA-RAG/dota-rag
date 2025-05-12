from ai71 import AI71
from ..pinecone_utils import (
    query_pinecone,  # unchanged
    aggregate_pinecone_context,  # unchanged
    list_namespaces,  # unchanged
)
from ..namespace_router import choose_namespaces, extract_boxed
from ..embedding import batch_embed_queries
from ai71.exceptions import APIError

# NEW imports for the two extra pruning stages
from pinecone import Pinecone
from rank_bm25 import BM25Okapi  # pip install rank_bm25
import os
import numpy as np
import asyncio
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import time

client = AsyncOpenAI(api_key=os.environ["AI71_API_KEY"], base_url="https://api.ai71.ai/v1/")
pc = Pinecone(os.environ["PINECONE_API_KEY"])  # reuse your env var


def _bm25_prune(question: str, candidates: list, keep: int = 50) -> list:
    """Reduce `candidates` to BM-25 top-`keep`."""
    corpus = [c["metadata"]["text"] for c in candidates]
    bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
    scores = bm25.get_scores(question.lower().split())
    top_ix = np.argsort(scores)[::-1][:keep]
    return [candidates[i] for i in top_ix]  # ⚡ no dict-merge


def _cohere_rerank(question: str, docs, top_n: int = 10):
    resp = pc.inference.rerank(
        model="cohere-rerank-3.5",
        query=question,
        documents=[d.metadata.get("text", "") for d in docs],
        top_n=top_n,
        return_documents=False,
    )
    return [docs[r.index] for r in resp.data]  # ⚡ objects intact


async def _get_answer_async(client, question, final_prompt, temperature=0.8):
    """Generate a single answer asynchronously."""
    resp = await client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=8192,
        temperature=temperature,
        top_p=0.95,
        messages=[
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


async def _generate_answers_concurrent(client, question, final_prompt, num_paths: int = 8):
    """Generate multiple answers concurrently using asyncio."""
    tasks = [_get_answer_async(client, question, final_prompt) for _ in range(num_paths)]
    return await asyncio.gather(*tasks)


def _average_internal_similarity(cluster_indices, sim_matrix):
    if len(cluster_indices) <= 1:
        return 0  # Can't compute similarity for single-item clusters
    sub_matrix = sim_matrix[np.ix_(cluster_indices, cluster_indices)]
    # Sum of upper triangle (excluding diagonal) / number of pairs
    upper_triangle_sum = np.sum(np.triu(sub_matrix, k=1))
    num_pairs = len(cluster_indices) * (len(cluster_indices) - 1) / 2
    if num_pairs == 0:
        return 0
    return upper_triangle_sum / num_pairs


async def run_rag_pipeline(question: str) -> dict:  # ↓ return type unchanged
    
    # ---------- original Step 1  (question-refine) ----------
    initial_resp = await client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=128,
        temperature=0.2,
        top_p=0.1,
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant. Your task is to rewrite sentences by correcting typos and improving the wording to ensure they are written in clear, natural English. If a typo is intentional or acceptable as-is, leave it unchanged."
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    print("old q:", question)
    question = str(initial_resp.choices[0].message.content) # new question
    print("new q:", question)

    # ---------- original Step 1  (namespace routing) ----------
    ns_to_use = await choose_namespaces(question, list_namespaces(), votes=4, top_n=2)

    # ---------- ★ NEW Step 2a  Dense semantic search (50) ----------
    dense_hits = query_pinecone(
        question,
        top_k=100,  # was 10
        namespaces=ns_to_use,
    ).get("matches", [])

    # ---------- ★ NEW Step 2b  Sparse BM-25 prune (→ 20) ----------
    sparse_hits = _bm25_prune(question, dense_hits, keep=20)

    # ---------- ★ NEW Step 2c  Cohere-3.5 re-rank (→ 10) ----------
    top_hits = _cohere_rerank(question, sparse_hits, top_n=10)

    # ---------- original Step 3  (aggregate & answer) ----------
    # aggregated_context = aggregate_pinecone_context(top_hits)
    aggregated_context = "\n\n".join(m["metadata"].get("text", "") for m in top_hits)

    final_prompt = f"You are a helpful assistant.\n\n### Context: {aggregated_context}"
    final_answer = await _get_answer_async(client, question, final_prompt)
    final_prompt += f"\n\nQuestion:{question}" # add the question
    
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
