"""
rag_pipeline_vrsd.py
Requires:  pip install rank_bm25 pinecone-client numpy scikit-learn
"""

from ai71 import AI71
from ..pinecone_utils import (
    query_pinecone,              # ⬅︎ now called with include_values=True
    aggregate_pinecone_context,  # unchanged
    list_namespaces,             # unchanged
)
from ..embedding import embed_query
from ..namespace_router import choose_namespaces
from ai71.exceptions import APIError

from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# ──────────────────────────────────────────────────────────────────────────────
# Initialise API clients
client = AI71()
pc     = Pinecone(os.environ["PINECONE_API_KEY"])
# ──────────────────────────────────────────────────────────────────────────────


# ----------------- identical helpers (BM-25 prune + Cohere rerank) -----------------
def _bm25_prune(question: str, candidates: list, keep: int = 20) -> list:
    corpus  = [c["metadata"]["text"] for c in candidates]
    bm25    = BM25Okapi([doc.lower().split() for doc in corpus])
    scores  = bm25.get_scores(question.lower().split())
    top_ix  = np.argsort(scores)[::-1][:keep]
    return [candidates[i] for i in top_ix]

def _cohere_rerank(question: str, docs, top_n: int = 10):
    resp = pc.inference.rerank(
        model="cohere-rerank-3.5",
        query=question,
        documents=[d.metadata.get("text", "") for d in docs],
        top_n=top_n,
        return_documents=False,
    )
    return [docs[r.index] for r in resp.data]

# -------------------------------- VRSD helpers --------------------------------
def _sum_vecs(vecs):                     # numpy-safe vector sum
    acc = vecs[0].copy()
    for v in vecs[1:]:
        acc += v
    return acc

def VecRetSimDiv(vectors, query_vec, k=50):
    """
    Diversified selection of `k` vectors.
    - `vectors`   : iterable[list[float]]  – candidate embeddings
    - `query_vec` : list[float]            – embedding of the query
    returns (hits, idx)
    """
    vectors = np.array(vectors)
    query_vec = np.array(query_vec)
    
    if k >= len(vectors):
        return vectors, list(range(len(vectors)))

    chosen, chosen_ix = [], []
    max_ix = 0                                           # 1st winner

    for _ in range(k):
        chosen.append(vectors[max_ix])
        chosen_ix.append(max_ix)

        summary = _sum_vecs(chosen)                      # centroid so far
        max_sim, max_ix = -1, None                       # reset

        for j, v in enumerate(vectors):
            if j in chosen_ix:
                continue
            sim = cosine_similarity(
                np.array(summary + v).reshape(1, -1),
                np.array(query_vec).reshape(1, -1),
            )[0][0]
            if sim > max_sim:
                max_sim, max_ix = sim, j

    return [vectors[i] for i in chosen_ix], chosen_ix
# ──────────────────────────────────────────────────────────────────────────────

# =============================================================================
#                                 PIPELINE
# =============================================================================
def run_rag_pipeline(question: str) -> dict:
    # 1️⃣  Question-refine (unchanged)
    initial_resp = client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=128,
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

    # 2️⃣  Namespace router (unchanged)
    ns_to_use = choose_namespaces(question, list_namespaces(), votes=4, top_n=2)

    # 3️⃣  VRSD diversified vector search  (⟵ replaces old “dense semantic search 50”)
    #
    # 3a.  Fetch a *large* candidate pool WITH EMBEDDINGS
    dense_hits = query_pinecone(
        query_context,
        top_k=100,                       # larger pool gives VRSD room to diversify
        namespaces=ns_to_use,
    )["matches"]

    # 3b.  Build embeddings array + query embedding
    candidate_vecs = [m["values"] for m in dense_hits]
    query_vec      = embed_query(query_context)

    # 3c.  Diversify   →  50 hits
    _, keep_ix      = VecRetSimDiv(candidate_vecs, query_vec, k=50)
    vrsd_hits       = [dense_hits[i] for i in keep_ix]

    # 4️⃣  Sparse BM-25 prune (→ 20)  – unchanged
    sparse_hits = _bm25_prune(question, vrsd_hits, keep=20)

    # 5️⃣  Cohere-3.5 rerank (→ 10)   – unchanged
    top_hits = _cohere_rerank(question, sparse_hits, top_n=10)

    # 6️⃣  Aggregate & answer (unchanged)
    context = "\n\n".join(m["metadata"].get("text", "") for m in top_hits)
    final_prompt = f"You are a helpful assistant.\n\n### Context:\n{context}"
    final_resp = client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=8192,
        temperature=0.6,
        top_p=0.95,
        messages=[
            {"role": "system", "content": final_prompt},
            {"role": "user",   "content": question},
        ],
    )

    passages = [
        {
            "passage": m["metadata"].get("text", ""),
            "doc_IDs": [m.get("id", "").split("::")[0].replace("doc-", "")],
        }
        for m in top_hits
    ]

    return {
        "question":       question,
        "passages":       passages,
        "final_prompt":   final_prompt,
        "answer":         final_resp.choices[0].message.content,
    }