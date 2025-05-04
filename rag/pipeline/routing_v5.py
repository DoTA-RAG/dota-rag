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


async def _get_answer_async(client, question, final_prompt):
    """Generate a single answer asynchronously."""
    resp = await client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=8192,
        temperature=0.8,
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


def _dynamic_cluster(similarity_matrix, min_threshold=0.80, max_threshold=1, step=0.05):
    """
    Dynamically cluster the answers based on cosine similarity.
    In case of a single cluster, increase the threshold until multiple clusters are found.
    """
    threshold = min_threshold
    while threshold <= max_threshold:
        clusters = []
        visited = set()
        for i in range(len(similarity_matrix)):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] >= threshold:
                    cluster.append(j)
                    visited.add(j)
            clusters.append(cluster)
        if len(clusters) > 1 or threshold == max_threshold:
            return clusters, threshold
        threshold += step
    return [list(range(len(similarity_matrix)))], max_threshold


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
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    f"Please refine the following support question for optimal "
                    f"information retrieval: {question}. Provide a step-by-step "
                    f"breakdown—minimal drafts, 5 words per step."
                ),
            },
        ],
    )
    query_context = f"Question: {question}\n\n### Planning to solve problem:" + initial_resp.choices[0].message.content

    # ---------- original Step 2  (namespace routing) ----------
    ns_to_use = choose_namespaces(question, list_namespaces(), votes=4, top_n=2)

    # ---------- ★ NEW Step 3a  Dense semantic search (50) ----------
    dense_hits = query_pinecone(
        query_context,
        top_k=100,  # was 10
        namespaces=ns_to_use,
    ).get("matches", [])

    # ---------- ★ NEW Step 3b  Sparse BM-25 prune (→ 20) ----------
    sparse_hits = _bm25_prune(question, dense_hits, keep=20)

    # ---------- ★ NEW Step 3c  Cohere-3.5 re-rank (→ 10) ----------
    top_hits = _cohere_rerank(question, sparse_hits, top_n=10)

    # ---------- original Step 4  (aggregate & answer) ----------
    # aggregated_context = aggregate_pinecone_context(top_hits)
    aggregated_context = "\n\n".join(m["metadata"].get("text", "") for m in top_hits)

    final_prompt = f"You are a helpful assistant.\n\n### Context: {aggregated_context}"

    # ---------- Step 4a (self-consistency) ----------
    try:
        num_paths = 8
        answer_list = await _generate_answers_concurrent(client, question, final_prompt, num_paths)

        # Cluster the answers
        embeddings = batch_embed_queries(answer_list)
        similarity_matrix = cosine_similarity(embeddings)
        clusters, _ = _dynamic_cluster(similarity_matrix)

        # Identify the largest cluster
        # If there's a tie in cluster sizes, use internal similarity to break it
        cluster_sizes = [len(c) for c in clusters]
        max_size = max(cluster_sizes)
        candidates = [c for c in clusters if len(c) == max_size]
        if len(candidates) == 1:
            selected_cluster = candidates[0]
        else:
            similarities = [_average_internal_similarity(c, similarity_matrix) for c in candidates]
            selected_cluster = candidates[np.argmax(similarities)]
        # Compute the centroid of the selected cluster
        cluster_embeddings = [embeddings[i] for i in selected_cluster]
        centroid = np.mean(cluster_embeddings, axis=0)
        # Find the answer closest to the centroid
        similarities = cosine_similarity([centroid], cluster_embeddings)[0]
        best_index_in_cluster = selected_cluster[np.argmax(similarities)]
        final_answer = answer_list[best_index_in_cluster]
    except Exception as e:
        print(f"[V5] Error during self-consistency: {e}.")
        time.sleep(2)
        final_answer = await _get_answer_async(client, question, final_prompt)
        print(f"Using fallback answer: {final_answer}")

    # ---------- NEW Step 5  (CoT Self-reflection) ----------
    num_retries = 3

    for _ in range(num_retries):
        self_reflection_resp = await client.chat.completions.create(
            model="tiiuae/falcon3-10b-instruct",
            max_tokens=None,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant critically evaluating and revising your own answer to a user's question.\n"
                        "Focus on two key aspects:\n"
                        "- Faithfulness: Is the answer grounded in the retrieved context? Avoid hallucinations or unsupported claims.\n"
                        "- Relevance: Does the answer directly and effectively address the user's question? Avoid vague or off-topic content.\n\n"
                        "Before evaluating the answer, you should follow these steps:\n"
                        "1. Restate the question and the model's answer.\n"
                        "2. Evaluate the faithfulness of the answer. Is it based on the retrieved text?\n"
                        "3. Evaluate the relevance of the answer. Does it directly address the question?\n"
                        "4. Provide the revised answer. The answer **must** be enclosed in LaTeX-style boxing, for example: \\boxed{}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"### Question: {question}\n\n"
                        f"### Context: {aggregated_context}\n\n"
                        f"### Model's Answer: {final_answer}\n\n"
                    ),
                },
            ],
        )
        revised_ans = extract_boxed(text=self_reflection_resp.choices[0].message.content, self_reflect=True)
        if revised_ans:
            break
    else:
        print("Failed to extract boxed answer after retries.")
        revised_ans = final_answer

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
        "answer": revised_ans,
    }
