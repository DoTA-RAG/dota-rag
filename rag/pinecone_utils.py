import os
from functools import cache
from pinecone import Pinecone
from multiprocessing.pool import ThreadPool

from .embedding import embed_query

# ─── CONFIG ─────────────────────────────────────────────────────────────
INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME", "fineweb10bt-768-arctic-m-v2-weborganizer-full"
)


# ─── INTERNAL INDEX INITIALIZER ───────────────────────────────────────────
@cache
def _index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(INDEX_NAME)


# ─── NAMESPACE LISTING ────────────────────────────────────────────────────
def list_namespaces() -> list[str]:
    stats = _index().describe_index_stats()
    return list(stats.get("namespaces", {}).keys())


# ─── SINGLE-QUERY EMBEDDING + QUERY ───────────────────────────────────────
def query_pinecone(
    query: str, top_k: int = 5, namespaces: str | list[str] = "default"
) -> dict:
    if namespaces == "default":
        namespaces = list_namespaces()
    elif isinstance(namespaces, str):
        namespaces = [namespaces]

    return _index().query_namespaces(
        vector=embed_query(query),
        top_k=top_k,
        namespaces=namespaces,
        metric="cosine",
        include_metadata=True,
        show_progress=False,
    )


# ─── BATCH-QUERY SUPPORT ──────────────────────────────────────────────────
def batch_query_pinecone(
    queries: list[str],
    top_k: int = 5,
    namespaces: str | list[str] = "default",
    n_parallel: int = 10,
) -> list[dict]:
    """
    Query Pinecone for each query in parallel.
    Returns a list of result dicts.
    """
    if namespaces == "default":
        namespaces = list_namespaces()
    elif isinstance(namespaces, str):
        namespaces = [namespaces]

    # Precompute all embeddings
    vectors = [embed_query(q) for q in queries]

    def _worker(vec):
        return _index().query_namespaces(
            vector=vec,
            top_k=top_k,
            namespaces=namespaces,
            metric="cosine",
            include_metadata=True,
            show_progress=False,
        )

    with ThreadPool(n_parallel) as pool:
        return pool.map(_worker, vectors)


# ─── CONTEXT AGGREGATION ──────────────────────────────────────────────────
def aggregate_pinecone_context(results: dict, field: str = "text") -> str:
    """
    Concatenate the `field` from each match's metadata, separated by blank lines.
    """
    return "\n\n".join(m["metadata"].get(field, "") for m in results.get("matches", []))
