from functools import cache
from typing import Dict, List
from multiprocessing.pool import ThreadPool
from pinecone import Pinecone
from .aws_ssm import get_ssm_secret
from .embedding import embed_query, batch_embed_queries

PINECONE_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
PINECONE_NAMESPACE = "default"


@cache
def get_pinecone_index(index_name: str = PINECONE_INDEX_NAME):
    """Initialize and return a Pinecone index using the secure API key from SSM."""
    pc = Pinecone(api_key=get_ssm_secret("/pinecone/ro_token"))
    return pc.Index(name=index_name)


def query_pinecone(
    query: str, top_k: int = 10, namespace: str = PINECONE_NAMESPACE
) -> Dict:
    """Query the Pinecone index for the given query."""
    index = get_pinecone_index()
    results = index.query(
        vector=embed_query(query),
        top_k=top_k,
        include_values=False,
        namespace=namespace,
        include_metadata=True,
    )
    return results


def batch_query_pinecone(
    queries: List[str],
    top_k: int = 10,
    namespace: str = PINECONE_NAMESPACE,
    n_parallel: int = 10,
) -> List[Dict]:
    """Batch query the Pinecone index using parallel threads."""
    index = get_pinecone_index()
    embeddings = batch_embed_queries(queries)
    with ThreadPool(n_parallel) as pool:
        results = pool.map(
            lambda vec: index.query(
                vector=vec,
                top_k=top_k,
                include_values=False,
                namespace=namespace,
                include_metadata=True,
            ),
            embeddings,
        )
    return results


def aggregate_pinecone_context(results: Dict) -> str:
    """Aggregate and return the text context from Pinecone query results (original order)."""
    matches = results.get("matches", [])
    return "\n\n".join(match["metadata"].get("text", "") for match in matches)


def rerank_documents(
    query: str, documents: List[str], top_n: int = 3, model: str = "bge-reranker-v2-m3"
) -> List[Dict]:
    """
    Use Pinecone's inference.rerank API to rerank a list of documents given a query.

    Parameters:
      - query: The query string.
      - documents: A list of document texts to rerank.
      - top_n: The number of top documents to return.
      - model: The reranking model to use.

    Returns:
      A list of dictionaries for the top_n documents, each containing reranking scores
      and document text.
    """
    # Create a new Pinecone client instance for inference rerank.
    pc = Pinecone(api_key=get_ssm_secret("/pinecone/ro_token"))
    rerank_results = pc.inference.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=True,
    )
    return rerank_results
