from ai71 import AI71
from .pinecone_utils import (
    query_pinecone,
    aggregate_pinecone_context,
    # rerank_documents, # can be used to rerank documents
)

client = AI71()  # Ensure the AI71 client is properly authenticated/configured


def run_rag_pipeline(question: str) -> str:
    """
    Run a retrieval-augmented generation (RAG) pipeline:
      1. Generate a query context from the support question.
      2. Query Pinecone to retrieve relevant context.
      3. Provide the aggregated context to a second chat request.

    Returns the final response from the chat API.
    """
    # First chat call: generate a brief query context
    # Chain of Draft: Thinking Faster by Writing Less (https://arxiv.org/abs/2502.18600)
    # Optimizing Temperature for Language Models with Multi-Sample Inference (https://arxiv.org/abs/2502.05234)
    initial_response = client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=128,
        temperature=0.2,
        top_p=0.1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Please refine the following support question for optimal information retrieval: {question}. Provide a step-by-step breakdown (minimal drafts, 5 words per step).",
            },
        ],
    )
    query_context = initial_response.choices[0].message.content

    # Query Pinecone with the generated context
    pinecone_results = query_pinecone(query_context, top_k=10)
    aggregated_context = aggregate_pinecone_context(pinecone_results)

    # Second chat call: generate final answer using the aggregated context
    final_response = client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=8192,
        temperature=0.6,
        top_p=0.95,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant. Here is some relevant context: {aggregated_context}",
            },
            {"role": "user", "content": question},
        ],
    )
    return final_response.choices[0].message.content
