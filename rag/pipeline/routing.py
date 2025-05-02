from ai71 import AI71
from ..pinecone_utils import (
    query_pinecone,
    batch_query_pinecone,
    aggregate_pinecone_context,
    list_namespaces,
)
from ..namespace_router import choose_namespaces
from ai71.exceptions import APIError

client = AI71()  # Ensure the AI71 client is properly authenticated/configured


def run_rag_pipeline(question: str) -> str:
    """
    Run a retrieval-augmented generation (RAG) pipeline:
      1. Refine the support question for optimal retrieval.
      2. Automatically route to the most relevant Pinecone namespaces.
      3. Query Pinecone and aggregate retrieved contexts.
      4. Generate the final answer using the aggregated context.
    """
    # Step 1: Refine the question into a better retrieval prompt
    initial_resp = client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=128,
        temperature=0.2,
        top_p=0.1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    f"Please refine the following support question for optimal information retrieval: "
                    f"{question}. Provide a step-by-step breakdown—minimal drafts, 5 words per step."
                ),
            },
        ],
    )
    query_context = (
        f"Question: {question}\n\n### Planning to solve problem:"
        + initial_resp.choices[0].message.content
    )

    print(f"Refined question for retrieval: {query_context}")

    # Step 2: Choose namespaces via AI routing
    available_ns = list_namespaces()
    ns_to_use = choose_namespaces(question, available_ns, votes=4, top_n=2)

    print(f"Chosen namespaces: {ns_to_use}")

    # Step 3: Retrieve and aggregate context
    pinecone_results = query_pinecone(query_context, top_k=10, namespaces=ns_to_use)
    aggregated_context = aggregate_pinecone_context(pinecone_results)

    # Step 4: Generate final answer with context
    final_prompt = f"You are a helpful assistant.\n\n### Context: {aggregated_context}"
    final_resp = client.chat.completions.create(
        model="tiiuae/falcon3-10b-instruct",
        max_tokens=8192,
        temperature=0.6,
        top_p=0.95,
        messages=[
            {
                "role": "system",
                "content": final_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    final_prompt += f"\n\n### Question: {question}"

    print(f"Final answer: {final_resp.choices[0].message.content}")

    # Format for final answer
    # https://huggingface.co/spaces/LiveRAG/Challenge/blob/main/Operational_Instructions/Live_Challenge_Day_and_Dry_Test_Instructions.md
    
    final_answer = final_resp.choices[0].message.content
    pinecone_results.get("matches", [])[0]['id']
    passages = []
    for m in pinecone_results.get("matches", []):
        passages.append({
            "passage": m['metadata'].get("text", []),
            "doc_IDs": [m.get("id", "").split("::")[0].replace("doc-", "")],
        })
    
    return {
        "question": question,
        "passages": passages,
        "final_prompt": final_prompt,
        "answer": final_answer
    }



## Not Implement
# def run_rag_pipeline_batch(questions: list[str]) -> list[str]:
#     """
#     Batch RAG:
#       1. Refine each question into a retrieval prompt.
#       2. Pick namespaces once (or you could route per-question).
#       3. Call batch_query_pinecone to get contexts in parallel.
#       4. Fire off the final chat for each question, using its aggregated context.
#     """
#     # 1) Refine all questions
#     refined: list[str] = []
#     for q in questions:
#         try:
#             resp = client.chat.completions.create(
#                 model="tiiuae/falcon3-10b-instruct",
#                 max_tokens=128,
#                 temperature=0.2,
#                 top_p=0.1,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {
#                         "role": "user",
#                         "content": (
#                             f"Refine this support question for retrieval: {q}. "
#                             "Give 5 one‑phrase steps."
#                         ),
#                     },
#                 ],
#             )
#             refined.append(resp.choices[0].message.content)
#         except APIError:
#             # fallback: use the original question if refine fails
#             refined.append(q)

#     # 2) Choose namespaces once (you could also do this per-question)
#     try:
#         available = list_namespaces()
#         ns_to_use = choose_namespaces(" / ".join(questions[:3]), available)
#     except Exception:
#         ns_to_use = ["default"]

#     # 3) Do one parallel Pinecone call for all refined prompts
#     pinecone_results = batch_query_pinecone(
#         refined, top_k=10, namespaces=ns_to_use, n_parallel=8
#     )

#     # 4) For each question, aggregate context and run final chat
#     answers: list[str] = []
#     for q, ctx in zip(questions, pinecone_results):
#         agg = aggregate_pinecone_context(ctx)
#         try:
#             final = client.chat.completions.create(
#                 model="tiiuae/falcon3-10b-instruct",
#                 max_tokens=8192,
#                 temperature=0.6,
#                 top_p=0.95,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": f"You are a helpful assistant.\nContext: {agg}",
#                     },
#                     {"role": "user", "content": q},
#                 ],
#             )
#             answers.append(final.choices[0].message.content)
#         except APIError:
#             answers.append("")  # or some default/fallback

#     return answers