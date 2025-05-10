from .pipeline import routing, routing_v2, routing_v3, routing_v4, routing_v5, routing_v6_final


async def run_rag_pipeline(question: str, mode="routing"):
    if mode == "routing":
        return routing.run_rag_pipeline(question)
    elif mode == "routing_v2":
        return routing_v2.run_rag_pipeline(question)
    elif mode == "routing_v3":
        return routing_v3.run_rag_pipeline(question)
    elif mode == "routing_v4":
        return routing_v4.run_rag_pipeline(question)
    elif mode == "routing_v5":
        return await routing_v5.run_rag_pipeline(question)
    elif mode == "routing_v6_final":
        return await routing_v6_final.run_rag_pipeline(question)