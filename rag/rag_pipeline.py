from .pipeline import routing, routing_v2, routing_v3


def run_rag_pipeline(question: str, mode="routing"):
    if mode == "routing":
        return routing.run_rag_pipeline(question)
    elif mode == "routing_v2":
        return routing_v2.run_rag_pipeline(question)
    elif mode == "routing_v3":
        return routing_v3.run_rag_pipeline(question)
