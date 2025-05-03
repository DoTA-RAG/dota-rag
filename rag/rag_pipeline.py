from .pipeline import (routing, routing_v2, routing_v2_mod, routing_v3_vrsd, routing_v2_no_refine)

def run_rag_pipeline(question: str, mode="routing"):
    
    if mode == "routing":
        return routing.run_rag_pipeline(question)
    elif mode == "routing_v2":
        return routing_v2.run_rag_pipeline(question)
    elif mode == "routing_v2_mod":
        return routing_v2_mod.run_rag_pipeline(question)
    elif mode == "routing_v3_vrsd":
        return routing_v3_vrsd.run_rag_pipeline(question)
    elif mode == "routing_v2_no_refine":
        return routing_v2_no_refine.run_rag_pipeline(question)