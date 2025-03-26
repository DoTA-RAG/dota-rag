from functools import cache
from typing import List, Literal
import torch
from transformers import AutoModel, AutoTokenizer


@cache
def has_mps() -> bool:
    return torch.backends.mps.is_available()


@cache
def has_cuda() -> bool:
    return torch.cuda.is_available()


@cache
def get_tokenizer(model_name: str = "intfloat/e5-base-v2") -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


@cache
def get_model(model_name: str = "intfloat/e5-base-v2"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if has_mps():
        return model.to("mps")
    elif has_cuda():
        return model.to("cuda")
    else:
        return model.to("cpu")


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Perform average pooling on token embeddings, ignoring padding."""
    masked_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def batch_embed_queries(
    queries: List[str],
    query_prefix: str = "query:",
    model_name: str = "intfloat/e5-base-v2",
    pooling: Literal["cls", "avg"] = "avg",
    normalize: bool = True,
) -> List[List[float]]:
    """
    Embed a batch of queries using the specified transformer model.
    """
    prefixed_queries = [f"{query_prefix} {query}" for query in queries]
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    with torch.no_grad():
        encoded = tokenizer(
            prefixed_queries, padding=True, return_tensors="pt", truncation=True
        )
        encoded = encoded.to(model.device)
        outputs = model(**encoded)
        if pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        elif pooling == "avg":
            embeddings = average_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


def embed_query(
    query: str,
    query_prefix: str = "query:",
    model_name: str = "intfloat/e5-base-v2",
    pooling: Literal["cls", "avg"] = "avg",
    normalize: bool = True,
) -> List[float]:
    """Embed a single query using the batch embedding function."""
    return batch_embed_queries([query], query_prefix, model_name, pooling, normalize)[0]
