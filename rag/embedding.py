from functools import cache
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

# ─── CONFIG ─────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("EMBED_MODEL", "Snowflake/snowflake-arctic-embed-m-v2.0")
MAX_LEN = int(os.getenv("EMBED_MAX_LEN", "8192"))


# ─── DEVICE SELECTION ────────────────────────────────────────────────────
@cache
def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── TOKENIZER & MODEL ───────────────────────────────────────────────────
@cache
def _tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@cache
def _model() -> AutoModel:
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # if xformers isn't installed (which it isn't on Mac M2), disable
    try:
        import xformers  # noqa: F401

        has_xformers = True
    except ImportError:
        has_xformers = False

    config.use_memory_efficient_attention = has_xformers
    config.attn_implementation = "eager"

    model = (
        AutoModel.from_pretrained(
            MODEL_NAME,
            config=config,
            add_pooling_layer=False,
            trust_remote_code=True,
        )
        .to(_device())
        .eval()
    )
    return model


# ─── POOLING ────────────────────────────────────────────────────────────
def _avg_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden.masked_fill(~mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / mask.sum(dim=1)[..., None]


# ─── BATCH EMBEDDING ────────────────────────────────────────────────────
def batch_embed_queries(
    queries: list[str],
    query_prefix: str = "query:",
    max_length: int = MAX_LEN,
    normalize: bool = True,
) -> list[list[float]]:
    """
    Embed a batch of queries using the Snowflake model.
    Returns one embedding vector per query.
    """
    # prepend prefix
    texts = [f"{query_prefix} {q}" for q in queries]
    tok = _tokenizer()(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(_device())

    with torch.no_grad():
        hidden = _model()(**tok)[0]
        pooled = _avg_pool(hidden, tok.attention_mask)
        if normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return [vec.cpu().tolist() for vec in pooled]


# ─── SINGLE QUERY EMBEDDING ─────────────────────────────────────────────
def embed_query(
    text: str,
    query_prefix: str = "query:",
    max_length: int = MAX_LEN,
    normalize: bool = True,
) -> list[float]:
    """
    Embed a single query by delegating to batch_embed_queries.
    """
    return batch_embed_queries([text], query_prefix, max_length, normalize)[0]
