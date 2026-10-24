"""Provide local embeddings for the RAG pipeline."""
from functools import lru_cache
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Load and cache the local text embedding model.

    Args:
        None.

    Returns:
        Embeddings: Cached LangChain embedding model.

    Raises:
        OSError: If model files cannot be loaded.
        RuntimeError: If model initialization fails.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
