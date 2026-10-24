
"""Provide chunkers based on the selected strategy."""

from app.chunkers.advanced import (
    SemanticChunker,
    SlidingWindowChunker,
)
from app.chunkers.base import BaseChunker
from app.chunkers.strategies import (
    FixedChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
)


CHUNKER_TYPES: dict[str, type[BaseChunker]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
    "paragraph": ParagraphChunker,
    "semantic": SemanticChunker,
    "sliding_window": SlidingWindowChunker,
}


def get_chunker(technique: str) -> BaseChunker:
    """Create a chunker for the specified technique.

    Args:
        technique (str): Selected chunking technique.

    Returns:
        BaseChunker: Instance of the requested chunker.

    Raises:
        ValueError: If the chunking technique is unsupported.
    """
    chunker_class = CHUNKER_TYPES.get(technique)

    if chunker_class is None:
        raise ValueError(
            f"Unsupported chunking technique: {technique}"
        )

    return chunker_class()
