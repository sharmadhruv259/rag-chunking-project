
"""Implement semantic and sliding-window chunking strategies."""

import re
from functools import lru_cache
from typing import TYPE_CHECKING

from app.chunkers.base import BaseChunker
from app.chunkers.strategies import _pack_units
from app.schemas import ChunkingSettings


if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> "SentenceTransformer":
    """Load and cache the local sentence embedding model.

    Args:
        None.

    Returns:
        SentenceTransformer: Cached embedding model.

    Raises:
        OSError: If the model files cannot be loaded.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )


def split_sentences(texts: list[str]) -> list[str]:
    """Split text blocks into individual sentences.

    Args:
        texts (list[str]): Extracted PDF text blocks.

    Returns:
        list[str]: Ordered sentences extracted from the blocks.

    Raises:
        None.
    """
    sentences: list[str] = []

    for text in texts:
        parts = re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )

        sentences.extend(
            sentence.strip()
            for sentence in parts
            if sentence.strip()
        )

    return sentences


class SlidingWindowChunker(BaseChunker):
    """Generate overlapping sentence-window chunks.

    Attributes:
        encoding (tiktoken.Encoding): Inherited token encoder.

    Functions:
        _split_page: Create overlapping chunks from page sentences.
    """

    def _split_page(
        self,
        texts: list[str],
        settings: ChunkingSettings,
    ) -> list[str]:
        """Generate chunks using window size and stride.

        Args:
            texts (list[str]): Extracted text blocks from a page.
            settings (ChunkingSettings): Window and token settings.

        Returns:
            list[str]: Overlapping, token-bounded chunks.

        Raises:
            ValueError: If the window parameters are invalid.
        """
        sentences = split_sentences(texts)

        if not sentences:
            return []

        chunks: list[str] = []

        window_size = settings.window_size
        window_stride = settings.window_stride

        start = 0

        while start < len(sentences):
            end = min(
                start + window_size,
                len(sentences),
            )

            window_sentences = sentences[start:end]

            # Create chunks for this window while enforcing
            # the configured maximum token count.
            window_chunks = _pack_units(
                units=window_sentences,
                separator=" ",
                chunker=self,
                max_tokens=settings.chunk_size,
            )

            chunks.extend(window_chunks)

            # Stop when the final sentence is reached.
            if end == len(sentences):
                break

            start += window_stride

        return chunks


class SemanticChunker(BaseChunker):
    """Generate chunks based on semantic sentence similarity.

    Attributes:
        encoding (tiktoken.Encoding): Inherited token encoder.

    Functions:
        _split_page: Identify topic changes and generate chunks.
    """

    def _split_page(
        self,
        texts: list[str],
        settings: ChunkingSettings,
    ) -> list[str]:
        """Split page text at semantic topic boundaries.

        Args:
            texts (list[str]): Extracted text blocks from a page.
            settings (ChunkingSettings): Semantic chunking settings.

        Returns:
            list[str]: Semantic chunks within the token budget.

        Raises:
            OSError: If the embedding model cannot be loaded.
        """
        sentences = split_sentences(texts)

        if not sentences:
            return []

        if len(sentences) == 1:
            return _pack_units(
                units=sentences,
                separator=" ",
                chunker=self,
                max_tokens=settings.chunk_size,
            )

        model = get_embedding_model()

        embeddings = model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        chunks: list[str] = []
        current_group: list[str] = [sentences[0]]

        for index in range(1, len(sentences)):
            previous_vector = embeddings[index - 1]
            current_vector = embeddings[index]

            # With normalized embeddings, dot product
            # equals cosine similarity.
            similarity = sum(
                float(a) * float(b)
                for a, b in zip(
                    previous_vector,
                    current_vector,
                )
            )

            if (
                similarity
                < settings.semantic_similarity_threshold
            ):
                chunks.extend(
                    _pack_units(
                        units=current_group,
                        separator=" ",
                        chunker=self,
                        max_tokens=settings.chunk_size,
                    )
                )

                current_group = []

            current_group.append(sentences[index])

        if current_group:
            chunks.extend(
                _pack_units(
                    units=current_group,
                    separator=" ",
                    chunker=self,
                    max_tokens=settings.chunk_size,
                )
            )

        return chunks
