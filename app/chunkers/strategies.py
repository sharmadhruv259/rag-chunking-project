
import re
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from app.chunkers.base import BaseChunker
from app.schemas import ChunkingSettings



def _pack_units(
    units: list[str],
    separator: str,
    chunker: BaseChunker,
    max_tokens: int,
) -> list[str]:
    """Combine text units without exceeding the token budget.

    Args:
        units (list[str]): Sentences or paragraphs to combine.
        separator (str): Separator between text units.
        chunker (BaseChunker): Chunker for token counting.
        max_tokens (int): Maximum tokens per generated chunk.

    Returns:
        list[str]: Generated token-bounded chunks.

    Raises:
        ValueError: If oversized text cannot be split.
    """
    chunks: list[str] = []
    current = ""

    for unit in units:
        unit = unit.strip()

        if not unit:
            continue

        # Handle paragraphs exceeding the token limit.
        if chunker.count_tokens(unit) > max_tokens:
            if current:
                chunks.append(current)
                current = ""

            pieces = _split_oversized_text(
                text=unit,
                chunker=chunker,
                max_tokens=max_tokens,
            )

            chunks.extend(pieces)
            continue

        # Attempt to combine smaller text units.
        candidate = (
            f"{current}{separator}{unit}"
            if current
            else unit
        )

        if chunker.count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)

            current = unit

    if current:
        chunks.append(current)

    return chunks


class FixedChunker(BaseChunker):
    """Split text by token-measured fixed-size windows.

    Attributes:
        encoding (tiktoken.Encoding): Inherited tokenizer.

    Functions:
        _split_page: Split page text into fixed-size chunks.
    """

    def _split_page(
        self,
        texts: list[str],
        settings: ChunkingSettings,
    ) -> list[str]:
        """Split page text using character boundaries.

        Args:
            texts (list[str]): Extracted page text blocks.
            settings (ChunkingSettings): Chunk configuration.

        Returns:
            list[str]: Token-bounded fixed-size chunks.

        Raises:
            ValueError: If the splitter configuration is invalid.
        """
        splitter = (
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separators=[""],
                keep_separator=False,
            )
        )

        return splitter.split_text("\n\n".join(texts))


class RecursiveChunker(BaseChunker):
    """Recursively split text using natural boundaries.

    Attributes:
        encoding (tiktoken.Encoding): Inherited tokenizer.

    Functions:
        _split_page: Split text recursively into chunks.
    """

    def _split_page(
        self,
        texts: list[str],
        settings: ChunkingSettings,
    ) -> list[str]:
        """Split text using progressively smaller boundaries.

        Args:
            texts (list[str]): Extracted page text blocks.
            settings (ChunkingSettings): Chunk configuration.

        Returns:
            list[str]: Recursively generated text chunks.

        Raises:
            ValueError: If the splitter configuration is invalid.
        """
        splitter = (
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        )

        return splitter.split_text("\n\n".join(texts))


class SentenceChunker(BaseChunker):
    """Group complete sentences into token-bounded chunks.

    Attributes:
        encoding (tiktoken.Encoding): Inherited tokenizer.

    Functions:
        _split_page: Group sentences from a PDF page.
    """

    def _split_page(
        self,
        texts: list[str],
        settings: ChunkingSettings,
    ) -> list[str]:
        """Generate chunks while preserving sentence boundaries.

        Args:
            texts (list[str]): Extracted page text blocks.
            settings (ChunkingSettings): Chunk configuration.

        Returns:
            list[str]: Sentence-grouped chunks.

        Raises:
            ValueError: If the token budget is invalid.
        """
        sentences: list[str] = []

        for text in texts:
            sentences.extend(
                sentence
                for sentence in re.split(
                    r"(?<=[.!?])\s+",
                    text.strip(),
                )
                if sentence.strip()
            )

        return _pack_units(
            units=sentences,
            separator=" ",
            chunker=self,
            max_tokens=settings.chunk_size,
        )


class ParagraphChunker(BaseChunker):
    """Group paragraphs without breaking natural boundaries.

    Attributes:
        encoding (tiktoken.Encoding): Inherited tokenizer.

    Functions:
        _split_page: Group paragraphs from a PDF page.
    """

    def _split_page(
        self,
        texts: list[str],
        settings: ChunkingSettings,
    ) -> list[str]:
        """Generate paragraph-based text chunks.

        Args:
            texts (list[str]): Extracted PDF paragraphs.
            settings (ChunkingSettings): Chunk configuration.

        Returns:
            list[str]: Paragraph-grouped text chunks.

        Raises:
            ValueError: If the token budget is invalid.
        """
        return _pack_units(
            units=texts,
            separator="\n\n",
            chunker=self,
            max_tokens=settings.chunk_size,
        )


def _split_oversized_text(
    text: str,
    chunker: BaseChunker,
    max_tokens: int,
) -> list[str]:
    """Split oversized text within a strict token budget.

    Args:
        text (str): Text exceeding the token limit.
        chunker (BaseChunker): Chunker providing token counts.
        max_tokens (int): Maximum tokens allowed per chunk.

    Returns:
        list[str]: Text pieces within the token limit.

    Raises:
        ValueError: If the token budget is invalid or
            no valid text prefix fits within the limit.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive.")

    pieces: list[str] = []
    remaining = text.strip()

    while remaining:
        if chunker.count_tokens(remaining) <= max_tokens:
            pieces.append(remaining)
            break

        low = 1
        high = len(remaining)
        best = 0

        while low <= high:
            mid = (low + high) // 2
            candidate = remaining[:mid].strip()

            if (
                candidate
                and chunker.count_tokens(candidate) <= max_tokens
            ):
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        if best == 0:
            raise ValueError(
                "Unable to split text within token budget."
            )

        # Prefer a nearby whitespace boundary.
        last_space = remaining.rfind(
            " ", 0, best + 1
        )

        if last_space > best // 2:
            split_at = last_space
        else:
            split_at = best

        piece = remaining[:split_at].strip()
        remaining = remaining[split_at:].strip()

        if piece:
            pieces.append(piece)

    return pieces
