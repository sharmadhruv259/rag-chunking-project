from abc import ABC, abstractmethod
import tiktoken
from app.schemas import Chunk, ChunkingSettings, Paragraph


class BaseChunker(ABC):
    """Define the common interface for text chunking.

    Attributes:
        encoding (tiktoken.Encoding): Tokenizer used for
            consistent chunk measurement.

    Functions:
        count_tokens: Count tokens in a text string.
        split: Generate chunks with source metadata.
        _split_page: Split text blocks from one PDF page.
    """

    def __init__(self) -> None:
        """Initialize the shared tokenizer.

        Args:
            None.

        Returns:
            None: Initializes the chunker instance.

        Raises:
            ValueError: If the tokenizer is unavailable.
        """
        self.encoding = tiktoken.get_encoding(
            "cl100k_base"
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text (str): Text to measure.

        Returns:
            int: Number of encoded tokens.

        Raises:
            ValueError: If the text cannot be tokenized.
        """
        return len(self.encoding.encode(text))

    def split(
        self,
        paragraphs: list[Paragraph],
        settings: ChunkingSettings,
    ) -> list[Chunk]:
        """Split document text and preserve source page numbers.

        Args:
            paragraphs (list[Paragraph]): Extracted PDF blocks.
            settings (ChunkingSettings): Chunking configuration.

        Returns:
            list[Chunk]: Generated chunks with page metadata.

        Raises:
            RuntimeError: If a chunk exceeds the token limit.
        """
        pages: dict[int, list[str]] = {}

        for paragraph in paragraphs:
            pages.setdefault(
                paragraph.page_number, []
            ).append(paragraph.text)

        chunks: list[Chunk] = []

        for page_number, texts in pages.items():
            page_chunks = self._split_page(
                texts,
                settings,
            )

            for content in page_chunks:
                content = content.strip()

                if not content:
                    continue

                token_count = self.count_tokens(content)

                if token_count > settings.chunk_size:
                    raise RuntimeError(
                        "Generated chunk exceeds token limit."
                    )

                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        text=content,
                        token_count=token_count,
                        page_numbers=[page_number],
                    )
                )

        return chunks

    @abstractmethod
    def _split_page(
        self,
        texts: list[str],
        settings: ChunkingSettings,
    ) -> list[str]:
        """Split the text blocks belonging to one page.

        Args:
            texts (list[str]): Text blocks from one PDF page.
            settings (ChunkingSettings): Chunking configuration.

        Returns:
            list[str]: Text chunks without metadata.

        Raises:
            NotImplementedError: If a subclass omits this method.
        """
        raise NotImplementedError
