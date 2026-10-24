
"""Define request and response schemas for PDF chunking and RAG."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class Paragraph(BaseModel):
    """Represent extracted PDF text with its page number.

    Attributes:
        page_number (int): Page containing the extracted text.
        text (str): Extracted text content.

    Functions:
        model_dump: Serialize paragraph information.
    """

    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)


class StoredDocument(BaseModel):
    """Represent a processed PDF document.

    Attributes:
        document_id (UUID): Unique document identifier.
        filename (str): Original uploaded filename.
        page_count (int): Total number of PDF pages.
        paragraphs (list[Paragraph]): Extracted text blocks.

    Functions:
        model_dump_json: Serialize document metadata to JSON.
        model_validate_json: Parse stored document JSON.
    """

    document_id: UUID
    filename: str
    page_count: int = Field(ge=1)
    paragraphs: list[Paragraph]


class UploadResponse(BaseModel):
    """Represent the PDF upload API response.

    Attributes:
        document_id (UUID): Unique uploaded document identifier.
        filename (str): Original uploaded filename.
        page_count (int): Number of pages in the document.
        total_paragraphs (int): Number of extracted text blocks.

    Functions:
        model_dump: Serialize the upload response.
    """

    document_id: UUID
    filename: str
    page_count: int
    total_paragraphs: int


class ChunkingSettings(BaseModel):
    """Configure chunking and RAG generation parameters.

    Attributes:
        technique (str): Selected chunking strategy.
        chunk_size (int): Maximum tokens per generated chunk.
        chunk_overlap (int): Overlap for fixed and recursive chunks.
        top_k (int): Number of chunks retrieved during RAG.
        max_output_tokens (int): Maximum LLM response tokens.
        semantic_similarity_threshold (float): Semantic similarity limit.
        window_size (int): Sentences in a sliding window.
        window_stride (int): Sentences between consecutive windows.

    Functions:
        validate_settings: Validate technique-specific parameters.
    """

    technique: Literal[
        "fixed",
        "recursive",
        "sentence",
        "paragraph",
        "semantic",
        "sliding_window",
    ] = "paragraph"

    chunk_size: int = Field(
        default=400,
        ge=50,
        le=2000,
    )

    chunk_overlap: int = Field(
        default=0,
        ge=0,
    )

    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
    )

    max_output_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
    )

    semantic_similarity_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
    )

    window_size: int = Field(
        default=3,
        ge=1,
        le=30,
    )

    window_stride: int = Field(
        default=2,
        ge=1,
        le=30,
    )

    @model_validator(mode="after")
    def validate_settings(self) -> "ChunkingSettings":
        """Validate the selected chunking configuration.

        Args:
            None.

        Returns:
            ChunkingSettings: Validated configuration.

        Raises:
            ValueError: If the selected parameters are invalid.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than chunk size."
            )

        if (
            self.technique in {
                "sentence",
                "paragraph",
                "semantic",
                "sliding_window",
            }
            and self.chunk_overlap != 0
        ):
            raise ValueError(
                f"{self.technique} chunking requires "
                "chunk_overlap=0."
            )

        if (
            self.technique == "sliding_window"
            and self.window_stride > self.window_size
        ):
            raise ValueError(
                "Window stride cannot exceed window size."
            )

        return self


class Chunk(BaseModel):
    """Represent a generated document chunk.

    Attributes:
        chunk_index (int): Zero-based index of the chunk.
        text (str): Chunk text content.
        token_count (int): Total tokens in the chunk.
        page_numbers (list[int]): Source PDF page numbers.

    Functions:
        model_dump: Serialize chunk information.
    """

    chunk_index: int = Field(ge=0)
    text: str
    token_count: int = Field(ge=0)
    page_numbers: list[int]


class ChunkPreviewRequest(BaseModel):
    """Define a chunk preview request.

    Attributes:
        document_id (UUID): Uploaded document identifier.
        settings (ChunkingSettings): Selected chunking settings.

    Functions:
        model_dump: Serialize the preview request.
    """

    document_id: UUID
    settings: ChunkingSettings = Field(
        default_factory=ChunkingSettings
    )


class ChunkPreviewResponse(BaseModel):
    """Define the generated chunk preview response.

    Attributes:
        document_id (UUID): Uploaded document identifier.
        settings (ChunkingSettings): Applied configuration.
        total_chunks (int): Total number of generated chunks.
        chunks (list[Chunk]): Generated document chunks.

    Functions:
        model_dump: Serialize the preview response.
    """

    document_id: UUID
    settings: ChunkingSettings
    total_chunks: int = Field(ge=0)
    chunks: list[Chunk]



class RetrievedChunk(BaseModel):
    """Represent a chunk retrieved from Qdrant.

    Attributes:
        document_id (UUID | None): Original PDF identifier.
        chunk_index (int): Index within the original PDF.
        text (str): Retrieved chunk content.
        token_count (int): Number of tokens in the chunk.
        page_numbers (list[int]): Source PDF page numbers.
        filename (str): Original PDF filename.
        score (float | None): Vector similarity score.

    Functions:
        model_dump: Serialize retrieved chunk information.
    """

    document_id: UUID | None = None
    chunk_index: int = Field(ge=0)
    text: str
    token_count: int = Field(ge=0)
    page_numbers: list[int]
    filename: str
    score: float | None = None


class ChatResponse(BaseModel):
    """Represent a generated answer with optional PDF sources.

    Attributes:
        index_id (UUID | None): Index used for retrieval.
        answer (str): Generated response.
        sources (list[RetrievedChunk]): Supporting PDF chunks.

    Functions:
        model_dump: Serialize the response.
    """

    index_id: UUID | None = None
    answer: str
    sources: list[RetrievedChunk] = Field(
        default_factory=list
    )


class IndexCreateRequest(BaseModel):
    """Define a single-PDF or multiple-PDF indexing request.

    Attributes:
        document_id (UUID | None): Optional legacy PDF identifier.
        document_ids (list[UUID]): Selected PDF identifiers.
        settings (ChunkingSettings): Selected chunking configuration.

    Functions:
        validate_documents: Validate and normalize document IDs.
    """

    document_id: UUID | None = None
    document_ids: list[UUID] = Field(
        default_factory=list
    )
    settings: ChunkingSettings = Field(
        default_factory=ChunkingSettings
    )

    @model_validator(mode="after")
    def validate_documents(self) -> "IndexCreateRequest":
        """Require one or more unique document identifiers.

        Args:
            None.

        Returns:
            IndexCreateRequest: Normalized indexing request.

        Raises:
            ValueError: If there are no documents, duplicate IDs,
                or conflicting single- and multi-document fields.
        """
        ids = self.document_ids or (
            [self.document_id]
            if self.document_id is not None
            else []
        )

        if not ids:
            raise ValueError(
                "Select at least one PDF."
            )

        if len(set(ids)) != len(ids):
            raise ValueError(
                "Duplicate document IDs are not allowed."
            )

        if (
            self.document_id is not None
            and self.document_ids
            and self.document_id != ids[0]
        ):
            raise ValueError(
                "document_id must match the first "
                "document_ids entry."
            )

        # Keep single-PDF requests working.
        self.document_ids = ids
        self.document_id = ids[0]

        return self


class IndexRecord(BaseModel):
    """Represent a saved PDF indexing experiment.

    Attributes:
        index_id (UUID): Unique vector index identifier.
        document_id (UUID): First source PDF identifier.
        filename (str): First source PDF filename.
        document_ids (list[UUID]): All indexed PDF identifiers.
        filenames (list[str]): All indexed PDF filenames.
        settings (ChunkingSettings): Applied chunking settings.
        total_chunks (int): Number of indexed chunks.
        created_at (datetime): Index creation timestamp.

    Functions:
        populate_legacy_metadata: Support older index records.
        model_dump_json: Serialize index metadata.
        model_validate_json: Parse saved index metadata.
    """

    index_id: UUID
    document_id: UUID
    filename: str

    document_ids: list[UUID] = Field(
        default_factory=list
    )
    filenames: list[str] = Field(
        default_factory=list
    )

    settings: ChunkingSettings
    total_chunks: int = Field(ge=0)
    created_at: datetime

    @model_validator(mode="after")
    def populate_legacy_metadata(self) -> "IndexRecord":
        """Populate multi-PDF fields in older saved records.

        Args:
            None.

        Returns:
            IndexRecord: Record with document lists populated.

        Raises:
            None.
        """
        if not self.document_ids:
            self.document_ids = [self.document_id]

        if not self.filenames:
            self.filenames = [self.filename]

        return self


class RetrievalRequest(BaseModel):
    """Define a vector similarity retrieval request.

    Attributes:
        index_id (UUID): Target vector index identifier.
        question (str): Question used for retrieval.
        top_k (int | None): Optional retrieval count override.

    Functions:
        validate_question: Validate the user's question.
    """

    index_id: UUID
    question: str = Field(
        min_length=1,
        max_length=2000,
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Validate and normalize a retrieval question.

        Args:
            value (str): Submitted question.

        Returns:
            str: Normalized question.

        Raises:
            ValueError: If the question contains only whitespace.
        """
        value = value.strip()

        if not value:
            raise ValueError(
                "Question cannot be empty."
            )

        return value


class RetrievalResponse(BaseModel):
    """Represent the results of vector similarity search.

    Attributes:
        index_id (UUID): Index searched for relevant chunks.
        total_results (int): Number of retrieved chunks.
        chunks (list[RetrievedChunk]): Retrieved PDF chunks.

    Functions:
        model_dump: Serialize retrieval results.
    """

    index_id: UUID
    total_results: int = Field(ge=0)
    chunks: list[RetrievedChunk]


class ChatRequest(BaseModel):
    """Represent a greeting or document question.

    Attributes:
        index_id (UUID | None): Optional PDF index identifier.
        question (str): User's message.
        top_k (int | None): Retrieval count override.
        max_output_tokens (int | None): LLM output limit.

    Functions:
        validate_question: Normalize and validate the message.
    """

    index_id: UUID | None = None
    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
    )

    max_output_tokens: int | None = Field(
        default=None,
        ge=64,
        le=4096,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Remove whitespace and reject empty messages.

        Args:
            value (str): Submitted message.

        Returns:
            str: Normalized message.

        Raises:
            ValueError: If the message contains no text.
        """
        value = value.strip()

        if not value:
            raise ValueError(
                "Question cannot be empty."
            )

        return value


class IndexChunksResponse(BaseModel):
    """Represent the chunks stored in a Qdrant index.

    Attributes:
        index_id (UUID): Vector index identifier.
        total_chunks (int): Number of stored chunks.
        chunks (list[RetrievedChunk]): Indexed PDF chunks.

    Functions:
        model_dump: Serialize stored chunks.
    """

    index_id: UUID
    total_chunks: int = Field(ge=0)
    chunks: list[RetrievedChunk]
