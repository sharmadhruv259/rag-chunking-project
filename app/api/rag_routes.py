
"""Expose PDF indexing, retrieval, and RAG chat endpoints."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.documents import Document

from app.api.routes import get_document_store
from app.chunkers.factory import get_chunker
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IndexChunksResponse,
    IndexCreateRequest,
    IndexRecord,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from app.services.document_store import DocumentStore
from app.services.index_store import (
    IndexStore,
    get_index_store,
)
from app.services.rag_service import (
    RAGService,
    get_rag_service,
)
from app.services.vector_service import (
    VectorService,
    get_vector_service,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["RAG"],
)


def build_retrieved_chunk(
    document: Document,
    score: float | None = None,
) -> RetrievedChunk:
    """Convert a LangChain document into an API source model.

    Args:
        document (Document): Retrieved or stored PDF chunk.
        score (float | None): Optional similarity score.

    Returns:
        RetrievedChunk: Chunk with source metadata.

    Raises:
        KeyError: If the document is missing required metadata.
    """
    metadata = document.metadata

    return RetrievedChunk(
        chunk_index=metadata["chunk_index"],
        text=document.page_content,
        token_count=metadata["token_count"],
        page_numbers=metadata["page_numbers"],
        filename=metadata["filename"],
        score=score,
    )


@router.post(
    "/indexes",
    response_model=IndexRecord,
    status_code=201,
)
def create_index(
    request: IndexCreateRequest,
    document_store: DocumentStore = Depends(
        get_document_store
    ),
    index_store: IndexStore = Depends(get_index_store),
    vector_service: VectorService = Depends(
        get_vector_service
    ),
) -> IndexRecord:
    """Chunk an uploaded PDF and create a vector index.

    Args:
        request (IndexCreateRequest): Document and chunk settings.
        document_store (DocumentStore): Uploaded PDF storage.
        index_store (IndexStore): Index metadata storage.
        vector_service (VectorService): Qdrant vector service.

    Returns:
        IndexRecord: Metadata for the newly created index.

    Raises:
        HTTPException: If the document is missing or no
            chunks can be generated.
    """
    document = document_store.get(request.document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    chunker = get_chunker(request.settings.technique)

    chunks = chunker.split(
        paragraphs=document.paragraphs,
        settings=request.settings,
    )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No chunks could be generated.",
        )

    index_id = uuid4()

    total_chunks = vector_service.create_index(
        index_id=index_id,
        document_id=document.document_id,
        filename=document.filename,
        chunks=chunks,
    )

    record = IndexRecord(
        index_id=index_id,
        document_id=document.document_id,
        filename=document.filename,
        settings=request.settings,
        total_chunks=total_chunks,
        created_at=datetime.now(timezone.utc),
    )

    try:
        index_store.save(record)
    except Exception:
        # Roll back vectors if metadata cannot be saved.
        vector_service.delete_index(index_id)
        raise

    return record


@router.get(
    "/indexes",
    response_model=list[IndexRecord],
)
def list_indexes(
    index_store: IndexStore = Depends(get_index_store),
) -> list[IndexRecord]:
    """List all saved PDF indexing experiments.

    Args:
        index_store (IndexStore): Index metadata storage.

    Returns:
        list[IndexRecord]: Saved indexing experiments.

    Raises:
        OSError: If stored metadata cannot be read.
    """
    return index_store.list_all()


@router.get(
    "/indexes/{index_id}",
    response_model=IndexRecord,
)
def get_index(
    index_id: UUID,
    index_store: IndexStore = Depends(get_index_store),
) -> IndexRecord:
    """Retrieve metadata for a saved indexing experiment.

    Args:
        index_id (UUID): Requested index identifier.
        index_store (IndexStore): Index metadata storage.

    Returns:
        IndexRecord: Saved index configuration.

    Raises:
        HTTPException: If the requested index is missing.
    """
    record = index_store.get(index_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Index not found.",
        )

    return record


@router.get(
    "/indexes/{index_id}/chunks",
    response_model=IndexChunksResponse,
)
def get_index_chunks(
    index_id: UUID,
    index_store: IndexStore = Depends(get_index_store),
    vector_service: VectorService = Depends(
        get_vector_service
    ),
) -> IndexChunksResponse:
    """List the text chunks stored in a vector index.

    Args:
        index_id (UUID): Requested index identifier.
        index_store (IndexStore): Index metadata storage.
        vector_service (VectorService): Qdrant vector service.

    Returns:
        IndexChunksResponse: Stored chunks and metadata.

    Raises:
        HTTPException: If the index is missing.
    """
    record = index_store.get(index_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Index not found.",
        )

    try:
        documents = vector_service.list_chunks(index_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    chunks = [
        build_retrieved_chunk(document)
        for document in documents
    ]

    return IndexChunksResponse(
        index_id=index_id,
        total_chunks=len(chunks),
        chunks=chunks,
    )


@router.delete(
    "/indexes/{index_id}",
    status_code=204,
)
def delete_index(
    index_id: UUID,
    index_store: IndexStore = Depends(get_index_store),
    vector_service: VectorService = Depends(
        get_vector_service
    ),
) -> None:
    """Delete a vector index and its saved metadata.

    Args:
        index_id (UUID): Index identifier to remove.
        index_store (IndexStore): Index metadata storage.
        vector_service (VectorService): Qdrant vector service.

    Returns:
        None: The index is deleted.

    Raises:
        HTTPException: If the index does not exist.
    """
    record = index_store.get(index_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Index not found.",
        )

    vector_service.delete_index(index_id)
    index_store.delete(index_id)


@router.post(
    "/retrieval/search",
    response_model=RetrievalResponse,
)
def search_chunks(
    request: RetrievalRequest,
    index_store: IndexStore = Depends(get_index_store),
    vector_service: VectorService = Depends(
        get_vector_service
    ),
) -> RetrievalResponse:
    """Retrieve relevant PDF chunks using similarity search.

    Args:
        request (RetrievalRequest): Index and user question.
        index_store (IndexStore): Index metadata storage.
        vector_service (VectorService): Qdrant vector service.

    Returns:
        RetrievalResponse: Retrieved chunks and scores.

    Raises:
        HTTPException: If the index does not exist.
    """
    record = index_store.get(request.index_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Index not found.",
        )

    top_k = (
        request.top_k
        if request.top_k is not None
        else record.settings.top_k
    )

    try:
        results = vector_service.search(
            index_id=request.index_id,
            question=request.question,
            top_k=top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    chunks = [
        build_retrieved_chunk(
            document=document,
            score=float(score),
        )
        for document, score in results
    ]

    return RetrievalResponse(
        index_id=request.index_id,
        total_results=len(chunks),
        chunks=chunks,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat_with_pdf(
    request: ChatRequest,
    index_store: IndexStore = Depends(get_index_store),
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """Answer a question using indexed PDF context and Groq.

    Args:
        request (ChatRequest): Question and generation settings.
        index_store (IndexStore): Index metadata storage.
        rag_service (RAGService): RAG question-answering service.

    Returns:
        ChatResponse: Generated answer and retrieved sources.

    Raises:
        HTTPException: If the index is missing, the Groq
            key is unconfigured, or RAG settings are invalid.
    """
    record = index_store.get(request.index_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Index not found.",
        )

    top_k = (
        request.top_k
        if request.top_k is not None
        else record.settings.top_k
    )

    max_output_tokens = (
        request.max_output_tokens
        if request.max_output_tokens is not None
        else record.settings.max_output_tokens
    )

    try:
        return rag_service.answer(
            index_id=request.index_id,
            question=request.question,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
        )

    except ValueError as exc:
        if "GROQ_API_KEY" in str(exc):
            raise HTTPException(
                status_code=503,
                detail="Groq API key is not configured.",
            ) from exc

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
