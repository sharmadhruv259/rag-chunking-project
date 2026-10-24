
from fastapi import APIRouter, Depends, HTTPException

from app.api.routes import get_document_store
from app.chunkers.factory import get_chunker
from app.schemas import (
    ChunkPreviewRequest,
    ChunkPreviewResponse,
)
from app.services.document_store import DocumentStore


router = APIRouter(
    prefix="/api/v1/chunks",
    tags=["Chunking"],
)


@router.post(
    "/preview",
    response_model=ChunkPreviewResponse,
)
def preview_chunks(
    request: ChunkPreviewRequest,
    store: DocumentStore = Depends(get_document_store),
) -> ChunkPreviewResponse:
    """Generate preview chunks for an uploaded document.

    Args:
        request (ChunkPreviewRequest): Document ID and settings.
        store (DocumentStore): Document storage dependency.

    Returns:
        ChunkPreviewResponse: Generated chunks and metadata.

    Raises:
        HTTPException: If the requested document is missing.
    """
    document = store.get(request.document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    chunker = get_chunker(
        request.settings.technique
    )

    chunks = chunker.split(
        paragraphs=document.paragraphs,
        settings=request.settings,
    )

    return ChunkPreviewResponse(
        document_id=document.document_id,
        settings=request.settings,
        total_chunks=len(chunks),
        chunks=chunks,
    )
