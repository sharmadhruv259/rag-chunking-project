
from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from app.config import settings
from app.schemas import StoredDocument, UploadResponse
from app.services.document_store import DocumentStore
from app.services.pdf_service import extract_pdf


router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)


def get_document_store() -> DocumentStore:
    """Provide the configured document store.

    Args:
        None.

    Returns:
        DocumentStore: Local document storage instance.

    Raises:
        OSError: If the storage directory cannot be created.
    """
    return DocumentStore(settings.data_dir)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=201,
)
async def upload_document(
    file: UploadFile = File(...),
    store: DocumentStore = Depends(get_document_store),
) -> UploadResponse:
    """Upload, validate, extract, and store a PDF.

    Args:
        file (UploadFile): PDF uploaded by the user.
        store (DocumentStore): Document storage dependency.

    Returns:
        UploadResponse: Uploaded document metadata.

    Raises:
        HTTPException: If the file is invalid,
            oversized, or cannot be processed.
    """
    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    max_size = settings.max_pdf_size_mb * 1024 * 1024

    try:
        pdf_bytes = await file.read(max_size + 1)
    finally:
        await file.close()

    if len(pdf_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the maximum allowed size.",
        )

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file.",
        )

    try:
        document = extract_pdf(
            pdf_bytes=pdf_bytes,
            filename=filename,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    store.save(
        document=document,
        pdf_bytes=pdf_bytes,
    )

    return UploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        total_paragraphs=len(document.paragraphs),
    )


@router.get(
    "/{document_id}",
    response_model=StoredDocument,
)
def get_document(
    document_id: UUID,
    store: DocumentStore = Depends(get_document_store),
) -> StoredDocument:
    """Retrieve previously extracted PDF content.

    Args:
        document_id (UUID): Identifier of the uploaded PDF.
        store (DocumentStore): Document storage dependency.

    Returns:
        StoredDocument: Extracted document and page metadata.

    Raises:
        HTTPException: If the requested document is not found.
    """
    document = store.get(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return document
