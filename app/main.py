
"""Initialize the FastAPI application for the RAG project."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chunk_routes import router as chunk_router
from app.api.routes import router
from app.config import settings
from app.api.rag_routes import router as rag_router


app = FastAPI(
    title=settings.app_name,
    description="PDF-based RAG with customizable chunking.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chunk_router)
app.include_router(router)
app.include_router(rag_router)

@app.get("/")
def root() -> dict[str, str]:
    """Return the application welcome message.

    Args:
        None.

    Returns:
        dict[str, str]: Application message and status.

    Raises:
        None.
    """
    return {
        "message": "RAG Chunking API is running",
        "status": "ok",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Check whether the backend is running.

    Args:
        None.

    Returns:
        dict[str, str]: Current application health status.

    Raises:
        None.
    """
    return {"status": "healthy"}
