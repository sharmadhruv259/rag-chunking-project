
"""Manage application configuration for the RAG backend."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Manage configuration for the RAG application.

    Attributes:
        app_name (str): Name of the application.
        data_dir (Path): Directory for uploaded PDF documents.
        index_dir (Path): Directory for index metadata.
        qdrant_path (Path): Local Qdrant database directory.
        max_pdf_size_mb (int): Maximum allowed PDF size in MB.
        embedding_model (str): Sentence Transformer model name.
        groq_api_key (str): API key for Groq integration.
        groq_model (str): Groq model used for answer generation.

    Functions:
        model_dump: Serialize application configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # Application
    app_name: str = "RAG Chunking Playground"

    # Document storage
    data_dir: Path = Path("data/documents")
    index_dir: Path = Path("data/indexes")

    # Qdrant local storage
    qdrant_path: Path = Path("data/qdrant")

    # PDF upload configuration
    max_pdf_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    # Local embedding model
    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Groq configuration
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"


settings = Settings()
