
from pathlib import Path
from uuid import UUID

from app.schemas import StoredDocument


class DocumentStore:
    """Manage uploaded PDFs and extracted document data.

    Attributes:
        directory (Path): Root directory for document storage.

    Functions:
        save: Save an uploaded PDF and extracted data.
        get: Retrieve a previously extracted document.
    """

    def __init__(self, directory: Path) -> None:
        """Initialize the document storage directory.

        Args:
            directory (Path): Root directory for stored PDFs.

        Returns:
            None: This method initializes the store.

        Raises:
            OSError: If the directory cannot be created.
        """
        self.directory = directory
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        document: StoredDocument,
        pdf_bytes: bytes,
    ) -> None:
        """Save the original PDF and extracted metadata.

        Args:
            document (StoredDocument): Extracted PDF information.
            pdf_bytes (bytes): Original PDF binary content.

        Returns:
            None: The document is saved to local storage.

        Raises:
            OSError: If writing files to disk fails.
        """
        document_dir = (
            self.directory / str(document.document_id)
        )

        document_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_path = document_dir / "source.pdf"
        metadata_path = document_dir / "document.json"

        pdf_path.write_bytes(pdf_bytes)

        metadata_path.write_text(
            document.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def get(
        self,
        document_id: UUID,
    ) -> StoredDocument | None:
        """Retrieve the extracted data for a document.

        Args:
            document_id (UUID): Unique document identifier.

        Returns:
            StoredDocument | None: Stored document if found,
                otherwise None.

        Raises:
            OSError: If the stored file cannot be read.
            ValueError: If the stored JSON is invalid.
        """
        metadata_path = (
            self.directory
            / str(document_id)
            / "document.json"
        )

        if not metadata_path.is_file():
            return None

        json_data = metadata_path.read_text(
            encoding="utf-8"
        )

        return StoredDocument.model_validate_json(
            json_data
        )
