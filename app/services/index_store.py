
"""Manage local metadata for RAG indexing experiments."""

from pathlib import Path
from uuid import UUID

from app.config import settings
from app.schemas import IndexRecord


class IndexStore:
    """Manage metadata for saved document indexes.

    Attributes:
        directory (Path): Directory storing index metadata.

    Functions:
        save: Persist indexing experiment metadata.
        get: Retrieve a saved indexing experiment.
        list_all: List saved indexes.
        delete: Remove index metadata.
    """

    def __init__(self, directory: Path) -> None:
        """Initialize the index metadata store.

        Args:
            directory (Path): Directory for index metadata.

        Returns:
            None: Initializes the storage directory.

        Raises:
            OSError: If the directory cannot be created.
        """
        self.directory = directory

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(self, record: IndexRecord) -> None:
        """Persist indexing metadata as JSON.

        Args:
            record (IndexRecord): Index metadata to save.

        Returns:
            None: Writes the metadata to disk.

        Raises:
            OSError: If writing the file fails.
        """
        path = self.directory / f"{record.index_id}.json"
        temp_path = self.directory / f"{record.index_id}.tmp"

        temp_path.write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )

        temp_path.replace(path)

    def get(self, index_id: UUID) -> IndexRecord | None:
        """Retrieve metadata for an indexing experiment.

        Args:
            index_id (UUID): Index identifier to retrieve.

        Returns:
            IndexRecord | None: Saved metadata if found,
                otherwise None.

        Raises:
            OSError: If reading the file fails.
            ValueError: If the stored metadata is invalid.
        """
        path = self.directory / f"{index_id}.json"

        if not path.is_file():
            return None

        return IndexRecord.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def list_all(self) -> list[IndexRecord]:
        """List all saved indexing experiments.

        Args:
            None.

        Returns:
            list[IndexRecord]: Indexes sorted newest first.

        Raises:
            OSError: If reading metadata fails.
            ValueError: If a stored record is invalid.
        """
        records: list[IndexRecord] = []

        for path in self.directory.glob("*.json"):
            record = IndexRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )

            records.append(record)

        return sorted(
            records,
            key=lambda record: record.created_at,
            reverse=True,
        )

    def delete(self, index_id: UUID) -> None:
        """Delete metadata for an indexing experiment.

        Args:
            index_id (UUID): Index identifier to remove.

        Returns:
            None: Deletes the file if it exists.

        Raises:
            OSError: If the metadata file cannot be deleted.
        """
        path = self.directory / f"{index_id}.json"
        path.unlink(missing_ok=True)


def get_index_store() -> IndexStore:
    """Provide the configured index metadata store.

    Args:
        None.

    Returns:
        IndexStore: Local index metadata store.

    Raises:
        OSError: If the storage directory cannot be created.
    """
    return IndexStore(settings.index_dir)
