
"""Manage local Qdrant vector indexing and retrieval."""

from collections.abc import Sequence
from functools import lru_cache
from uuid import UUID, uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings
from app.schemas import Chunk
from app.services.embedding_service import get_embeddings


class VectorService:
    """Manage PDF chunk embeddings using Qdrant.

    Attributes:
        client (QdrantClient): Qdrant database client.
        embeddings (Embeddings): Text embedding model.

    Functions:
        collection_name: Build a collection name from an index ID.
        create_index: Index one PDF for backward compatibility.
        create_multi_index: Index chunks from multiple PDFs.
        search: Find relevant chunks for a question.
        list_chunks: Retrieve stored chunks and metadata.
        delete_index: Delete a Qdrant collection.
    """

    def __init__(
        self,
        client: QdrantClient,
        embeddings: Embeddings,
    ) -> None:
        """Initialize the vector service.

        Args:
            client (QdrantClient): Qdrant database connection.
            embeddings (Embeddings): LangChain embedding provider.

        Returns:
            None: Initializes the service.

        Raises:
            None.
        """
        self.client = client
        self.embeddings = embeddings

    @staticmethod
    def collection_name(index_id: UUID) -> str:
        """Generate a collection name for an indexing experiment.

        Args:
            index_id (UUID): Unique index identifier.

        Returns:
            str: Qdrant collection name.

        Raises:
            None.
        """
        return f"rag_{index_id.hex}"

    def create_index(
        self,
        index_id: UUID,
        document_id: UUID,
        filename: str,
        chunks: list[Chunk],
    ) -> int:
        """Index a single PDF using the shared implementation.

        Args:
            index_id (UUID): Unique index identifier.
            document_id (UUID): Source PDF identifier.
            filename (str): Original PDF filename.
            chunks (list[Chunk]): Generated document chunks.

        Returns:
            int: Number of indexed chunks.

        Raises:
            ValueError: If the index exists or chunks are empty.
            Exception: If embedding or Qdrant indexing fails.
        """
        return self.create_multi_index(
            index_id=index_id,
            document_batches=[
                (
                    document_id,
                    filename,
                    chunks,
                )
            ],
        )

    def create_multi_index(
        self,
        index_id: UUID,
        document_batches: Sequence[
            tuple[UUID, str, list[Chunk]]
        ],
    ) -> int:
        """Embed chunks from several PDFs in one Qdrant collection.

        Args:
            index_id (UUID): Unique index identifier.
            document_batches (Sequence): Tuples containing a
                PDF document ID, filename, and generated chunks.

        Returns:
            int: Total number of stored chunks across all PDFs.

        Raises:
            ValueError: If no chunks exist, the index exists,
                or the embedding model returns an empty vector.
            Exception: If embedding or Qdrant indexing fails.
        """
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "document_id": str(document_id),
                    "filename": filename,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "page_numbers": chunk.page_numbers,
                },
            )
            for document_id, filename, chunks in document_batches
            for chunk in chunks
        ]

        if not documents:
            raise ValueError(
                "Cannot create an index with no chunks."
            )

        collection = self.collection_name(index_id)

        if self.client.collection_exists(collection):
            raise ValueError(
                f"Index {index_id} already exists."
            )

        sample_embedding = self.embeddings.embed_query(
            "embedding dimension"
        )

        vector_size = len(sample_embedding)

        if vector_size == 0:
            raise ValueError(
                "Embedding model returned an empty vector."
            )

        created = False

        try:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

            created = True

            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection,
                embedding=self.embeddings,
            )

            vector_store.add_documents(
                documents=documents,
                ids=[
                    str(uuid4())
                    for _ in documents
                ],
            )

        except Exception:
            # Roll back the new collection on indexing failure.
            if created:
                self.client.delete_collection(
                    collection_name=collection
                )

            raise

        return len(documents)

    def search(
        self,
        index_id: UUID,
        question: str,
        top_k: int = 4,
    ) -> list[tuple[Document, float]]:
        """Retrieve the most relevant chunks from an index.

        Args:
            index_id (UUID): Index to search.
            question (str): User question.
            top_k (int): Maximum number of returned chunks.

        Returns:
            list[tuple[Document, float]]: Matching chunks
                and their similarity scores.

        Raises:
            ValueError: If the index does not exist,
                the question is empty, or top_k is invalid.
        """
        collection = self.collection_name(index_id)

        if not self.client.collection_exists(collection):
            raise ValueError(
                "Index does not exist."
            )

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not 1 <= top_k <= 20:
            raise ValueError(
                "top_k must be between 1 and 20."
            )

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection,
            embedding=self.embeddings,
        )

        return vector_store.similarity_search_with_score(
            query=question,
            k=top_k,
        )

    def list_chunks(
        self,
        index_id: UUID,
    ) -> list[Document]:
        """Retrieve stored chunks grouped by PDF and chunk index.

        Args:
            index_id (UUID): Target index identifier.

        Returns:
            list[Document]: Stored chunks in document order.

        Raises:
            ValueError: If the index does not exist.
        """
        collection = self.collection_name(index_id)

        if not self.client.collection_exists(collection):
            raise ValueError(
                "Index does not exist."
            )

        documents: list[Document] = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                documents.append(
                    Document(
                        page_content=payload.get(
                            "page_content",
                            "",
                        ),
                        metadata=payload.get(
                            "metadata",
                            {},
                        ),
                    )
                )

            if next_offset is None:
                break

            offset = next_offset

        return sorted(
            documents,
            key=lambda document: (
                document.metadata.get(
                    "filename",
                    "",
                ),
                document.metadata.get(
                    "document_id",
                    "",
                ),
                document.metadata.get(
                    "chunk_index",
                    0,
                ),
            ),
        )

    def delete_index(
        self,
        index_id: UUID,
    ) -> bool:
        """Delete a Qdrant collection.

        Args:
            index_id (UUID): Index identifier to delete.

        Returns:
            bool: True when deleted, False if absent.

        Raises:
            RuntimeError: If collection deletion fails.
        """
        collection = self.collection_name(index_id)

        if not self.client.collection_exists(collection):
            return False

        return self.client.delete_collection(
            collection_name=collection
        )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Initialize the persistent local Qdrant database.

    Args:
        None.

    Returns:
        QdrantClient: Reusable local database client.

    Raises:
        OSError: If the storage directory cannot be created.
        RuntimeError: If Qdrant initialization fails.
    """
    settings.qdrant_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return QdrantClient(
        path=str(settings.qdrant_path.resolve())
    )


def get_vector_service() -> VectorService:
    """Create a vector service using shared dependencies.

    Args:
        None.

    Returns:
        VectorService: Configured Qdrant vector service.

    Raises:
        RuntimeError: If embeddings or Qdrant cannot initialize.
    """
    return VectorService(
        client=get_qdrant_client(),
        embeddings=get_embeddings(),
    )

"""Manage local Qdrant vector indexing and retrieval."""

from collections.abc import Sequence
from functools import lru_cache
from uuid import UUID, uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings
from app.schemas import Chunk
from app.services.embedding_service import get_embeddings


class VectorService:
    """Manage PDF chunk embeddings using Qdrant.

    Attributes:
        client (QdrantClient): Qdrant database client.
        embeddings (Embeddings): Text embedding model.

    Functions:
        collection_name: Build a collection name from an index ID.
        create_index: Index one PDF for backward compatibility.
        create_multi_index: Index chunks from multiple PDFs.
        search: Find relevant chunks for a question.
        list_chunks: Retrieve stored chunks and metadata.
        delete_index: Delete a Qdrant collection.
    """

    def __init__(
        self,
        client: QdrantClient,
        embeddings: Embeddings,
    ) -> None:
        """Initialize the vector service.

        Args:
            client (QdrantClient): Qdrant database connection.
            embeddings (Embeddings): LangChain embedding provider.

        Returns:
            None: Initializes the service.

        Raises:
            None.
        """
        self.client = client
        self.embeddings = embeddings

    @staticmethod
    def collection_name(index_id: UUID) -> str:
        """Generate a collection name for an indexing experiment.

        Args:
            index_id (UUID): Unique index identifier.

        Returns:
            str: Qdrant collection name.

        Raises:
            None.
        """
        return f"rag_{index_id.hex}"

    def create_index(
        self,
        index_id: UUID,
        document_id: UUID,
        filename: str,
        chunks: list[Chunk],
    ) -> int:
        """Index a single PDF using the shared implementation.

        Args:
            index_id (UUID): Unique index identifier.
            document_id (UUID): Source PDF identifier.
            filename (str): Original PDF filename.
            chunks (list[Chunk]): Generated document chunks.

        Returns:
            int: Number of indexed chunks.

        Raises:
            ValueError: If the index exists or chunks are empty.
            Exception: If embedding or Qdrant indexing fails.
        """
        return self.create_multi_index(
            index_id=index_id,
            document_batches=[
                (
                    document_id,
                    filename,
                    chunks,
                )
            ],
        )

    def create_multi_index(
        self,
        index_id: UUID,
        document_batches: Sequence[
            tuple[UUID, str, list[Chunk]]
        ],
    ) -> int:
        """Embed chunks from several PDFs in one Qdrant collection.

        Args:
            index_id (UUID): Unique index identifier.
            document_batches (Sequence): Tuples containing a
                PDF document ID, filename, and generated chunks.

        Returns:
            int: Total number of stored chunks across all PDFs.

        Raises:
            ValueError: If no chunks exist, the index exists,
                or the embedding model returns an empty vector.
            Exception: If embedding or Qdrant indexing fails.
        """
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "document_id": str(document_id),
                    "filename": filename,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "page_numbers": chunk.page_numbers,
                },
            )
            for document_id, filename, chunks in document_batches
            for chunk in chunks
        ]

        if not documents:
            raise ValueError(
                "Cannot create an index with no chunks."
            )

        collection = self.collection_name(index_id)

        if self.client.collection_exists(collection):
            raise ValueError(
                f"Index {index_id} already exists."
            )

        sample_embedding = self.embeddings.embed_query(
            "embedding dimension"
        )

        vector_size = len(sample_embedding)

        if vector_size == 0:
            raise ValueError(
                "Embedding model returned an empty vector."
            )

        created = False

        try:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

            created = True

            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection,
                embedding=self.embeddings,
            )

            vector_store.add_documents(
                documents=documents,
                ids=[
                    str(uuid4())
                    for _ in documents
                ],
            )

        except Exception:
            # Roll back the new collection on indexing failure.
            if created:
                self.client.delete_collection(
                    collection_name=collection
                )

            raise

        return len(documents)

    def search(
        self,
        index_id: UUID,
        question: str,
        top_k: int = 4,
    ) -> list[tuple[Document, float]]:
        """Retrieve the most relevant chunks from an index.

        Args:
            index_id (UUID): Index to search.
            question (str): User question.
            top_k (int): Maximum number of returned chunks.

        Returns:
            list[tuple[Document, float]]: Matching chunks
                and their similarity scores.

        Raises:
            ValueError: If the index does not exist,
                the question is empty, or top_k is invalid.
        """
        collection = self.collection_name(index_id)

        if not self.client.collection_exists(collection):
            raise ValueError(
                "Index does not exist."
            )

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not 1 <= top_k <= 20:
            raise ValueError(
                "top_k must be between 1 and 20."
            )

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection,
            embedding=self.embeddings,
        )

        return vector_store.similarity_search_with_score(
            query=question,
            k=top_k,
        )

    def list_chunks(
        self,
        index_id: UUID,
    ) -> list[Document]:
        """Retrieve stored chunks grouped by PDF and chunk index.

        Args:
            index_id (UUID): Target index identifier.

        Returns:
            list[Document]: Stored chunks in document order.

        Raises:
            ValueError: If the index does not exist.
        """
        collection = self.collection_name(index_id)

        if not self.client.collection_exists(collection):
            raise ValueError(
                "Index does not exist."
            )

        documents: list[Document] = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                documents.append(
                    Document(
                        page_content=payload.get(
                            "page_content",
                            "",
                        ),
                        metadata=payload.get(
                            "metadata",
                            {},
                        ),
                    )
                )

            if next_offset is None:
                break

            offset = next_offset

        return sorted(
            documents,
            key=lambda document: (
                document.metadata.get(
                    "filename",
                    "",
                ),
                document.metadata.get(
                    "document_id",
                    "",
                ),
                document.metadata.get(
                    "chunk_index",
                    0,
                ),
            ),
        )

    def delete_index(
        self,
        index_id: UUID,
    ) -> bool:
        """Delete a Qdrant collection.

        Args:
            index_id (UUID): Index identifier to delete.

        Returns:
            bool: True when deleted, False if absent.

        Raises:
            RuntimeError: If collection deletion fails.
        """
        collection = self.collection_name(index_id)

        if not self.client.collection_exists(collection):
            return False

        return self.client.delete_collection(
            collection_name=collection
        )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Initialize the persistent local Qdrant database.

    Args:
        None.

    Returns:
        QdrantClient: Reusable local database client.

    Raises:
        OSError: If the storage directory cannot be created.
        RuntimeError: If Qdrant initialization fails.
    """
    settings.qdrant_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return QdrantClient(
        path=str(settings.qdrant_path.resolve())
    )


def get_vector_service() -> VectorService:
    """Create a vector service using shared dependencies.

    Args:
        None.

    Returns:
        VectorService: Configured Qdrant vector service.

    Raises:
        RuntimeError: If embeddings or Qdrant cannot initialize.
    """
    return VectorService(
        client=get_qdrant_client(),
        embeddings=get_embeddings(),
    )
