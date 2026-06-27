"""
embedder.py – HuggingFace embeddings stored in ChromaDB.

Creates (or loads) a ChromaDB vector store backed by the
``sentence-transformers/all-MiniLM-L6-v2`` model.  Duplicate documents
are detected by content hash so re-ingestion is safe.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_PERSIST_DIR: str = "chroma_db"
_DEFAULT_COLLECTION: str = "rag_knowledge_base"


def _document_id(doc: Document) -> str:
    """Return a deterministic ID based on the document's content and metadata.

    This ensures that the same chunk of text from the same source file will
    always map to the same ID, preventing duplicate embeddings.
    """
    source = doc.metadata.get("source", "")
    page = str(doc.metadata.get("page", ""))
    payload = f"{source}::{page}::{doc.page_content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EmbeddingManager:
    """Manage the lifecycle of a ChromaDB-backed vector store.

    Parameters
    ----------
    model_name:
        HuggingFace sentence-transformer model identifier.
    persist_directory:
        Local path where ChromaDB persists its data.
    collection_name:
        Name of the ChromaDB collection.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        persist_directory: str = _DEFAULT_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        self.model_name: str = model_name
        self.persist_directory: str = persist_directory
        self.collection_name: str = collection_name

        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vectorstore: Optional[Chroma] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create (or update) the vector store with *documents*.

        Documents that already exist in the store (detected via a
        deterministic content hash) are silently skipped.

        Parameters
        ----------
        documents:
            LangChain ``Document`` objects – typically produced by
            :pyclass:`rag.loader.DocumentLoader`.

        Returns
        -------
        Chroma
            The initialised vector store instance.
        """
        if not documents:
            logger.warning("No documents provided – returning empty store.")
            return self.load_vector_store()

        # Compute deterministic IDs for every incoming document.
        doc_ids: List[str] = [_document_id(d) for d in documents]

        # Load or create the store.
        persist_path = Path(self.persist_directory)

        if persist_path.exists() and any(persist_path.iterdir()):
            logger.info("Existing ChromaDB found – loading and deduplicating.")
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                persist_directory=self.persist_directory,
            )

            # Determine which IDs already exist to avoid duplicates.
            existing_data = self._vectorstore.get()
            existing_ids: set[str] = set(existing_data.get("ids", []))

            new_docs: List[Document] = []
            new_ids: List[str] = []
            for doc, did in zip(documents, doc_ids):
                if did not in existing_ids:
                    new_docs.append(doc)
                    new_ids.append(did)

            if new_docs:
                self._vectorstore.add_documents(
                    documents=new_docs,
                    ids=new_ids,
                )
                logger.info(
                    "Added %d new chunks (skipped %d duplicates).",
                    len(new_docs),
                    len(documents) - len(new_docs),
                )
            else:
                logger.info("All %d chunks already present – nothing to add.", len(documents))
        else:
            logger.info("Creating new ChromaDB vector store with %d chunks.", len(documents))
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self._embeddings,
                ids=doc_ids,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
            )

        return self._vectorstore

    def load_vector_store(self) -> Chroma:
        """Load an existing persisted ChromaDB vector store.

        Returns
        -------
        Chroma
            The loaded vector store.

        Raises
        ------
        FileNotFoundError
            If no persisted database is found at *persist_directory*.
        """
        persist_path = Path(self.persist_directory)
        if not persist_path.exists() or not any(persist_path.iterdir()):
            raise FileNotFoundError(
                f"No ChromaDB data found at {persist_path.resolve()}. "
                "Run create_vector_store() first."
            )

        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info("ChromaDB loaded from %s", persist_path.resolve())
        return self._vectorstore

    @property
    def vectorstore(self) -> Optional[Chroma]:
        """Return the currently loaded vector store (may be ``None``)."""
        return self._vectorstore
