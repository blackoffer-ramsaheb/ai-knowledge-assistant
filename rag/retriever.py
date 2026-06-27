"""
retriever.py – Semantic retrieval over the ChromaDB vector store.

Provides a simple ``retrieve(query, k)`` interface that returns the
top-*k* most relevant ``Document`` chunks for a given natural-language
query.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

from rag.embedder import EmbeddingManager

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieve semantically relevant chunks from ChromaDB.

    Parameters
    ----------
    embedding_manager:
        An :pyclass:`EmbeddingManager` instance.  If ``None`` a new one
        with default settings is created and its persisted store is loaded
        automatically.
    """

    def __init__(self, embedding_manager: EmbeddingManager | None = None) -> None:
        if embedding_manager is None:
            embedding_manager = EmbeddingManager()

        self._manager: EmbeddingManager = embedding_manager

        # Ensure the vector store is loaded.
        if self._manager.vectorstore is None:
            self._manager.load_vector_store()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Return the top-*k* chunks most relevant to *query*.

        Parameters
        ----------
        query:
            Natural-language question or search string.
        k:
            Number of results to return (default ``4``).

        Returns
        -------
        List[Document]
            Ranked list of ``Document`` objects (most relevant first).

        Raises
        ------
        ValueError
            If the vector store has not been initialised.
        """
        store = self._manager.vectorstore
        if store is None:
            raise ValueError(
                "Vector store is not initialised. "
                "Call EmbeddingManager.create_vector_store() or "
                "EmbeddingManager.load_vector_store() first."
            )

        try:
            results: List[Document] = store.similarity_search(query, k=k)
            logger.info(
                "Retrieved %d chunk(s) for query: %.80s…",
                len(results),
                query,
            )
            return results
        except Exception:
            logger.exception("Retrieval failed for query: %s", query)
            raise
