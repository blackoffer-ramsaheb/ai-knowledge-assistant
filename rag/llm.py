"""
llm.py – Ollama-backed answer generation with context injection.

Builds a structured prompt from retrieved chunks and forwards it to a
locally-running Ollama instance (model: ``llama3.2``).
"""

from __future__ import annotations

import logging
from typing import List

import ollama
from langchain_core.documents import Document

from rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_MODEL: str = "llama3.2"
_DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"

_PROMPT_TEMPLATE: str = """\
You are a helpful AI Knowledge Assistant.

Context:
{context}

Question:
{question}

Instructions:
- Answer the question based ONLY on the provided context above.
- If the answer is not available in the context, say "I don't have enough information in the provided documents to answer that question."
- Be concise, accurate, and cite relevant details from the context.
"""


class OllamaLLM:
    """Generate answers using a local Ollama model.

    Parameters
    ----------
    model:
        Ollama model tag (default ``llama3.2``).
    host:
        Base URL of the Ollama server.
    retriever:
        A :pyclass:`DocumentRetriever` used to fetch context for each
        question.  If ``None`` a default retriever is created.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_OLLAMA_HOST,
        retriever: DocumentRetriever | None = None,
    ) -> None:
        self.model: str = model
        self.host: str = host
        self._retriever: DocumentRetriever = retriever or DocumentRetriever()
        self._client = ollama.Client(host=self.host)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        """Join document chunks into a single context string."""
        parts: list[str] = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(
                f"[Chunk {idx} | source: {source}, page: {page}]\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Render the final prompt from the template."""
        return _PROMPT_TEMPLATE.format(context=context, question=question)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str, k: int = 4) -> str:
        """Retrieve context and generate an answer for *question*.

        Parameters
        ----------
        question:
            The user's natural-language question.
        k:
            Number of context chunks to retrieve.

        Returns
        -------
        str
            The model's response text.
        """
        # 1. Retrieve relevant chunks.
        retrieved_docs: List[Document] = self._retriever.retrieve(question, k=k)

        if not retrieved_docs:
            logger.warning("No relevant chunks found for: %s", question)
            return (
                "I couldn't find any relevant information in the uploaded "
                "documents to answer your question."
            )

        # 2. Build prompt.
        context = self._format_context(retrieved_docs)
        prompt = self._build_prompt(question, context)
        logger.debug("Prompt length: %d characters", len(prompt))

        # 3. Call Ollama.
        try:
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
            )
            answer: str = response.get("response", "").strip()
            logger.info("Generated answer (%d chars) for: %.80s…", len(answer), question)
            return answer
        except ollama.ResponseError as exc:
            logger.error("Ollama response error: %s", exc)
            raise RuntimeError(
                f"Ollama returned an error (model={self.model}): {exc}"
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error while calling Ollama.")
            raise RuntimeError(
                "Failed to communicate with Ollama. "
                "Is the server running at %s?" % self.host
            ) from exc
