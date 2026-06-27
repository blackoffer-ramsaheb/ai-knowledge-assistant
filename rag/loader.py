"""
loader.py – PDF document ingestion and chunking.

Reads every PDF inside the ``uploads/`` directory, splits each into
overlapping text chunks, and returns a flat list of LangChain
``Document`` objects ready for embedding.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_UPLOADS_DIR: str = "uploads"
_CHUNK_SIZE: int = 1000
_CHUNK_OVERLAP: int = 200


class DocumentLoader:
    """Load and chunk PDF files from a local directory.

    Parameters
    ----------
    uploads_dir:
        Path to the directory that contains the PDF files.
        Defaults to ``uploads/`` relative to the current working directory.
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of characters that consecutive chunks share.
    """

    def __init__(
        self,
        uploads_dir: str = _DEFAULT_UPLOADS_DIR,
        chunk_size: int = _CHUNK_SIZE,
        chunk_overlap: int = _CHUNK_OVERLAP,
    ) -> None:
        self.uploads_dir: Path = Path(uploads_dir)
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise garbled PDF text.

        Many PDFs (especially .docx → PDF conversions) produce text where
        individual words land on separate lines separated by whitespace.
        This collapses those artefacts into clean, readable paragraphs so
        that embeddings capture the actual semantic meaning.
        """
        # Replace any run of whitespace (including \r\n) with a single space.
        text = re.sub(r'[ \t]*\r?\n[ \t]*', ' ', text)

        # Collapse multiple spaces into one.
        text = re.sub(r' {2,}', ' ', text)

        # Re-insert paragraph breaks before numbered list items.
        text = re.sub(r'([.!?])\s+(\d+\.)', r'\1\n\n\2', text)

        return text.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_documents(self) -> List[Document]:
        """Load every PDF in *uploads_dir*, split, and return chunks.

        Returns
        -------
        List[Document]
            A flat list of chunked ``Document`` objects.  Each document's
            ``metadata`` preserves the source filename and page number.

        Raises
        ------
        FileNotFoundError
            If the uploads directory does not exist.
        """
        if not self.uploads_dir.exists():
            raise FileNotFoundError(
                f"Uploads directory not found: {self.uploads_dir.resolve()}"
            )

        pdf_files: List[Path] = sorted(self.uploads_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in %s", self.uploads_dir.resolve())
            return []

        all_documents: List[Document] = []
        for pdf_path in pdf_files:
            try:
                logger.info("Loading PDF: %s", pdf_path.name)
                loader = PyPDFLoader(str(pdf_path))
                raw_pages: List[Document] = loader.load()

                # Clean the garbled text from PDF extraction.
                for page in raw_pages:
                    page.page_content = self._clean_text(page.page_content)

                chunks: List[Document] = self._splitter.split_documents(raw_pages)
                all_documents.extend(chunks)
                logger.info(
                    "  → %d pages → %d chunks", len(raw_pages), len(chunks)
                )
            except Exception:
                logger.exception("Failed to process %s – skipping.", pdf_path.name)

        logger.info(
            "Total: %d PDF(s) → %d chunks",
            len(pdf_files),
            len(all_documents),
        )
        return all_documents
