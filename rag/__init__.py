"""
RAG (Retrieval-Augmented Generation) pipeline for the AI Knowledge Assistant.

Modules:
    loader    - PDF ingestion and chunking via LangChain.
    embedder  - HuggingFace embeddings stored in ChromaDB.
    retriever - Semantic search over the vector store.
    llm       - Ollama-backed answer generation with context injection.
    graph     - Lightweight GraphRAG using NetworkX for enriched retrieval.
"""

from rag.loader import DocumentLoader
from rag.embedder import EmbeddingManager
from rag.retriever import DocumentRetriever
from rag.llm import OllamaLLM
from rag.graph import GraphRAG

__all__ = [
    "DocumentLoader",
    "EmbeddingManager",
    "DocumentRetriever",
    "OllamaLLM",
    "GraphRAG",
]
