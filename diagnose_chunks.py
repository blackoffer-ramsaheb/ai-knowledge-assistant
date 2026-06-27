"""Quick test: show cleaned chunks only (no retrieval — ChromaDB was reset)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from rag.loader import DocumentLoader

loader = DocumentLoader(uploads_dir="uploads")
docs = loader.load_documents()

print(f"Total chunks: {len(docs)}\n")
for i, doc in enumerate(docs):
    page = doc.metadata.get("page", "?")
    print(f"--- Chunk {i+1} (page {page}) [{len(doc.page_content)} chars] ---")
    print(doc.page_content[:500])
    print("...\n" if len(doc.page_content) > 500 else "\n")
