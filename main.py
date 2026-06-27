"""
main.py – FastAPI AI Knowledge Assistant.

Endpoints
---------
GET  /              → Chat UI
GET  /dashboard     → Analytics dashboard
POST /chat          → Plain Ollama chat (original)
POST /upload        → Upload PDF(s) to uploads/
POST /ingest        → Ingest uploaded PDFs into ChromaDB
POST /rag-chat      → RAG-powered chat
POST /graph-chat    → GraphRAG-powered chat
GET  /documents     → List uploaded PDFs
GET  /api/analytics → Dashboard analytics (JSON)
GET  /api/chat-history   → Chat history (JSON)
GET  /api/documents-db   → Documents from SQLite (JSON)
GET  /health        → Health check
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List

import requests as http_requests
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from database import Database
from rag.loader import DocumentLoader
from rag.embedder import EmbeddingManager
from rag.retriever import DocumentRetriever
from rag.llm import OllamaLLM
from rag.graph import GraphRAG

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
db = Database()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Knowledge Assistant", version="3.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    mode: str = "chat"
    sources: List[str] = []

# ---------------------------------------------------------------------------
# Lazy-initialised RAG components
# ---------------------------------------------------------------------------
_embedding_manager: EmbeddingManager | None = None
_retriever: DocumentRetriever | None = None
_rag_llm: OllamaLLM | None = None
_graph_rag: GraphRAG | None = None


def _get_embedding_manager() -> EmbeddingManager:
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager


def _get_retriever() -> DocumentRetriever:
    global _retriever
    if _retriever is None:
        manager = _get_embedding_manager()
        manager.load_vector_store()
        _retriever = DocumentRetriever(embedding_manager=manager)
    return _retriever


def _get_rag_llm() -> OllamaLLM:
    global _rag_llm
    if _rag_llm is None:
        _rag_llm = OllamaLLM(retriever=_get_retriever())
    return _rag_llm


def _get_graph_rag() -> GraphRAG:
    global _graph_rag
    if _graph_rag is None:
        _graph_rag = GraphRAG(retriever=_get_retriever())
    return _graph_rag


def _reset_rag_components() -> None:
    """Force re-initialisation after new documents are ingested."""
    global _retriever, _rag_llm, _graph_rag
    _retriever = None
    _rag_llm = None
    _graph_rag = None

# ---------------------------------------------------------------------------
# Original Ollama helper (kept for plain chat mode)
# ---------------------------------------------------------------------------

def ask_ollama(prompt: str) -> str:
    """Send a raw prompt to the local Ollama instance."""
    response = http_requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
        },
    )
    return response.json()["response"]

# ===========================================================================
# PAGES
# ===========================================================================

@app.get("/")
def home(request: Request):
    """Serve the main chat UI."""
    return templates.TemplateResponse("ui.html", {"request": request})


@app.get("/dashboard")
def dashboard(request: Request):
    """Serve the analytics dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


# ===========================================================================
# CHAT ENDPOINTS (all log to SQLite)
# ===========================================================================

@app.post("/chat")
def chat(data: ChatRequest):
    """Original plain Ollama chat – no RAG context."""
    user_message = data.message
    response = ask_ollama(user_message)

    # Log to database
    db.add_chat(question=user_message, answer=response, mode="chat", sources=[])

    return {"response": response, "mode": "chat", "sources": []}


@app.post("/rag-chat")
def rag_chat(data: ChatRequest):
    """Answer using plain RAG (retrieval + Ollama)."""
    try:
        llm = _get_rag_llm()
        answer = llm.ask(data.message, k=4)

        # Collect source metadata from the last retrieval.
        retriever = _get_retriever()
        docs = retriever.retrieve(data.message, k=4)
        sources = list({
            d.metadata.get("source", "unknown") for d in docs
        })

        # Log to database
        db.add_chat(
            question=data.message, answer=answer,
            mode="rag", sources=sources,
        )

        return {"response": answer, "mode": "rag", "sources": sources}
    except FileNotFoundError:
        return JSONResponse(
            status_code=400,
            content={
                "response": "No documents have been ingested yet. Please upload PDFs and click Ingest first.",
                "mode": "rag",
                "sources": [],
            },
        )
    except Exception as exc:
        logger.exception("RAG chat failed")
        return JSONResponse(
            status_code=500,
            content={"response": f"Error: {exc}", "mode": "rag", "sources": []},
        )


@app.post("/graph-chat")
def graph_chat(data: ChatRequest):
    """Answer using GraphRAG (graph-enriched retrieval + Ollama)."""
    try:
        graph_rag = _get_graph_rag()
        enriched_docs = graph_rag.get_related_chunks(data.message, k=6)

        if not enriched_docs:
            return {
                "response": "No relevant information found in the knowledge base.",
                "mode": "graph",
                "sources": [],
            }

        # Build context and generate answer via Ollama.
        llm = _get_rag_llm()
        context = llm._format_context(enriched_docs)
        prompt = llm._build_prompt(data.message, context)

        import ollama
        client = ollama.Client(host="http://localhost:11434")
        result = client.generate(model="llama3.2", prompt=prompt, stream=False)
        answer = result.get("response", "").strip()

        sources = list({
            d.metadata.get("source", "unknown") for d in enriched_docs
        })

        # Log to database
        db.add_chat(
            question=data.message, answer=answer,
            mode="graph", sources=sources,
        )

        return {"response": answer, "mode": "graph", "sources": sources}
    except FileNotFoundError:
        return JSONResponse(
            status_code=400,
            content={
                "response": "No documents have been ingested yet. Please upload PDFs and click Ingest first.",
                "mode": "graph",
                "sources": [],
            },
        )
    except Exception as exc:
        logger.exception("GraphRAG chat failed")
        return JSONResponse(
            status_code=500,
            content={"response": f"Error: {exc}", "mode": "graph", "sources": []},
        )


# ===========================================================================
# FILE UPLOAD & INGEST
# ===========================================================================

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more PDF files to the uploads/ directory."""
    saved: list[str] = []
    errors: list[str] = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            errors.append(f"Skipped non-PDF file: {file.filename}")
            continue
        try:
            dest = UPLOADS_DIR / file.filename
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved.append(file.filename)
            logger.info("Uploaded: %s", file.filename)

            # Record in database
            file_size = dest.stat().st_size
            db.add_document(filename=file.filename, file_size=file_size)
        except Exception as exc:
            logger.exception("Failed to save %s", file.filename)
            errors.append(f"Failed to save {file.filename}: {exc}")
        finally:
            await file.close()

    return {
        "saved": saved,
        "errors": errors,
        "total_uploaded": len(saved),
    }


@app.post("/ingest")
def ingest_documents():
    """Load all PDFs from uploads/, chunk, embed, and store in ChromaDB."""
    try:
        loader = DocumentLoader(uploads_dir=str(UPLOADS_DIR))
        documents = loader.load_documents()

        if not documents:
            return {"status": "warning", "message": "No PDF documents found in uploads/.", "chunks": 0}

        manager = _get_embedding_manager()
        manager.create_vector_store(documents)

        # Reset downstream components so they pick up the new data.
        _reset_rag_components()

        # Update all documents in DB as ingested
        db.mark_all_ingested(total_chunks=len(documents))

        return {
            "status": "success",
            "message": f"Ingested {len(documents)} chunks into ChromaDB.",
            "chunks": len(documents),
        }
    except Exception as exc:
        logger.exception("Ingestion failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


# ===========================================================================
# DOCUMENT LIST (filesystem)
# ===========================================================================

@app.get("/documents")
def list_documents():
    """Return a list of uploaded PDF filenames."""
    pdfs = sorted(f.name for f in UPLOADS_DIR.glob("*.pdf"))
    return {"documents": pdfs, "count": len(pdfs)}


# ===========================================================================
# DASHBOARD API ENDPOINTS (Retool-compatible)
# ===========================================================================

@app.get("/api/analytics")
def api_analytics():
    """Aggregated analytics for the dashboard."""
    return db.get_analytics()


@app.get("/api/chat-history")
def api_chat_history(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Paginated chat history."""
    rows = db.get_chat_history(limit=limit, offset=offset)
    return {"history": rows, "count": len(rows), "limit": limit, "offset": offset}


@app.get("/api/documents-db")
def api_documents_db():
    """Documents from the SQLite database."""
    docs = db.get_documents()
    return {"documents": docs, "count": len(docs)}


# ===========================================================================
# HEALTH
# ===========================================================================

@app.get("/health")
def health():
    """Basic health check."""
    chroma_ready = Path("chroma_db").exists() and any(Path("chroma_db").iterdir()) if Path("chroma_db").exists() else False
    return {
        "status": "healthy",
        "uploads_dir": str(UPLOADS_DIR.resolve()),
        "uploaded_pdfs": len(list(UPLOADS_DIR.glob("*.pdf"))),
        "chroma_ready": chroma_ready,
        "database": str(Path(db.db_path).resolve()),
    }