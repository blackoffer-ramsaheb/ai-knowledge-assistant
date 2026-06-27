# AI Knowledge Assistant (FastAPI + RAG + GraphRAG + SQLite + Dashboard)

An advanced AI Knowledge Assistant built on top of FastAPI, offering standard Retrieval-Augmented Generation (RAG) and keyword-relationship GraphRAG pipelines powered by local Ollama (`llama3.2`), HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`), ChromaDB, NetworkX, and SQLite. It includes a beautiful modern glassmorphism UI for chatting and a comprehensive Analytics Dashboard showing document ingestion status, chat histories, and usage trends.

---

## 🚀 Features

- **Multi-Mode Chatting**:
  - **Chat**: Plain Ollama (`llama3.2`) with no document context.
  - **RAG**: Semantic chunk retrieval from ChromaDB injected into the LLM prompt.
  - **GraphRAG**: Context-enrichment via a NetworkX-based keyword similarity graph, ranking adjacent document nodes using PageRank to retrieve related information.
- **One-Click Ingestion Flow**:
  - Drag-and-drop or browse PDF uploads.
  - Automatic conversion, text-normalization (collapsing garbled linebreaks/whitespace from document conversions), chunking, and embedding.
- **SQLite Database Persistence**:
  - Automatically logs users, document statuses (size, chunks, ingestion timestamp), and chat histories (questions, responses, mode, and source document citations).
- **Interactive Analytics Dashboard**:
  - Stat cards (Total Chats, Documents, Total Chunks, Ingested Docs) with animated counters.
  - Chats per mode (CSS horizontal bar charts).
  - Most Asked Questions list.
  - Full tabular views of document statuses and chat logs.
  - Retool-compatible JSON API endpoints.

---

## 📁 Directory Structure

```text
FastAPI Demo Project/
├── chroma_db/                  # Persisted ChromaDB Vector Store
├── uploads/                    # PDF uploads folder
├── rag/                        # RAG & GraphRAG Pipeline Modules
│   ├── __init__.py             # Exposes classes
│   ├── loader.py               # PDF loading (PyPDFLoader) & Text Splitter/Normalizer
│   ├── embedder.py             # HuggingFace Embeddings & ChromaDB storage
│   ├── retriever.py            # Similarity search retriever
│   ├── graph.py                # Lightweight GraphRAG using NetworkX
│   └── llm.py                  # Local Ollama llama3.2 connection & Prompting
├── static/
│   ├── script.js               # Client-side chat logic & upload handlers
│   ├── dashboard.js            # Client-side analytics dashboard logic
│   └── style.css               # Design system, chat, and dashboard styles
├── templates/
│   ├── ui.html                 # Chat UI
│   └── dashboard.html          # Analytics Dashboard UI
├── database.py                 # SQLite persistence layer
├── main.py                     # FastAPI application endpoints
├── requirements.txt            # Python dependencies list
└── knowledge_assistant.db      # Persisted SQLite database (auto-generated)
```

---

## 🛠️ Requirements & Setup

### 1. Prerequisites
- **Python**: version 3.11 recommended.
- **Ollama**: Must be running locally.
  - Download and install Ollama.
  - Pull the model:
    ```bash
    ollama pull llama3.2
    ```

### 2. Install Dependencies
Install all package requirements:
```bash
pip install -r requirements.txt
```

### 3. Running the Server
Start the FastAPI server:
```bash
uvicorn main:app --reload
```
Once running, you can access the application interfaces at:
- **Chat Interface**: [http://localhost:8000/](http://localhost:8000/)
- **Analytics Dashboard**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)

---

## 🔌 API Endpoints

### User Interface Routes
- `GET /` - Serves the Chat UI.
- `GET /dashboard` - Serves the Analytics Dashboard.

### Chat Operations (State Logged to SQLite)
- `POST /chat` - Plain Ollama chat (no context retrieval).
- `POST /rag-chat` - Standard RAG response.
- `POST /graph-chat` - Graph-enriched context RAG response.

### File Operations
- `POST /upload` - Upload PDF files to `uploads/` directory and log to SQLite.
- `POST /ingest` - Read uploaded PDFs, apply text-cleaning, compute embeddings, persist to ChromaDB, and mark as `ingested` in SQLite.
- `GET /documents` - Returns flat list of uploaded PDFs in `uploads/`.

### Dashboard & Retool-Compatible REST APIs
- `GET /api/analytics` - Aggregated stats, chats per mode, top questions, daily activity.
- `GET /api/chat-history` - Returns paginated list of chat history, modes used, and cited source documents.
- `GET /api/documents-db` - Returns all documents cataloged in the SQLite database.
- `GET /health` - Checks storage directory, database, and vector store health status.

---

## 🗄️ Database Schema (SQLite)

- **`users`**:
  - `id` (INTEGER, Primary Key)
  - `username` (TEXT, Unique)
  - `created_at` (TIMESTAMP)
- **`documents`**:
  - `id` (INTEGER, Primary Key)
  - `filename` (TEXT)
  - `file_size` (INTEGER)
  - `chunks` (INTEGER)
  - `status` (TEXT: 'uploaded' | 'ingested' | 'failed')
  - `uploaded_at` (TIMESTAMP)
  - `ingested_at` (TIMESTAMP)
- **`chat_history`**:
  - `id` (INTEGER, Primary Key)
  - `user_id` (INTEGER, Foreign Key)
  - `question` (TEXT)
  - `answer` (TEXT)
  - `mode` (TEXT: 'chat' | 'rag' | 'graph')
  - `sources` (TEXT: JSON String Array of file references)
  - `created_at` (TIMESTAMP)
