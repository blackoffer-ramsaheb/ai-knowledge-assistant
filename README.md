# FastAPI Demo Chatbot

A minimal FastAPI project with a simple browser chat UI wired to the `/chat` endpoint.

## What it does
- Serves a small single-page chat UI at `/`.
- Forwards user messages from `/chat` to a local model endpoint (`http://localhost:11434/api/generate`).

## Requirements
- Python 3.8+
- The app expects an LLM service listening at `http://localhost:11434` (the demo uses the `ask_ollama` helper in `main.py`).

Install Python dependencies:

```bash
python -m pip install fastapi uvicorn requests jinja2
```

## Run locally

Start the FastAPI app with Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open the UI in your browser:

- http://localhost:8000/

## Files of interest
- `main.py` — FastAPI app and `/chat` POST handler.
- `templates/ui.html` — chat UI template.
- `static/script.js` — client-side fetch logic for `/chat`.
- `static/style.css` — UI styles.

## Notes
- The backend's `ask_ollama` function calls `http://localhost:11434/api/generate`. If you don't have that service, `/chat` will fail; either run the model server or replace `ask_ollama` with a mock response for testing.

If you'd like, I can add a `requirements.txt` and a dev script next.
