from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests

class ChatRequest(BaseModel):
    message: str

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("ui.html", {"request": request})


def ask_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


@app.post("/chat")
def chat(data: ChatRequest):
    user_message = data.message
    response = ask_ollama(user_message)
    return {"response": response}


@app.get("/health")
def health():
    return {"status": "healthy"}