"""
HASANEX AI SERVER
Single-file FastAPI backend for HASANEX AI STUDIO.

Render build command:
pip install fastapi "uvicorn[standard]" httpx python-dotenv

Render start command:
uvicorn HASANEX_AI_SERVER:app --host 0.0.0.0 --port $PORT

Set OPENROUTER_API_KEY in Render -> Environment.
Never put your API key in this file or GitHub.
"""

import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

APP_NAME = "HASANEX AI SERVER"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

app = FastAPI(title=APP_NAME, version="1.0.0")

# Initial setup CORS. For production, restrict this to your app's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    model: Optional[str] = None


@app.get("/")
async def home():
    return {"name": APP_NAME, "status": "running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "api_key_configured": bool(os.getenv("OPENROUTER_API_KEY")),
        "default_model": DEFAULT_MODEL,
    }


@app.get("/models")
async def models():
    return {"models": [{"id": "openrouter/free", "name": "OpenRouter Free Router"}]}


@app.post("/chat")
async def chat(body: ChatRequest):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured on the server.",
        )

    model = (body.model or DEFAULT_MODEL).strip()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": body.message}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_URL", "https://hasanex-ai-studio.example"),
        "X-Title": "HASANEX AI STUDIO",
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="OpenRouter timeout. Try again.")
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Cannot connect to OpenRouter.")

    if response.status_code == 401:
        raise HTTPException(
            status_code=502,
            detail="OpenRouter rejected the API key. Check the server secret.",
        )
    if response.status_code == 402:
        raise HTTPException(
            status_code=402,
            detail="This model/request requires credits. Select a free model or add credits only if you approve.",
        )
    if response.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="OpenRouter rate limit or free-model limit reached. Try later.",
        )
    if response.status_code >= 400:
        try:
            detail = response.json().get("error", {}).get(
                "message", "OpenRouter request failed."
            )
        except Exception:
            detail = "OpenRouter request failed."
        raise HTTPException(status_code=502, detail=str(detail)[:500])

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="OpenRouter returned no answer.")

    answer = choices[0].get("message", {}).get("content", "")
    if isinstance(answer, list):
        answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )

    return {
        "reply": answer,
        "model": data.get("model", model),
        "usage": data.get("usage", {}),
    }
