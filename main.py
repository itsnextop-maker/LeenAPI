"""
LeenAPI (free version): chat, image and voice agent in one API.
- Chat + understanding speech: Google Gemini free tier (needs GEMINI_API_KEY)
- Images: Pollinations (free, no key)
- Speech output: gTTS (free, no key)
"""
import asyncio
import base64
import io
import os
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="LeenAPI")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# ---------- security: only people with your password can use LeenAPI ----------
def auth(x_api_key: str = Header(default="")):
    secret = os.getenv("APP_API_KEY")
    if not secret or x_api_key != secret:
        raise HTTPException(401, "Invalid or missing x-api-key header")


# ---------- helpers ----------
async def http(method: str, url: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        r = await client.request(method, url, **kwargs)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"Upstream error: {r.text[:300]}")
    return r


async def gemini(contents: list, system: str | None = None) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(500, "GEMINI_API_KEY is not set on the server")
    body: dict = {"contents": contents}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    r = await http(
        "POST",
        GEMINI_URL.format(model=GEMINI_MODEL),
        headers={"x-goog-api-key": key},
        json=body,
    )
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(502, "Gemini returned no text (blocked or rate limited)")


async def transcribe(file: UploadFile) -> str:
    data = await file.read()
    mime = file.content_type or ""
    if not mime.startswith("audio/"):
        mime = "audio/mp3"
    contents = [{
        "role": "user",
        "parts": [
            {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}},
            {"text": "Transcribe this audio exactly. Reply with only the transcript."},
        ],
    }]
    return (await gemini(contents)).strip()


def _tts(text: str, lang: str) -> bytes:
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()


async def speak(text: str, lang: str = "en") -> bytes:
    try:
        return await asyncio.to_thread(_tts, text, lang)
    except Exception as e:
        raise HTTPException(502, f"Text-to-speech failed: {e}")


# ---------- request models ----------
class Msg(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatReq(BaseModel):
    messages: list[Msg]
    system: str = "You are a helpful assistant."


class ImageReq(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024


class SpeakReq(BaseModel):
    text: str
    lang: str = "en"


# ---------- endpoints ----------
@app.get("/")
def health():
    return {"name": "LeenAPI", "status": "ok"}


@app.post("/chat", dependencies=[Depends(auth)])
async def chat(req: ChatReq):
    contents = [
        {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
        for m in req.messages
    ]
    return {"reply": await gemini(contents, req.system)}


@app.post("/image", dependencies=[Depends(auth)])
async def image(req: ImageReq):
    url = (
        f"https://image.pollinations.ai/prompt/{quote(req.prompt)}"
        f"?width={req.width}&height={req.height}&nologo=true"
    )
    r = await http("GET", url)
    kind = r.headers.get("content-type", "")
    if not kind.startswith("image/"):
        raise HTTPException(502, "Image service did not return an image, try again in a minute")
    return {"mime": kind, "image_base64": base64.b64encode(r.content).decode()}


@app.post("/voice/transcribe", dependencies=[Depends(auth)])
async def voice_transcribe(file: UploadFile = File(...)):
    return {"text": await transcribe(file)}


@app.post("/voice/speak", dependencies=[Depends(auth)])
async def voice_speak(req: SpeakReq):
    audio = await speak(req.text, req.lang)
    return {"audio_base64": base64.b64encode(audio).decode()}


@app.post("/voice/agent", dependencies=[Depends(auth)])
async def voice_agent(file: UploadFile = File(...), lang: str = "en"):
    """Audio in -> understand -> reply -> speech out."""
    heard = await transcribe(file)
    reply = await gemini(
        [{"role": "user", "parts": [{"text": heard}]}],
        "You are a voice assistant. Reply in short, natural spoken sentences.",
    )
    audio = await speak(reply, lang)
    return {"heard": heard, "reply": reply, "audio_base64": base64.b64encode(audio).decode()}


@app.post("/video", dependencies=[Depends(auth)])
async def video():
    raise HTTPException(501, "Video generation has no free API, so it is turned off in the free version.")
