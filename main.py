"""
LeenAPI: chat, image, voice agent and video in one API.
This server is light: it only forwards requests to hosted AI services,
so it runs on a small free cloud server. No models run locally.
"""
import os
import base64
import httpx
from anthropic import AsyncAnthropic
from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="LeenAPI")

CHAT_MODEL = os.getenv("CHAT_MODEL", "claude-sonnet-5-5")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "minimax/video-01")  # any Replicate text-to-video model


# ---------- security: protect your keys from strangers ----------
def auth(x_api_key: str = Header(default="")):
    secret = os.getenv("APP_API_KEY")
    if not secret or x_api_key != secret:
        raise HTTPException(401, "Invalid or missing x-api-key header")


# ---------- helpers ----------
def claude_client() -> AsyncAnthropic:
    return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def openai_headers() -> dict:
    return {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}


def replicate_headers() -> dict:
    return {"Authorization": f"Bearer {os.getenv('REPLICATE_API_TOKEN')}"}


async def call(method: str, url: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=180) as http:
        r = await http.request(method, url, **kwargs)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"Upstream error: {r.text[:300]}")
    return r


async def ask_claude(messages: list, system: str) -> str:
    r = await claude_client().messages.create(
        model=CHAT_MODEL, max_tokens=1000, system=system, messages=messages
    )
    return r.content[0].text


async def speak(text: str, voice: str = "alloy") -> bytes:
    r = await call(
        "POST",
        "https://api.openai.com/v1/audio/speech",
        headers=openai_headers(),
        json={"model": "tts-1", "input": text, "voice": voice, "response_format": "mp3"},
    )
    return r.content


async def transcribe(file: UploadFile) -> str:
    data = await file.read()
    r = await call(
        "POST",
        "https://api.openai.com/v1/audio/transcriptions",
        headers=openai_headers(),
        files={"file": (file.filename or "audio.m4a", data, file.content_type or "audio/mp4")},
        data={"model": "whisper-1"},
    )
    return r.json()["text"]


# ---------- models ----------
class Msg(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatReq(BaseModel):
    messages: list[Msg]
    system: str = "You are a helpful assistant."


class ImageReq(BaseModel):
    prompt: str
    size: str = "1024x1024"


class SpeakReq(BaseModel):
    text: str
    voice: str = "alloy"


class VideoReq(BaseModel):
    prompt: str


# ---------- endpoints ----------
@app.get("/")
def health():
    return {"name": "LeenAPI", "status": "ok"}


@app.post("/chat", dependencies=[Depends(auth)])
async def chat(req: ChatReq):
    reply = await ask_claude([m.model_dump() for m in req.messages], req.system)
    return {"reply": reply}


@app.post("/image", dependencies=[Depends(auth)])
async def image(req: ImageReq):
    r = await call(
        "POST",
        "https://api.openai.com/v1/images/generations",
        headers=openai_headers(),
        json={"model": "gpt-image-1", "prompt": req.prompt, "size": req.size, "n": 1},
    )
    return {"image_base64": r.json()["data"][0]["b64_json"]}


@app.post("/voice/transcribe", dependencies=[Depends(auth)])
async def voice_transcribe(file: UploadFile = File(...)):
    return {"text": await transcribe(file)}


@app.post("/voice/speak", dependencies=[Depends(auth)])
async def voice_speak(req: SpeakReq):
    audio = await speak(req.text, req.voice)
    return {"audio_base64": base64.b64encode(audio).decode()}


@app.post("/voice/agent", dependencies=[Depends(auth)])
async def voice_agent(file: UploadFile = File(...)):
    """Audio in -> transcribe -> chat -> speech out."""
    heard = await transcribe(file)
    reply = await ask_claude(
        [{"role": "user", "content": heard}],
        "You are a voice assistant. Reply in short, natural spoken sentences.",
    )
    audio = await speak(reply)
    return {
        "heard": heard,
        "reply": reply,
        "audio_base64": base64.b64encode(audio).decode(),
    }


@app.post("/video", dependencies=[Depends(auth)])
async def video_start(req: VideoReq):
    """Starts a video job. Video takes minutes, so poll GET /video/{id}."""
    r = await call(
        "POST",
        f"https://api.replicate.com/v1/models/{VIDEO_MODEL}/predictions",
        headers=replicate_headers(),
        json={"input": {"prompt": req.prompt}},
    )
    return {"id": r.json()["id"], "status": r.json()["status"]}


@app.get("/video/{job_id}", dependencies=[Depends(auth)])
async def video_status(job_id: str):
    r = await call(
        "GET",
        f"https://api.replicate.com/v1/predictions/{job_id}",
        headers=replicate_headers(),
    )
    d = r.json()
    return {"status": d["status"], "video_url": d.get("output"), "error": d.get("error")}
