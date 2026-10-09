"""
LeenAPI (free version with customer keys): chat, image and voice agent in one API.
- Chat + understanding speech: Google Gemini free tier (needs GEMINI_API_KEY)
- Images: Pollinations (free, no key). Speech output: gTTS (free).
- Customer keys: stored in a free Neon Postgres database (needs DATABASE_URL).
- APP_API_KEY is your master/admin password. Customers get their own keys.
"""
import asyncio
import base64
import hashlib
import hmac
import io
import os
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx
import psycopg
from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FREE_LIMIT = int(os.getenv("FREE_DAILY_LIMIT", "10"))        # requests per day for each free key
FREE_TOTAL = int(os.getenv("FREE_DAILY_TOTAL", "150"))      # requests per day for ALL free keys together
SIGNUPS_PER_IP = int(os.getenv("SIGNUPS_PER_IP", "3"))      # free signups per network per day
BANGLADESH = timezone(timedelta(hours=6))  # daily limits reset at midnight Bangladesh time


# ---------- database ----------
def db():
    # prepare_threshold=None keeps this safe with Neon's pooled connection
    return psycopg.connect(
        os.environ["DATABASE_URL"], autocommit=True, prepare_threshold=None, connect_timeout=15
    )


def today():
    return datetime.now(BANGLADESH).date()


def _init_db():
    with db() as conn:
        conn.execute(
            """create table if not exists api_keys (
                id serial primary key,
                name text not null,
                key_hash text unique not null,
                key_prefix text not null,
                daily_limit int not null default 100,
                active boolean not null default true,
                created_at timestamptz not null default now())"""
        )
        conn.execute(
            """create table if not exists usage (
                key_id int references api_keys(id) on delete cascade,
                day date not null,
                count int not null default 0,
                primary key (key_id, day))"""
        )
        conn.execute("alter table api_keys add column if not exists plan text not null default 'paid'")
        conn.execute("alter table api_keys add column if not exists email text")
        conn.execute("create unique index if not exists api_keys_email_idx on api_keys (lower(email)) where email is not null")
        conn.execute(
            """create table if not exists signups (
                ip_hash text not null, day date not null, count int not null default 0,
                primary key (ip_hash, day))"""
        )


async def run_db(fn, *args):
    if not os.getenv("DATABASE_URL"):
        raise HTTPException(503, "Database is not set up (DATABASE_URL missing)")
    try:
        return await asyncio.to_thread(fn, *args)
    except psycopg.Error as e:
        raise HTTPException(503, f"Database error: {str(e)[:150]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("DATABASE_URL"):
        try:
            await asyncio.to_thread(_init_db)
        except Exception as e:  # keep the API running even if the database is down
            print("Database init failed:", e)
    yield


app = FastAPI(title="LeenAPI", lifespan=lifespan)

# lets your website (on another address) call the API from a browser
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- security ----------
def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def is_master(key: str) -> bool:
    master = os.getenv("APP_API_KEY")
    return bool(master) and hmac.compare_digest(key.encode(), master.encode())


def _use_key(key_hash: str):
    """Checks a customer key and counts one request. Returns (status, key_id)."""
    with db() as conn:
        row = conn.execute(
            "select id, daily_limit, active, plan from api_keys where key_hash = %s", (key_hash,)
        ).fetchone()
        if not row:
            return "invalid", None
        key_id, limit, active, plan = row
        if not active:
            return "disabled", None
        if limit <= 0:
            return "limit", None
        if plan == "free":
            used = conn.execute(
                """select coalesce(sum(u.count), 0) from usage u
                   join api_keys k on k.id = u.key_id
                   where k.plan = 'free' and u.day = %s""",
                (today(),),
            ).fetchone()[0]
            if used >= FREE_TOTAL:
                return "freecap", None
        counted = conn.execute(
            """insert into usage (key_id, day, count) values (%s, %s, 1)
               on conflict (key_id, day) do update set count = usage.count + 1
               where usage.count < %s
               returning count""",
            (key_id, today(), limit),
        ).fetchone()
        if counted is None:
            return "limit", None
        return "ok", key_id


async def auth(x_api_key: str = Header(default="")):
    """Master key = unlimited. Customer key = checked against the database and daily limit."""
    if not x_api_key:
        raise HTTPException(401, "Missing x-api-key header")
    if is_master(x_api_key):
        return {"admin": True}
    status, key_id = await run_db(_use_key, hash_key(x_api_key))
    if status == "invalid":
        raise HTTPException(401, "Invalid API key")
    if status == "disabled":
        raise HTTPException(403, "This API key has been disabled")
    if status == "limit":
        raise HTTPException(429, "Daily limit reached. It resets at midnight Bangladesh time.")
    if status == "freecap":
        raise HTTPException(429, "Free capacity for today is used up. Try again tomorrow or get a paid plan.")
    return {"admin": False, "key_id": key_id}


async def admin_only(x_api_key: str = Header(default="")):
    if not x_api_key or not is_master(x_api_key):
        raise HTTPException(401, "Admin key required")


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


TLDS = {"com", "co.uk", "com.au", "co.in", "co.za", "ca", "ie"}


def _tts(text: str, lang: str, tld: str = "com") -> bytes:
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=lang, tld=tld if tld in TLDS else "com").write_to_fp(buf)
    return buf.getvalue()


async def speak(text: str, lang: str = "en", tld: str = "com") -> bytes:
    try:
        return await asyncio.to_thread(_tts, text, lang, tld)
    except Exception as e:
        raise HTTPException(502, f"Text-to-speech failed: {e}")


# ---------- request models ----------
class Msg(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatReq(BaseModel):
    messages: list[Msg]
    system: str = "You are Leen AI, a friendly assistant created by Leen from Bangladesh. If anyone asks who made you, say you were made by Leen from Bangladesh. Answer in the same language the user writes in, keep answers short and clear, and be helpful and honest."


class ImageReq(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024


class SpeakReq(BaseModel):
    text: str
    lang: str = "en"
    tld: str = "com"  # accent: com, co.uk, com.au, co.in, co.za, ca, ie


class NewKey(BaseModel):
    name: str
    daily_limit: int = 50


class LimitReq(BaseModel):
    daily_limit: int


# ---------- main endpoints ----------
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
    audio = await speak(req.text, req.lang, req.tld)
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


# ---------- customer: check own usage ----------
def _my_usage(key_id: int):
    with db() as conn:
        row = conn.execute(
            """select k.name, k.daily_limit, coalesce(u.count, 0)
               from api_keys k left join usage u on u.key_id = k.id and u.day = %s
               where k.id = %s""",
            (today(), key_id),
        ).fetchone()
    return {"name": row[0], "daily_limit": row[1], "used_today": row[2]}


@app.get("/me")
async def me(who: dict = Depends(auth)):
    if who.get("admin"):
        return {"admin": True}
    return await run_db(_my_usage, who["key_id"])


# ---------- admin: manage customer keys (master key only) ----------
def _create_key(name: str, key_hash: str, prefix: str, limit: int):
    with db() as conn:
        conn.execute(
            "insert into api_keys (name, key_hash, key_prefix, daily_limit) values (%s, %s, %s, %s)",
            (name, key_hash, prefix, limit),
        )


def _list_keys():
    with db() as conn:
        rows = conn.execute(
            """select k.id, k.name, k.key_prefix, k.daily_limit, k.active, k.created_at,
                      coalesce(u.count, 0), k.plan
               from api_keys k left join usage u on u.key_id = k.id and u.day = %s
               order by k.id""",
            (today(),),
        ).fetchall()
    return [
        {
            "id": r[0], "name": r[1], "key_starts_with": r[2], "daily_limit": r[3],
            "active": r[4], "created": r[5].isoformat(), "used_today": r[6], "plan": r[7],
        }
        for r in rows
    ]


def _update_key(key_id: int, column: str, value):
    allowed = {"active", "daily_limit"}
    if column not in allowed:
        raise ValueError("bad column")
    with db() as conn:
        row = conn.execute(
            f"update api_keys set {column} = %s where id = %s returning id", (value, key_id)
        ).fetchone()
    return row is not None


@app.post("/admin/keys", dependencies=[Depends(admin_only)])
async def create_key(req: NewKey):
    key = "leen_" + secrets.token_urlsafe(24)
    await run_db(_create_key, req.name, hash_key(key), key[:9], req.daily_limit)
    return {"api_key": key, "note": "Copy this key now. It cannot be shown again."}


@app.get("/admin/keys", dependencies=[Depends(admin_only)])
async def list_keys():
    return await run_db(_list_keys)


@app.post("/admin/keys/{key_id}/disable", dependencies=[Depends(admin_only)])
async def disable_key(key_id: int):
    if not await run_db(_update_key, key_id, "active", False):
        raise HTTPException(404, "Key not found")
    return {"id": key_id, "active": False}


@app.post("/admin/keys/{key_id}/enable", dependencies=[Depends(admin_only)])
async def enable_key(key_id: int):
    if not await run_db(_update_key, key_id, "active", True):
        raise HTTPException(404, "Key not found")
    return {"id": key_id, "active": True}


@app.post("/admin/keys/{key_id}/limit", dependencies=[Depends(admin_only)])
async def set_limit(key_id: int, req: LimitReq):
    if not await run_db(_update_key, key_id, "daily_limit", req.daily_limit):
        raise HTTPException(404, "Key not found")
    return {"id": key_id, "daily_limit": req.daily_limit}


# ---------- free developer signup ----------
class Signup(BaseModel):
    email: str
    name: str = ""


def _signup(email: str, name: str, ip_hash: str, key_hash: str, prefix: str):
    with db() as conn:
        row = conn.execute(
            """insert into signups (ip_hash, day, count) values (%s, %s, 1)
               on conflict (ip_hash, day) do update set count = signups.count + 1
               where signups.count < %s
               returning count""",
            (ip_hash, today(), SIGNUPS_PER_IP),
        ).fetchone()
        if row is None:
            return "ip"
        try:
            conn.execute(
                """insert into api_keys (name, key_hash, key_prefix, daily_limit, plan, email)
                   values (%s, %s, %s, %s, 'free', %s)""",
                (name or email, key_hash, prefix, FREE_LIMIT, email),
            )
        except psycopg.errors.UniqueViolation:
            return "exists"
    return "ok"


@app.post("/signup")
async def signup(req: Signup, request: Request):
    """Anyone can get ONE free key with a small daily limit."""
    email = req.email.strip().lower()
    if len(email) > 120 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(422, "Please enter a valid email address")
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    ip = fwd or (request.client.host if request.client else "unknown")
    key = "leen_" + secrets.token_urlsafe(24)
    result = await run_db(_signup, email, req.name.strip()[:60], hash_key(ip), hash_key(key), key[:9])
    if result == "ip":
        raise HTTPException(429, "Too many signups from this network today. Please try tomorrow.")
    if result == "exists":
        raise HTTPException(409, "This email already has a free key. Contact the owner if you lost it.")
    return {"api_key": key, "daily_limit": FREE_LIMIT, "note": "Copy this key now. It cannot be shown again."}
