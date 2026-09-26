import asyncio
import hashlib
import html
import hmac
import json
import os
import re
import secrets
import tempfile
import time
import urllib.parse
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langdetect import DetectorFactory, detect
from pydantic import BaseModel, Field
from merge_glb import (
    CachedAnimation,
    compute_timeline,
    extract_animation_cache,
    merge_glb_with_timeline,
)

DetectorFactory.seed = 0

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
ACTION_DIR = APP_DIR / "adjusted_actions"
MAPPING_PATH = APP_DIR / "tsl_animation_mapping.json"
MERGE_LOCK = asyncio.Lock()
GEMMA_API_URL = os.getenv("GEMMA_API_URL", "http://127.0.0.1:8001").rstrip("/")
MAX_SIGN_VIDEO_MB = int(os.getenv("MAX_SIGN_VIDEO_MB", "250"))
MAX_SIGN_VIDEO_BYTES = MAX_SIGN_VIDEO_MB * 1024 * 1024
GLB_CACHE_MAX = int(os.getenv("GLB_CACHE_MAX", "100"))
SIGN_VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-m4v",
    "application/octet-stream",
}
_sign_recognition_service = None
_sign_service_lock = asyncio.Lock()
_sign_stream_sessions = {}
_sign_stream_lock = asyncio.Lock()
CHAT_ROOM_TTL_SECONDS = int(os.getenv("CHAT_ROOM_TTL_SECONDS", "1800"))
CHAT_MAX_MESSAGE_LENGTH = 2000
_chat_rooms: dict[str, dict] = {}
_chat_lock = asyncio.Lock()

with MAPPING_PATH.open(encoding="utf-8") as mapping_file:
    TSL_ANIMATION_MAPPING = json.load(mapping_file)

ANIMATION_CACHE: dict[str, CachedAnimation] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gemma_client = httpx.AsyncClient(
        base_url=GEMMA_API_URL,
        timeout=300.0,
    )

    async def warm_animation_cache() -> None:
        print("[startup] 開始背景預載動畫關鍵影格...", flush=True)
        started = time.perf_counter()

        for glb_path in sorted(ACTION_DIR.glob("*.glb")):
            if glb_path.stem in ANIMATION_CACHE:
                continue

            try:
                ANIMATION_CACHE[glb_path.stem] = await asyncio.to_thread(
                    extract_animation_cache,
                    glb_path,
                )
            except Exception as exc:
                print(
                    f"[startup] 無法快取 {glb_path.name}: {exc}",
                    flush=True,
                )

        elapsed = time.perf_counter() - started
        print(
            f"[startup] 動畫快取完成：{len(ANIMATION_CACHE)} 個動作，"
            f"耗時 {elapsed:.2f} 秒",
            flush=True,
        )

    cache_task = asyncio.create_task(warm_animation_cache())

    try:
        yield
    finally:
        if not cache_task.done():
            cache_task.cancel()
        await app.state.gemma_client.aclose()


app = FastAPI(title="雙向手語翻譯 API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允許所有來源 (開發階段設為 "*" 最方便)
    allow_credentials=True,
    allow_methods=["*"], # 允許所有方法 (GET, POST 等)
    allow_headers=["*"], # 允許所有標頭
)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/actions", StaticFiles(directory=ACTION_DIR), name="actions")


@app.get("/health")
async def health():
    return {"status": "ok"}

class TslTranslationResponse(BaseModel):
    input_text: str
    tsl: str
    glb_url: str | None = None
    animation_words: list[str] = Field(default_factory=list)
    animation_timeline: list[dict[str, str | float]] = Field(default_factory=list)
    missing_words: list[str] = Field(default_factory=list)
    detected_language: str | None = None
    translated_text: str | None = None


def is_primarily_chinese(text: str) -> bool:
    """Return whether at least half of the letters are CJK ideographs."""
    letters = re.sub(r"[\s\d\W_]+", "", text)
    if not letters:
        return True
    chinese_characters = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", letters)
    return len(chinese_characters) / len(letters) >= 0.5


async def translate_foreign_text_to_zhtw(text: str, source_language: str) -> str:
    """Translate foreign-language input to Traditional Chinese with a fallback."""
    language_pair = f"{source_language}|zh-TW" if source_language else "en|zh-TW"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text, "langpair": language_pair},
                headers={"User-Agent": "TSL-App/1.0"},
            )
            if response.status_code == 200:
                translated = response.json().get("responseData", {}).get(
                    "translatedText"
                )
                if translated and not translated.startswith("MYMEMORY WARNING:"):
                    return html.unescape(translated).strip()
    except (httpx.HTTPError, ValueError, TypeError):
        pass

    try:
        query = urllib.parse.urlencode(
            {"sl": source_language or "auto", "tl": "zh-TW", "q": text}
        )
        async with httpx.AsyncClient(timeout=3.5) as client:
            response = await client.get(
                f"https://translate.google.com/m?{query}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if response.status_code == 200:
                match = re.search(
                    r'class=["\']result-container["\']>(.*?)</div>',
                    response.text,
                    flags=re.DOTALL,
                )
                if match:
                    translated = re.sub(r"<[^>]+>", "", match.group(1))
                    return html.unescape(translated).strip()
    except httpx.HTTPError:
        pass

    return text


async def process_input_language(text: str) -> tuple[str, str, str | None]:
    """Detect non-Chinese input and translate it before TSL generation."""
    normalized_text = text.strip()
    if not normalized_text:
        return normalized_text, "zh", None
    if is_primarily_chinese(normalized_text):
        return normalized_text, "zh", None

    try:
        detected_language = await asyncio.to_thread(detect, normalized_text)
    except Exception:
        detected_language = "en"
    if detected_language.startswith("zh"):
        return normalized_text, "zh", None

    translated_text = await translate_foreign_text_to_zhtw(
        normalized_text, detected_language
    )
    print(
        "[Language Detect] "
        f"language={detected_language} input={normalized_text!r} "
        f"translated={translated_text!r}",
        flush=True,
    )
    return translated_text, detected_language, translated_text

# 定義前端傳來的 JSON 資料結構
class TextToSignRequest(BaseModel):
    text: str


class GlossToSentenceRequest(BaseModel):
    words: list[str]


class GlossToSentenceResponse(BaseModel):
    words: list[str]
    sentence: str


class ChatRoomRequest(BaseModel):
    device_id: str = Field(min_length=8, max_length=128)


class ChatJoinRequest(ChatRoomRequest):
    room_id: str = Field(min_length=6, max_length=12)
    join_token: str | None = Field(default=None, max_length=256)


def _new_room_id() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(8))


def _room_payload(room_id: str, room: dict, device_id: str) -> dict:
    return {
        "room_id": room_id,
        "join_token": room["join_token"],
        "member_token": room["members"][device_id],
        "expires_at": room["expires_at"],
        "member_count": len(room["members"]),
    }


async def _prune_chat_rooms() -> None:
    now = time.time()
    expired = [
        room_id
        for room_id, room in _chat_rooms.items()
        if room["expires_at"] <= now
    ]
    for room_id in expired:
        room = _chat_rooms.pop(room_id)
        for socket in list(room["sockets"].values()):
            try:
                await socket.close(code=4001, reason="Room expired")
            except RuntimeError:
                pass


@app.post("/api/v1/chat/rooms", status_code=201)
async def create_chat_room(request: ChatRoomRequest):
    async with _chat_lock:
        await _prune_chat_rooms()
        room_id = _new_room_id()
        while room_id in _chat_rooms:
            room_id = _new_room_id()
        room = {
            "join_token": secrets.token_urlsafe(24),
            "expires_at": time.time() + CHAT_ROOM_TTL_SECONDS,
            "members": {request.device_id: secrets.token_urlsafe(32)},
            "sockets": {},
            "messages": deque(maxlen=100),
        }
        _chat_rooms[room_id] = room
        return _room_payload(room_id, room, request.device_id)


@app.post("/api/v1/chat/rooms/join")
async def join_chat_room(request: ChatJoinRequest):
    room_id = request.room_id.strip().upper()
    async with _chat_lock:
        await _prune_chat_rooms()
        room = _chat_rooms.get(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="找不到房間或房間已過期")
        if request.join_token is not None and not hmac.compare_digest(
            request.join_token, room["join_token"]
        ):
            raise HTTPException(status_code=403, detail="邀請已失效")
        if request.device_id not in room["members"] and len(room["members"]) >= 2:
            raise HTTPException(status_code=409, detail="房間已有兩位成員")
        room["members"].setdefault(request.device_id, secrets.token_urlsafe(32))
        return _room_payload(room_id, room, request.device_id)


async def _broadcast_chat(room: dict, payload: dict) -> None:
    dead: list[str] = []
    for device_id, socket in list(room["sockets"].items()):
        try:
            await socket.send_json(payload)
        except (RuntimeError, WebSocketDisconnect):
            dead.append(device_id)
    for device_id in dead:
        room["sockets"].pop(device_id, None)


@app.websocket("/api/v1/chat/ws/{room_id}")
async def chat_websocket(
    websocket: WebSocket,
    room_id: str,
    device_id: str,
    member_token: str,
):
    room_id = room_id.strip().upper()
    async with _chat_lock:
        await _prune_chat_rooms()
        room = _chat_rooms.get(room_id)
        expected_token = None if room is None else room["members"].get(device_id)
        if expected_token is None or not hmac.compare_digest(member_token, expected_token):
            await websocket.close(code=4003, reason="Invalid membership")
            return
        await websocket.accept()
        old_socket = room["sockets"].get(device_id)
        if old_socket is not None:
            await old_socket.close(code=4000, reason="Reconnected")
        room["sockets"][device_id] = websocket
        history = list(room["messages"])

    await websocket.send_json({"type": "history", "messages": history})
    await _broadcast_chat(
        room,
        {"type": "presence", "member_count": len(room["sockets"])},
    )
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") != "message":
                continue
            text = str(data.get("text", "")).strip()
            if not text or len(text) > CHAT_MAX_MESSAGE_LENGTH:
                await websocket.send_json(
                    {"type": "error", "message": "訊息內容無效或過長"}
                )
                continue
            message = {
                "type": "message",
                "id": uuid.uuid4().hex,
                "sender_id": device_id,
                "text": text,
                "source": str(data.get("source", "text"))[:32],
                "tsl": str(data.get("tsl", ""))[:CHAT_MAX_MESSAGE_LENGTH],
                "animation_url": str(data.get("animation_url", ""))[:2048],
                "created_at": int(time.time() * 1000),
            }
            room["messages"].append(message)
            await _broadcast_chat(room, message)
    except (WebSocketDisconnect, ValueError, json.JSONDecodeError):
        pass
    finally:
        if room["sockets"].get(device_id) is websocket:
            room["sockets"].pop(device_id, None)
        await _broadcast_chat(
            room,
            {"type": "presence", "member_count": len(room["sockets"])},
        )


@app.post("/api/v1/gloss-to-sentence", response_model=GlossToSentenceResponse)
async def gloss_to_sentence(request: GlossToSentenceRequest):
    words = [word.strip() for word in request.words if word and word.strip()]
    if not words:
        return GlossToSentenceResponse(words=[], sentence="")

    fallback_sentence = " ".join(words)
    try:
        response = await app.state.gemma_client.post(
            "/api/v1/reconstruct-sentence",
            json={"words": words},
            timeout=30.0,
        )
        response.raise_for_status()
        sentence = response.json().get("sentence", "").strip()
        return GlossToSentenceResponse(
            words=words,
            sentence=sentence or fallback_sentence,
        )
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        print(f"[gloss-to-sentence fallback] {exc}", flush=True)
        return GlossToSentenceResponse(words=words, sentence=fallback_sentence)


async def get_sign_recognition_service():
    global _sign_recognition_service
    if _sign_recognition_service is None:
        async with _sign_service_lock:
            if _sign_recognition_service is None:
                try:
                    from sign_recognition import SignRecognitionService

                    _sign_recognition_service = await asyncio.to_thread(
                        SignRecognitionService
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"手語辨識服務初始化失敗：{exc}",
                    ) from exc
    return _sign_recognition_service


def is_recognition_error(exc: Exception) -> bool:
    from sign_recognition import SignRecognitionError

    return isinstance(exc, SignRecognitionError)


async def save_uploaded_video(upload: UploadFile, output_path: Path) -> None:
    total_bytes = 0
    with output_path.open("wb") as output_file:
        while chunk := await upload.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_SIGN_VIDEO_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"影片不可超過 {MAX_SIGN_VIDEO_MB} MB",
                )
            output_file.write(chunk)
    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="上傳的影片是空檔案")


@app.post("/api/v1/recognize-sign")
async def recognize_sign(video: UploadFile = File(...)):
    if video.content_type not in SIGN_VIDEO_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="僅接受 MP4、MOV 或 M4V 影片")

    suffix = Path(video.filename or "recording.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v"}:
        suffix = ".mp4"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
        await save_uploaded_video(video, temp_path)
        service = await get_sign_recognition_service()
        result = await asyncio.to_thread(service.recognize_video, temp_path)
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        if is_recognition_error(exc):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail="手語辨識失敗") from exc
    finally:
        await video.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/api/v1/recognize-sign-stream/start")
async def start_sign_stream():
    service = await get_sign_recognition_service()
    session_id = uuid.uuid4().hex
    try:
        session = await asyncio.to_thread(service.start_streaming_session)
    except Exception as exc:
        if is_recognition_error(exc):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail="無法啟動即時辨識") from exc
    async with _sign_stream_lock:
        _sign_stream_sessions[session_id] = {"session": session}
    return {"session_id": session_id}


def _parse_int_list(value: str | None, field: str) -> list[int]:
    try:
        result = [int(item) for item in (value or "").split(",")]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field} 格式錯誤") from exc
    if len(result) != 3:
        raise HTTPException(status_code=400, detail=f"{field} 必須包含三個值")
    return result


@app.post("/api/v1/recognize-sign-stream/{session_id}/frame")
async def add_sign_stream_frame(session_id: str, request: Request):
    async with _sign_stream_lock:
        session_entry = _sign_stream_sessions.get(session_id)
    if session_entry is None:
        raise HTTPException(status_code=404, detail="即時辨識工作階段不存在")
    session = session_entry["session"]

    try:
        width = int(request.headers.get("x-frame-width", "0"))
        height = int(request.headers.get("x-frame-height", "0"))
        rotation = int(request.headers.get("x-frame-rotation", "0"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="影格尺寸格式錯誤") from exc
    if width <= 0 or height <= 0 or rotation not in {0, 90, 180, 270}:
        raise HTTPException(status_code=400, detail="影格尺寸或旋轉角度錯誤")

    plane_lengths = _parse_int_list(
        request.headers.get("x-plane-lengths"), "x-plane-lengths"
    )
    row_strides = _parse_int_list(
        request.headers.get("x-row-strides"), "x-row-strides"
    )
    pixel_strides = _parse_int_list(
        request.headers.get("x-pixel-strides"), "x-pixel-strides"
    )
    payload = await request.body()
    if len(payload) > 4 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="相機影格過大")
    try:
        await asyncio.to_thread(
            session.add_yuv420_frame,
            payload,
            width=width,
            height=height,
            rotation=rotation,
            plane_lengths=plane_lengths,
            row_strides=row_strides,
            pixel_strides=pixel_strides,
        )
    except Exception as exc:
        if is_recognition_error(exc):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail="無法處理即時影格") from exc
    return {"accepted": True}


@app.post("/api/v1/recognize-sign-stream/{session_id}/finish")
async def finish_sign_stream(session_id: str):
    async with _sign_stream_lock:
        session_entry = _sign_stream_sessions.pop(session_id, None)
    if session_entry is None:
        raise HTTPException(status_code=404, detail="即時辨識工作階段不存在")
    session = session_entry["session"]
    try:
        result = await asyncio.to_thread(session.finish)
        payload = result.to_dict()
        print(
            f"[sign-stream] model={result.model_source} "
            f"frames={result.frame_count} label={result.raw_label}", flush=True
        )
        return payload
    except Exception as exc:
        session.close()
        if is_recognition_error(exc):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail="即時手語辨識失敗") from exc


@app.delete("/api/v1/recognize-sign-stream/{session_id}")
async def cancel_sign_stream(session_id: str):
    async with _sign_stream_lock:
        session_entry = _sign_stream_sessions.pop(session_id, None)
    if session_entry is not None:
        await asyncio.to_thread(session_entry["session"].close)
    return {"cancelled": True}


def resolve_animation_paths(tsl: str) -> tuple[list[Path], list[str], list[str]]:
    """把 Gemma 的空白分隔 TSL 詞彙轉成實際 GLB 路徑。"""
    paths: list[Path] = []
    animation_words: list[str] = []
    missing_words: list[str] = []

    for raw_word in tsl.split():
        word = raw_word.strip("，。！？、,.；;：:（）()[]【】")
        if not word:
            continue
        animation_name = TSL_ANIMATION_MAPPING.get(word)
        if not animation_name:
            missing_words.append(word)
            continue
        animation_path = ACTION_DIR / f"{animation_name}.glb"
        if not animation_path.is_file():
            missing_words.append(word)
            continue
        paths.append(animation_path)
        animation_words.append(word)

    return paths, animation_words, missing_words

def evict_glb_cache() -> None:
    """移除最舊的快取 GLB，使數量不超過設定上限。"""
    cached_files = sorted(
        STATIC_DIR.glob("output_sign_*.glb"),
        key=lambda path: path.stat().st_mtime,
    )

    while len(cached_files) > GLB_CACHE_MAX:
        oldest_file = cached_files.pop(0)
        oldest_file.unlink(missing_ok=True)

async def create_animation(
    tsl: str,
) -> tuple[str | None, list[str], list[str], list[dict[str, str | float]]]:
    paths, animation_words, missing_words = resolve_animation_paths(tsl)

    if not paths:
        return None, animation_words, missing_words, []

    idle_path = ACTION_DIR / "idle.glb"
    if not idle_path.is_file():
        raise FileNotFoundError(f"找不到起始／結尾動作：{idle_path}")

    # 取得每個動作的關鍵影格快取。
    action_caches: list[CachedAnimation] = []

    for animation_path in paths:
        cached_animation = ANIMATION_CACHE.get(animation_path.stem)

        if cached_animation is None:
            cached_animation = await asyncio.to_thread(
                extract_animation_cache,
                animation_path,
            )
            ANIMATION_CACHE[animation_path.stem] = cached_animation

        action_caches.append(cached_animation)

    idle_cache = ANIMATION_CACHE.get("idle")

    if idle_cache is None:
        idle_cache = await asyncio.to_thread(
            extract_animation_cache,
            idle_path,
        )
        ANIMATION_CACHE["idle"] = idle_cache

    # 相同的手語詞彙序列會得到相同檔名。
    cache_source = " ".join(animation_words)
    cache_key = hashlib.sha256(cache_source.encode("utf-8")).hexdigest()[:16]

    filename = f"output_sign_{cache_key}.glb"
    output_path = STATIC_DIR / filename

    # 相同句子先前已經合併過，直接重用檔案。
    if output_path.is_file():
        animation_timeline = compute_timeline(
            action_caches,
            animation_words,
            first_duration_seconds=0.8,
            transition_seconds=0.8,
        )

        return (
            filename,
            animation_words,
            missing_words,
            animation_timeline,
        )

    async with MERGE_LOCK:
        # 等待鎖期間，其他請求可能已經產生同一個檔案。
        if output_path.is_file():
            animation_timeline = compute_timeline(
                action_caches,
                animation_words,
                first_duration_seconds=0.8,
                transition_seconds=0.8,
            )
        else:
            animation_timeline = await asyncio.to_thread(
                merge_glb_with_timeline,
                idle_path,
                action_caches,
                animation_words,
                idle_cache,
                output_path,
                transition_seconds=0.8,
                first_duration_seconds=0.8,
                first_transition_seconds=0.8,
            )

    await asyncio.to_thread(evict_glb_cache)

    return (
        filename,
        animation_words,
        missing_words,
        animation_timeline,
    )

@app.post(
    "/api/v1/translate-to-tsl",
    response_model=TslTranslationResponse,
)
async def translate_to_tsl(request: TextToSignRequest):
    raw_input = request.text.strip()
    if not raw_input:
        raise HTTPException(status_code=400, detail="text 不可為空")
    target_chinese, detected_language, translated_text = (
        await process_input_language(raw_input)
    )
    try:
        print(
            f"[API] 收到翻譯請求：{raw_input} (處理繁中: {target_chinese})",
            flush=True,
        )
        response = await app.state.gemma_client.post(
            "/api/v1/translate",
            json={"text": target_chinese},
        )
        response.raise_for_status()

        tsl = response.json()["tsl"]
        print(f"[API] Gemma 回傳 TSL：{tsl}", flush=True)
        filename, animation_words, missing_words, animation_timeline = (
            await create_animation(tsl)
        )
        print(
            f"[API] 動畫完成：{filename}; 使用={animation_words}; 缺少={missing_words}",
            flush=True,
        )
        return {
            "input_text": raw_input,
            "tsl": tsl,
            "glb_url": f"/static/{filename}?t={time.time()}" if filename else None,
            "animation_words": animation_words,
            "animation_timeline": animation_timeline,
            "missing_words": missing_words,
            "detected_language": detected_language,
            "translated_text": translated_text,
        }

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Gemma API 無法連線",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=exc.response.text,
        ) from exc

    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail="TSL 已翻譯，但 GLB 動畫合併失敗",
        ) from exc

@app.post("/api/v1/text-to-sign")
async def generate_sign_animation(request: TextToSignRequest):
    """相容舊入口：直接把已翻譯好的 TSL 詞彙合併成動畫。"""
    try:
        filename, animation_words, missing_words, animation_timeline = (
            await create_animation(request.text.replace(",", " "))
        )
        if not filename:
            raise HTTPException(
                status_code=422,
                detail={"message": "沒有可用的動畫詞彙", "missing_words": missing_words},
            )
        return {
            "status": "success",
            "glb_url": f"/static/{filename}?t={time.time()}",
            "animation_words": animation_words,
            "animation_timeline": animation_timeline,
            "missing_words": missing_words,
        }
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=500, detail="GLB 動畫合併失敗") from exc
