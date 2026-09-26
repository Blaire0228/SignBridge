import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Gemma_Web_with_RAG import (
    close_db_pool,
    initialize_db_pool,
    reconstruct_sentence,
    translate_tsl,
    warm_up_model,
)

TRANSLATION_CACHE_MAX_SIZE = int(os.getenv("TRANSLATION_CACHE_MAX_SIZE", "512"))
TRANSLATION_CACHE_VERSION = os.getenv("TRANSLATION_CACHE_VERSION", "2")
translation_cache = OrderedDict()
translation_cache_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_db_pool()
    try:
        warm_up_model()
        yield
    finally:
        close_db_pool()


app = FastAPI(title="Gemma TSL RAG API", lifespan=lifespan)
inference_lock = Lock()


def get_cached_translation(text: str):
    key = (TRANSLATION_CACHE_VERSION, text)
    with translation_cache_lock:
        result = translation_cache.get(key)
        if result is not None:
            translation_cache.move_to_end(key)
    return key, result


def cache_translation(key, result):
    if TRANSLATION_CACHE_MAX_SIZE <= 0:
        return
    with translation_cache_lock:
        translation_cache[key] = result
        translation_cache.move_to_end(key)
        while len(translation_cache) > TRANSLATION_CACHE_MAX_SIZE:
            translation_cache.popitem(last=False)


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    tsl: str
    prompt: str
    context: str


class ReconstructRequest(BaseModel):
    words: list[str]


class ReconstructResponse(BaseModel):
    words: list[str]
    sentence: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="text 不可為空")

    try:
        print(f"[Gemma API] 開始翻譯：{text}", flush=True)
        started = time.perf_counter()
        cache_key, result = get_cached_translation(text)
        cache_hit = result is not None
        if result is None:
            # 避免多個請求同時占用 GPU，並防止同一句同時重複推論。
            with inference_lock:
                cache_key, result = get_cached_translation(text)
                cache_hit = result is not None
                if result is None:
                    result = translate_tsl(text)
                    cache_translation(cache_key, result)
        tsl, prompt, context = result
        print(
            "[Gemma API timing] "
            f"cache={'hit' if cache_hit else 'miss'} "
            f"total={time.perf_counter() - started:.3f}s",
            flush=True,
        )
        print(f"[Gemma API] 翻譯完成：{tsl}", flush=True)

        return {
            "tsl": tsl,
            "prompt": prompt,
            "context": context,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gemma 翻譯失敗：{exc}",
        ) from exc


@app.post("/api/v1/reconstruct-sentence", response_model=ReconstructResponse)
def reconstruct_sign_sentence(request: ReconstructRequest):
    words = [word.strip() for word in request.words if word and word.strip()]
    if not words:
        return {"words": [], "sentence": ""}
    try:
        with inference_lock:
            sentence = reconstruct_sentence(words)
        return {"words": words, "sentence": sentence}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gemma 語句重組失敗：{exc}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
    )
