"""
FastAPI-based TTS API server with voice management.

Provides OpenAI-compatible endpoints for text-to-speech synthesis
and voice profile CRUD, compatible with vLLM-Omni's /v1/audio/speech and /v1/audio/voices API.

Usage:
    python api_server.py --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base
    python api_server.py --model_path ./models/local --host 0.0.0.0 --port 8000
"""
import argparse
import asyncio
import base64
import logging
import re
import tempfile
import threading

import urllib.request
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from faster_qwen3_tts.audio_utils import (
    audio_to_pcm16_bytes,
    create_wav_header,
    encode_audio,
    media_type,
)
from faster_qwen3_tts.voice_manager import MODE_ICL, MODE_XVEC

logger = logging.getLogger(__name__)

# ── Globals (initialized at startup) ────────────────────────────

_model = None  # FasterQwen3TTS
_voice_manager = None  # VoiceManager
_gpu_lock = asyncio.Lock()  # Serialize GPU inference
_active_cancel_flag: threading.Event | None = None  # Cancel flag of in-flight request


# ── Pydantic schemas ────────────────────────────────────────────


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Registered voice name")
    ref_audio: Optional[str] = Field(
        None,
        description="Reference audio: base64 data URI, http(s) URL, or file:// path",
    )
    ref_text: Optional[str] = Field(None, description="Transcript of ref_audio")
    language: str = Field("Auto", description="Target language")
    xvec_only: bool = Field(
        True,
        description="True=x-vector only (fast), False=ICL mode (higher quality)",
    )
    stream: bool = Field(False, description="Enable streaming response")
    response_format: str = Field(
        "wav", description="Output format: wav, pcm, flac, mp3"
    )
    temperature: float = Field(0.9, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=1)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    max_new_tokens: int = Field(2048, ge=1)
    repetition_penalty: float = Field(1.05, ge=1.0)
    eos_logit_bias: float = Field(0.0, description="Additive bias on the EOS logit (positive = shorter, negative = longer)")
    instruct: Optional[str] = Field(None, description="Style/dialect instruction")
    chunk_size: int = Field(4, ge=1, description="Codec frames per streaming chunk")


class VoiceInfo(BaseModel):
    name: str
    mode: str
    ref_text: str = ""
    persistent: bool = True
    created_at: float = 0.0
    last_used_at: float = 0.0
    ttl: Optional[float] = None

    @classmethod
    def from_entry(cls, entry) -> "VoiceInfo":
        return cls(
            name=entry.name, mode=entry.mode, ref_text=entry.ref_text,
            persistent=entry.persistent, created_at=entry.created_at,
            last_used_at=entry.last_used_at, ttl=entry.ttl,
        )


class ErrorResponse(BaseModel):
    error: dict


# ── Audio helpers ────────────────────────────────────────────────

_REF_AUDIO_MIN_DURATION = 1.0  # seconds
_REF_AUDIO_MAX_DURATION = 30.0  # seconds


def _mime_to_ext(content_type: str) -> str:
    """Infer file extension from a MIME type or data-URI header string."""
    if "mp3" in content_type or "mpeg" in content_type:
        return ".mp3"
    if "flac" in content_type:
        return ".flac"
    if "ogg" in content_type:
        return ".ogg"
    return ".wav"


def _write_temp_file(data: bytes, ext: str) -> str:
    """Write bytes to a temp file and return its path (caller must clean up)."""
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name



# ── ref_audio resolution ────────────────────────────────────────


async def _resolve_ref_audio(ref_audio_str: str) -> tuple:
    """Resolve ref_audio string to (temp_file_path, should_cleanup).

    Supports:
    - data:audio/...;base64,{data}  → decode to temp file
    - http:// / https://            → download to temp file
    - file:///path                  → use path directly
    - /absolute/path                → use path directly
    """
    if ref_audio_str.startswith("data:"):
        try:
            header, b64data = ref_audio_str.split(",", 1)
        except ValueError:
            raise HTTPException(400, "Invalid data URI format")
        audio_bytes = base64.b64decode(b64data)
        return _write_temp_file(audio_bytes, _mime_to_ext(header)), True

    if ref_audio_str.startswith(("http://", "https://")):
        path = await asyncio.to_thread(_download_ref_audio, ref_audio_str)
        return path, True

    if ref_audio_str.startswith("file://"):
        path = ref_audio_str[7:]
    else:
        path = ref_audio_str

    if not Path(path).is_file():
        raise HTTPException(400, f"ref_audio file not found: {path}")
    return path, False


def _download_ref_audio(url: str) -> str:
    """Download ref audio from URL to a temp file (synchronous, meant for to_thread)."""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read(10 * 1024 * 1024 + 1)
            if len(data) > 10 * 1024 * 1024:
                raise HTTPException(400, "ref_audio download exceeds 10MB limit")
            ct = resp.headers.get("Content-Type", "")
    except urllib.error.URLError as e:
        raise HTTPException(400, f"Failed to download ref_audio: {e}")
    return _write_temp_file(data, _mime_to_ext(ct))


def _validate_ref_audio_duration(path: str):
    """Validate reference audio duration."""
    info = sf.info(path)
    duration = info.duration
    if duration < _REF_AUDIO_MIN_DURATION:
        raise HTTPException(
            400,
            f"Reference audio too short ({duration:.1f}s). "
            f"At least {_REF_AUDIO_MIN_DURATION:.0f}s required.",
        )
    if duration > _REF_AUDIO_MAX_DURATION:
        raise HTTPException(
            400,
            f"Reference audio too long ({duration:.1f}s). "
            f"Maximum {_REF_AUDIO_MAX_DURATION:.0f}s supported.",
        )


def _prepare_server_runtime() -> None:
    """Clean leftover runtime cache and warm up the model before serving."""
    cleanup = _voice_manager.startup_cleanup(clear_runtime_cache=True)
    logger.info(
        "Startup cache cleanup finished: removed %d registry entries, %d file(s)",
        cleanup["removed_entries"],
        cleanup["removed_files"],
    )

    if getattr(_model, "_warmed_up", False):
        return

    logger.info("Running startup warmup...")
    _model._warmup(prefill_len=100)
    logger.info("Startup warmup completed")


async def _listen_for_disconnect(
    raw_request: Request,
    cancel_flag: threading.Event,
) -> None:
    """Flip ``cancel_flag`` when the client closes the HTTP connection."""
    while True:
        msg = await raw_request.receive()
        if msg["type"] == "http.disconnect":
            cancel_flag.set()
            return


def _start_disconnect_watcher(
    raw_request: Request,
) -> tuple[threading.Event, asyncio.Task]:
    """Start a single request-scoped disconnect watcher."""
    cancel_flag = threading.Event()
    watcher = asyncio.create_task(_listen_for_disconnect(raw_request, cancel_flag))
    return cancel_flag, watcher


async def _stop_disconnect_watcher(watcher: asyncio.Task | None) -> None:
    """Stop a request-scoped disconnect watcher."""
    if watcher is None:
        return
    watcher.cancel()
    with suppress(asyncio.CancelledError):
        await watcher


async def _acquire_gpu_lock_or_abort(cancel_flag: threading.Event) -> bool:
    """Acquire the GPU lock unless the request is already aborted."""
    acquire_task = asyncio.create_task(_gpu_lock.acquire())
    cancel_waiter = asyncio.create_task(asyncio.to_thread(cancel_flag.wait))

    try:
        done, _pending = await asyncio.wait(
            {acquire_task, cancel_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if acquire_task in done:
            if cancel_flag.is_set():
                _gpu_lock.release()
                return False
            return True

        acquire_task.cancel()
        with suppress(asyncio.CancelledError):
            await acquire_task
        return False
    finally:
        cancel_waiter.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_waiter


# ── Streaming generator ─────────────────────────────────────────


async def _stream_tts(
    request: SpeechRequest,
    ref_audio_path: str,
    cancel_flag: threading.Event,
    watcher: asyncio.Task,
) -> AsyncGenerator[bytes, None]:
    """Async generator for streaming TTS audio chunks.

    The sync generator ``gen`` blocks the event loop during each GPU step,
    preventing uvicorn from detecting TCP disconnects.  We therefore run
    ``next(gen)`` in a thread-pool executor so the event loop stays free,
    and a background watcher task polls ``is_disconnected()`` concurrently.
    """
    import time as _time

    fmt = request.response_format
    if fmt not in ("pcm", "wav"):
        fmt = "pcm"

    first_chunk = True
    stream_start = _time.monotonic()
    chunk_idx = 0
    total_audio_samples = 0
    sr = getattr(_model, "sample_rate", 24000)
    gen = None
    lock_acquired = False
    _sentinel = object()
    _pending_future = None
    loop = asyncio.get_running_loop()
    if not await _acquire_gpu_lock_or_abort(cancel_flag):
        logger.info("Stream aborted before GPU lock acquisition")
        return

    lock_acquired = True
    try:
        gen = _model.generate_voice_clone_streaming(
            text=request.input,
            language=request.language,
            ref_audio=ref_audio_path,
            ref_text=request.ref_text or "",
            xvec_only=request.xvec_only,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            max_new_tokens=request.max_new_tokens,
            repetition_penalty=request.repetition_penalty,
            chunk_size=request.chunk_size,
            instruct=request.instruct,
            eos_logit_bias=request.eos_logit_bias,
            cancel_event=cancel_flag,
        )

        def _next_sync():
            try:
                return next(gen)
            except StopIteration:
                return _sentinel

        try:
            while True:
                _pending_future = loop.run_in_executor(None, _next_sync)
                result = await _pending_future
                _pending_future = None
                if result is _sentinel or cancel_flag.is_set():
                    break
                audio_chunk, sr, timing = result

                if audio_chunk is None or len(audio_chunk) == 0:
                    continue

                now = _time.monotonic()
                elapsed = now - stream_start
                total_audio_samples += len(audio_chunk)
                audio_dur = total_audio_samples / sr if sr else 0

                if first_chunk:
                    logger.debug(
                        "TTFA=%.3fs | prefill=%.1fms gen=%.1fms codec_decode=? | chunk_samples=%d sr=%d",
                        elapsed,
                        timing.get('prefill_ms', 0),
                        timing.get('decode_ms', 0),
                        len(audio_chunk), sr,
                    )
                    if fmt == "wav":
                        yield create_wav_header(sr)
                    first_chunk = False
                else:
                    logger.debug(
                        "chunk#%d t=%.3fs | gen=%.1fms | samples=%d audio_so_far=%.2fs RTF=%.2f",
                        chunk_idx, elapsed,
                        timing.get('decode_ms', 0),
                        len(audio_chunk), audio_dur,
                        audio_dur / elapsed if elapsed > 0 else 0,
                    )

                chunk_idx += 1
                yield audio_to_pcm16_bytes(audio_chunk)
        finally:
            client_disconnected = cancel_flag.is_set()

        if client_disconnected:
            logger.info("Stream aborted: client disconnected after %d chunks", chunk_idx)
        else:
            total_elapsed = _time.monotonic() - stream_start
            total_audio_dur = total_audio_samples / sr if sr else 0
            logger.debug(
                "stream done: %d chunks, %.2fs audio in %.3fs wall (RTF=%.2f)",
                chunk_idx, total_audio_dur, total_elapsed,
                total_audio_dur / total_elapsed if total_elapsed > 0 else 0,
            )
    except GeneratorExit:
        logger.warning("Stream aborted: generator closed")
    except asyncio.CancelledError:
        logger.warning("Stream aborted: task cancelled")
        # Do NOT re-raise: propagating CancelledError out of an async
        # generator crashes Starlette's internal TaskGroup.  Just let
        # the generator end; the finally block handles cleanup.
    finally:
        global _active_cancel_flag
        cancel_flag.set()
        if _active_cancel_flag is cancel_flag:
            _active_cancel_flag = None

        def _do_cleanup():
            """Close generator + release GPU lock (must run on event-loop thread)."""
            if gen is not None:
                try:
                    gen.close()
                except (ValueError, RuntimeError):
                    pass
            if lock_acquired:
                _gpu_lock.release()
            elapsed = _time.monotonic() - stream_start
            logger.debug("_stream_tts cleanup: GPU lock released after %.3fs", elapsed)

        if _pending_future is not None and not _pending_future.done():
            # Thread pool is still executing next(gen).  Defer cleanup
            # until the thread finishes so we don't call gen.close()
            # while the generator is executing, and don't release the
            # GPU lock while the GPU is still in use.
            _pending_future.add_done_callback(lambda _fut: _do_cleanup())
        else:
            _do_cleanup()

        # Cancel disconnect watcher (fire-and-forget; no await needed)
        if watcher is not None:
            watcher.cancel()


# ── App lifecycle ────────────────────────────────────────────────


import torch as _torch

# Jetson Thor 的 CUDA runtime 在 GPU 空闲 ~10-15 秒后会进入休眠状态，
# 此后第一次 GPU 操作（哪怕只是一个小 tensor 的 .to(cuda)）需要 ~2 秒唤醒。
# 这不是 GPU 时钟降频问题（jetson_clocks 锁频无效），而是 driver 层面的行为。
# 受影响的调用链：_tokenize_texts → processor() → input_ids.to(device)，
# 这一步位于 model.py 的 _prepare_generation() 中。
# 通过每隔几秒对一个 GPU tensor 做一次 add_ 操作来保持 CUDA context 活跃。
_KEEPALIVE_INTERVAL = 5  # seconds between CUDA keepalive pings
_keepalive_tensor = None  # tiny tensor pre-allocated on GPU


async def _cuda_keepalive():
    """Periodically nudge the GPU to prevent CUDA context from sleeping."""
    global _keepalive_tensor
    while True:
        await asyncio.sleep(_KEEPALIVE_INTERVAL)
        if _keepalive_tensor is not None:
            _keepalive_tensor.add_(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _keepalive_tensor
    if _model is not None:
        device = next(_model.model.model.parameters()).device
        _keepalive_tensor = _torch.zeros(1, device=device)
    task = asyncio.create_task(_cuda_keepalive())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(
    title="Faster Qwen3-TTS API",
    version="0.1.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Rate", "X-Request-Id"],
)


# ── TTS endpoint ─────────────────────────────────────────────────


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest, raw_request: Request):
    """Synthesize speech from text.

    Provide either `voice` (registered name) or `ref_audio` (inline audio).
    """
    global _active_cancel_flag
    ref_audio_path = None
    cleanup_path = None
    cancel_flag, watcher = _start_disconnect_watcher(raw_request)
    watcher_owned_by_stream = False

    # Auto-cancel any in-flight inference so the new request doesn't queue.
    prev = _active_cancel_flag
    if prev is not None:
        prev.set()
    _active_cancel_flag = cancel_flag

    try:
        # Validate ICL mode requires ref_text
        if not request.xvec_only and not request.ref_text:
            raise HTTPException(
                400,
                "ref_text is required when xvec_only=false (ICL mode)",
            )

        # Resolve voice source
        if request.voice:
            entry = _voice_manager.get(request.voice)
            if entry is None:
                raise HTTPException(404, f"Voice '{request.voice}' not found")
            ref_audio_path = str(_voice_manager.get_pt_path(request.voice))
        elif request.ref_audio:
            path, should_cleanup = await _resolve_ref_audio(request.ref_audio)
            if should_cleanup:
                cleanup_path = path
            _validate_ref_audio_duration(path)
            # Cache via VoiceManager
            entry = _voice_manager.get_or_create(
                ref_audio=path,
                ref_text=request.ref_text or "",
                xvec_only=request.xvec_only,
                append_silence=True,
            )
            ref_audio_path = str(_voice_manager.get_pt_path(entry.name))
        else:
            raise HTTPException(
                400, "Either 'voice' or 'ref_audio' must be provided"
            )

        if request.stream:
            # Streaming response
            fmt = request.response_format
            if fmt not in ("pcm", "wav"):
                fmt = "pcm"
            if cancel_flag.is_set():
                logger.info("Streaming request aborted before response started")
                return Response(status_code=499)
            watcher_owned_by_stream = True
            return StreamingResponse(
                _stream_tts(request, ref_audio_path, cancel_flag, watcher),
                media_type=media_type(fmt),
                headers={
                    "X-Sample-Rate": str(_model.sample_rate),
                    "X-Request-Id": str(uuid.uuid4()),
                },
            )

        # Non-streaming response — use streaming generator internally
        # so that client disconnect can abort GPU inference between chunks.
        audio_chunks = []
        sr = getattr(_model, "sample_rate", 24000)
        gen = None
        lock_acquired = False
        _sentinel = object()
        loop = asyncio.get_running_loop()
        if not await _acquire_gpu_lock_or_abort(cancel_flag):
            logger.info("Non-streaming request aborted before GPU lock acquisition")
            return Response(status_code=499)

        lock_acquired = True
        try:
            gen = _model.generate_voice_clone_streaming(
                text=request.input,
                language=request.language,
                ref_audio=ref_audio_path,
                ref_text=request.ref_text or "",
                xvec_only=request.xvec_only,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                max_new_tokens=request.max_new_tokens,
                repetition_penalty=request.repetition_penalty,
                chunk_size=request.chunk_size,
                instruct=request.instruct,
                eos_logit_bias=request.eos_logit_bias,
                cancel_event=cancel_flag,
            )

            def _next_sync():
                try:
                    return next(gen)
                except StopIteration:
                    return _sentinel

            try:
                while True:
                    result = await loop.run_in_executor(None, _next_sync)
                    if result is _sentinel or cancel_flag.is_set():
                        break
                    chunk, chunk_sr, _timing = result
                    if chunk is not None and len(chunk) > 0:
                        audio_chunks.append(chunk)
                        sr = chunk_sr
            finally:
                client_disconnected = cancel_flag.is_set()

            if client_disconnected:
                logger.info("Non-streaming request aborted: client disconnected")
                return Response(status_code=499)
        finally:
            cancel_flag.set()
            if gen is not None:
                gen.close()
            if lock_acquired:
                _gpu_lock.release()

        if not audio_chunks:
            raise HTTPException(500, "Generation produced no audio")

        audio = np.concatenate(audio_chunks)
        audio_bytes = encode_audio(audio, sr, request.response_format)

        return Response(
            content=audio_bytes,
            media_type=media_type(request.response_format),
            headers={
                "X-Sample-Rate": str(sr),
                "X-Request-Id": str(uuid.uuid4()),
            },
        )

    finally:
        if not watcher_owned_by_stream:
            cancel_flag.set()
            if _active_cancel_flag is cancel_flag:
                _active_cancel_flag = None
            await _stop_disconnect_watcher(watcher)
        if cleanup_path:
            Path(cleanup_path).unlink(missing_ok=True)


# ── Voice management endpoints ───────────────────────────────────


@app.get("/v1/audio/voices")
async def list_voices():
    """List all registered voices."""
    voices = _voice_manager.list_voices()
    return {
        "voices": [VoiceInfo.from_entry(v).model_dump() for v in voices]
    }


@app.get("/v1/audio/voices/{name}")
async def get_voice(name: str):
    """Get voice profile details."""
    entry = _voice_manager.get(name)
    if entry is None:
        raise HTTPException(404, f"Voice '{name}' not found")
    return VoiceInfo.from_entry(entry).model_dump()


_VOICE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _validate_voice_name(name: str):
    if not _VOICE_NAME_RE.match(name):
        raise HTTPException(
            400,
            "Voice name must be 1-64 chars, start with alphanumeric, "
            "and contain only [a-zA-Z0-9_-]",
        )


@app.post("/v1/audio/voices", status_code=201)
async def create_voice(
    name: str = Form(..., description="Voice name"),
    ref_text: str = Form("", description="Transcript of the audio"),
    mode: str = Form(MODE_XVEC, description="xvec or icl"),
    audio_sample: UploadFile = File(..., description="Reference audio file"),
):
    """Register a new persistent voice from uploaded audio."""
    _validate_voice_name(name)
    if mode not in (MODE_XVEC, MODE_ICL):
        raise HTTPException(400, "mode must be 'xvec' or 'icl'")
    if mode == MODE_ICL and not ref_text:
        raise HTTPException(400, "ref_text is required for ICL mode")

    # Check name not taken
    if _voice_manager.get(name) is not None:
        raise HTTPException(
            409, f"Voice '{name}' already exists. Delete it first."
        )

    # Save upload to temp file
    content = await audio_sample.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "Audio file exceeds 10MB limit")

    ext = Path(audio_sample.filename or "upload.wav").suffix or ".wav"
    tmp_path = _write_temp_file(content, ext)

    try:
        _validate_ref_audio_duration(tmp_path)

        entry = _voice_manager.register(
            name=name,
            ref_audio=tmp_path,
            ref_text=ref_text,
            xvec_only=(mode == MODE_XVEC),
        )

        return VoiceInfo.from_entry(entry).model_dump()

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.delete("/v1/audio/voices/{name}", status_code=204)
async def delete_voice(name: str):
    """Delete a voice profile."""
    if not _voice_manager.delete(name):
        raise HTTPException(404, f"Voice '{name}' not found")
    return Response(status_code=204)


# ── Error handler ────────────────────────────────────────────────


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error"
                if exc.status_code < 500
                else "server_error",
                "code": exc.status_code,
            }
        },
    )


# ── Main ─────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Faster Qwen3-TTS API Server")
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model path or HuggingFace Hub ID",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    # 在 Jetson Thor 上已确认 float16 可能出现问题，bfloat16 正常。
    # 必须使用 bfloat16。float16 动态范围小（max ~65504），talker logits 容易溢出为
    # inf/nan，导致采样时 CUDA device-side assert（"probability tensor contains
    # either inf, nan or element < 0"）或 EOS 永远采不到而生成超长无意义音频。
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype (bfloat16 recommended, float16 causes sampling errors)")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8000, help="Listen port")
    parser.add_argument(
        "--voices_dir",
        default="voices",
        help="Directory for voice profile storage",
    )
    parser.add_argument(
        "--default_ttl",
        type=float,
        default=3600.0,
        help="Default TTL for cached voices (seconds)",
    )
    parser.add_argument("--log_level", default="info", help="Log level")
    return parser.parse_args()


def main():
    global _model, _voice_manager

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("numba").setLevel(logging.WARNING)

    from faster_qwen3_tts import FasterQwen3TTS
    from faster_qwen3_tts.voice_manager import VoiceManager

    logger.info(f"Loading model from {args.model_path}...")
    _model = FasterQwen3TTS.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        # we use sdpa, flash attention 2 is not compatible with CUDA graphs
        # btw, flash-attn cost almost the same as sdpa, so no quality/speed tradeoff here
        attn_implementation="sdpa",
    )
    logger.info("Model loaded")

    _voice_manager = VoiceManager(
        model=_model,
        storage_dir=args.voices_dir,
        default_ttl=args.default_ttl,
    )
    _prepare_server_runtime()
    logger.info(
        f"VoiceManager ready ({len(_voice_manager.list_voices())} voices, "
        f"storage={args.voices_dir})"
    )

    # 强制使用 asyncio 事件循环而非 uvloop。uvloop 在 StreamingResponse 场景下
    # 会攒多个小 write 到下一次 I/O poll 才 flush，而 TTS 生成是同步阻塞的，
    # 导致事件循环一直拿不到控制权，多个 chunk 被攒在一起才发送，客户端 TTFA
    # 从 ~300ms 飙升到 ~1s。使用默认 asyncio loop 则每次 yield 都能及时 flush。
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
