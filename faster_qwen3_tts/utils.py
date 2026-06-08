"""Shared utilities for TTS servers."""

import io
import sys
import base64
import struct
import contextlib
import numpy as np
import soundfile as sf


# ── Format registry ──────────────────────────────────────────────
# Maps format name → (soundfile format, MIME type, extra sf.write kwargs)

SUPPORTED_FORMATS = {
    "wav": ("WAV", "audio/wav", {"subtype": "PCM_16"}),
    "pcm": ("RAW", "audio/pcm", {"subtype": "PCM_16"}),
    "flac": ("FLAC", "audio/flac", {}),
    "mp3": ("MP3", "audio/mpeg", {}),
}


def media_type(fmt: str) -> str:
    """Return the MIME type for *fmt*, falling back to WAV for unknowns."""
    return SUPPORTED_FORMATS.get(fmt, SUPPORTED_FORMATS["wav"])[1]


# ── PCM16 conversion ────────────────────────────────────────────


def audio_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert a float32 audio array to raw 16-bit little-endian PCM bytes.

    Uses the standard ``*32768 + clip`` approach for full dynamic range.
    """
    np.nan_to_num(audio, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(audio * 32768, -32768, 32767).astype(np.int16).tobytes()


# ── WAV header ───────────────────────────────────────────────────


def create_wav_header(
    sample_rate: int,
    data_len: int = 0xFFFFFFFF,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Build a RIFF/WAVE header.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    data_len : int
        Byte length of the PCM data chunk.  Use the default
        ``0xFFFFFFFF`` for streaming (unknown size).
    num_channels : int
        Number of audio channels (default mono).
    bits_per_sample : int
        Bit depth (default 16).
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format tag
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_len,
    )


# ── Multi-format encoding ───────────────────────────────────────


def encode_audio(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode a numpy audio array to bytes in the specified format.

    Falls back to WAV for unrecognised *fmt* values.
    """
    if fmt not in SUPPORTED_FORMATS:
        fmt = "wav"
    sf_format, _, kwargs = SUPPORTED_FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format=sf_format, **kwargs)
    return buf.getvalue()


def encode_audio_base64(
    audio: np.ndarray,
    sample_rate: int,
    fmt: str = "wav",
) -> str:
    """Encode audio and return the result as a base64 string.

    Normalises the input to float32 and squeezes extra dimensions
    before encoding, matching the behaviour of the former
    ``demo/server.py::_to_wav_b64``.
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    return base64.b64encode(encode_audio(audio, sample_rate, fmt)).decode()


# ── Output filtering ─────────────────────────────────────────────

class _FilteredStdout:
    def __init__(self, stream, suppress_substrings):
        self._stream = stream
        self._suppress = suppress_substrings

    def write(self, data):
        if any(s in data for s in self._suppress):
            return len(data)
        return self._stream.write(data)

    def flush(self):
        return self._stream.flush()


# ── FlashAttention warning suppression ─────────────────────────

@contextlib.contextmanager
def suppress_flash_attn_warning():
    filtered = _FilteredStdout(
        sys.stdout,
        suppress_substrings=(
            "flash-attn is not installed",
            "manual PyTorch version",
            "Please install flash-attn",
        ),
    )
    with contextlib.redirect_stdout(filtered):
        yield
