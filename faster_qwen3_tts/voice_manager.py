"""Voice profile management with persistent storage and TTL-based caching."""
import atexit
import hashlib
import json
import logging
import os
import struct
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

if TYPE_CHECKING:
    from .model import FasterQwen3TTS

logger = logging.getLogger(__name__)

# Centralised mode identifiers — avoids silent bugs from typos like "Xvec".
MODE_XVEC = "xvec"
MODE_ICL = "icl"


@dataclass
class VoiceEntry:
    """Metadata for a registered voice profile."""

    name: str
    pt_path: str  # Relative to storage_dir
    mode: str  # "xvec" | "icl"
    ref_text: str = ""
    persistent: bool = True
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Seconds since last use; None = never expires

    @property
    def expired(self) -> bool:
        if self.ttl is None or self.persistent:
            return False
        return time.time() - self.last_used_at > self.ttl

    def touch(self):
        """Update last_used_at to now."""
        self.last_used_at = time.time()


class VoiceManager:
    """Manages voice profiles with persistent storage and TTL-based caching.

    Storage layout::

        {storage_dir}/
        ├── registry.json
        ├── persistent/
        │   └── {name}.pt
        └── cache/
            └── {content_hash}.pt
    """

    def __init__(
        self,
        model: "FasterQwen3TTS",
        storage_dir: Union[str, Path],
        default_ttl: float = 3600.0,
    ):
        self._model = model
        self.storage_dir = Path(storage_dir)
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._registry: Dict[str, VoiceEntry] = {}
        self._dirty = False

        (self.storage_dir / "persistent").mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "cache").mkdir(parents=True, exist_ok=True)

        self._load_registry()

        # Ensure dirty TTL touches are persisted on clean shutdown,
        # so last_used_at is not lost for read-only server processes.
        atexit.register(self.flush)

    # ── Persistence ──────────────────────────────────────────────

    @property
    def _registry_path(self) -> Path:
        return self.storage_dir / "registry.json"

    def _load_registry(self):
        if not self._registry_path.exists():
            return
        with open(self._registry_path, "r") as f:
            data = json.load(f)
        for name, entry_data in data.items():
            self._registry[name] = VoiceEntry(**entry_data)
        logger.info(f"Loaded {len(self._registry)} voice(s) from registry")

    def _save_registry(self):
        """Persist registry to disk.  Caller must hold ``_lock``."""
        data = {name: asdict(entry) for name, entry in self._registry.items()}
        tmp = self._registry_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(self._registry_path)
        self._dirty = False

    def _mark_dirty(self):
        """Mark registry as needing persistence.  Caller must hold ``_lock``.

        Used instead of ``_save_registry()`` on hot-path read operations (e.g.
        ``get()`` touch) to avoid blocking I/O on every request.  Changes are
        flushed to disk on the next mutation or explicit ``flush()`` call.
        """
        self._dirty = True

    def flush(self):
        """Persist pending changes to disk (if any)."""
        with self._lock:
            if self._dirty:
                self._save_registry()

    # ── CRUD ─────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        ref_audio: Union[str, Path],
        ref_text: str = "",
        xvec_only: bool = True,
        append_silence: bool = True,
    ) -> VoiceEntry:
        """Register a persistent (pre-defined) voice profile.

        Extracts the voice prompt from *ref_audio*, saves as ``.pt``,
        and records metadata in the registry.
        """
        with self._lock:
            if name in self._registry:
                raise ValueError(
                    f"Voice '{name}' already exists. Delete it first or choose another name."
                )

        # Extract outside the lock (GPU work, may take a moment)
        prompt = self._model.extract_voice_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
        )

        with self._lock:
            # Double-check after re-acquiring
            if name in self._registry:
                raise ValueError(f"Voice '{name}' was created concurrently.")

            pt_rel = f"persistent/{name}.pt"
            torch.save(prompt, self.storage_dir / pt_rel)

            entry = VoiceEntry(
                name=name,
                pt_path=pt_rel,
                mode=prompt["mode"],
                ref_text=ref_text,
                persistent=True,
                ttl=None,
            )
            self._registry[name] = entry
            self._save_registry()
            logger.info(f"Registered voice '{name}' (mode={prompt['mode']})")
            return entry

    def delete(self, name: str) -> bool:
        """Delete a voice profile and its ``.pt`` file."""
        with self._lock:
            entry = self._registry.pop(name, None)
            if entry is None:
                return False
            pt_path = self.storage_dir / entry.pt_path
            if pt_path.exists():
                pt_path.unlink()
            self._save_registry()
            logger.info(f"Deleted voice '{name}'")
            return True

    def get(self, name: str) -> Optional[VoiceEntry]:
        """Get a voice entry by name.  Returns *None* if not found or expired."""
        with self._lock:
            entry = self._registry.get(name)
            if entry is None:
                return None
            if entry.expired:
                self._remove_entry(name)
                self._save_registry()
                return None
            entry.touch()
            self._mark_dirty()
            return entry

    def list_voices(self, include_expired: bool = False) -> List[VoiceEntry]:
        """List all registered voices (lazy-cleans expired entries)."""
        with self._lock:
            if include_expired:
                return list(self._registry.values())
            result = []
            expired = []
            for name, entry in self._registry.items():
                if entry.expired:
                    expired.append(name)
                else:
                    result.append(entry)
            for name in expired:
                self._remove_entry(name)
            if expired:
                self._save_registry()
            return result

    def update(self, name: str, **kwargs) -> VoiceEntry:
        """Update mutable metadata fields of a voice entry.

        Allowed fields: ``persistent``, ``ttl``, ``ref_text``.
        """
        allowed = {"persistent", "ttl", "ref_text"}
        bad = set(kwargs) - allowed
        if bad:
            raise ValueError(f"Cannot update field(s): {bad}")
        with self._lock:
            entry = self._registry.get(name)
            if entry is None:
                raise KeyError(f"Voice '{name}' not found")
            for key, value in kwargs.items():
                setattr(entry, key, value)
            self._save_registry()
            return entry

    # ── Runtime cache ────────────────────────────────────────────

    def get_or_create(
        self,
        ref_audio: Union[str, Path],
        ref_text: str = "",
        xvec_only: bool = True,
        append_silence: bool = True,
        ttl: Optional[float] = None,
    ) -> VoiceEntry:
        """Look up or create a cached (non-persistent) voice entry.

        The cache key is derived from the audio file path, size, mtime,
        ``ref_text``, and mode — so the same file at the same path hits
        the cache without re-reading the file content.
        """
        if ttl is None:
            ttl = self._default_ttl

        cache_name = self._audio_cache_key(str(ref_audio), ref_text, xvec_only, append_silence)

        with self._lock:
            entry = self._registry.get(cache_name)
            if entry is not None and not entry.expired:
                entry.touch()
                self._mark_dirty()
                return entry
            if entry is not None:
                self._remove_entry(cache_name)
                self._save_registry()

        # Extract outside lock (GPU work)
        prompt = self._model.extract_voice_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
        )

        with self._lock:
            # Double-check
            if cache_name in self._registry and not self._registry[cache_name].expired:
                return self._registry[cache_name]

            pt_rel = f"cache/{cache_name}.pt"
            torch.save(prompt, self.storage_dir / pt_rel)

            entry = VoiceEntry(
                name=cache_name,
                pt_path=pt_rel,
                mode=prompt["mode"],
                ref_text=ref_text,
                persistent=False,
                ttl=ttl,
            )
            self._registry[cache_name] = entry
            self._save_registry()
            logger.info(
                f"Cached voice '{cache_name}' (mode={prompt['mode']}, ttl={ttl}s)"
            )
            return entry

    def cleanup_expired(self) -> int:
        """Remove all expired cache entries.  Returns count removed."""
        with self._lock:
            expired = [n for n, e in self._registry.items() if e.expired]
            for name in expired:
                self._remove_entry(name)
            if expired:
                self._save_registry()
            return len(expired)

    # ── Loading ──────────────────────────────────────────────────

    def load_prompt(self, name: str) -> dict:
        """Load a voice's ``.pt`` as a raw dict (version, mode, tensors …).

        Can be fed to ``model._load_voice_prompt_pt`` or used directly.
        """
        entry = self.get(name)
        if entry is None:
            raise KeyError(f"Voice '{name}' not found or expired")
        pt_path = self.storage_dir / entry.pt_path
        # weights_only=True prevents arbitrary code execution via pickle
        return torch.load(pt_path, map_location=self._model.device, weights_only=True)

    def get_pt_path(self, name: str) -> Path:
        """Return the absolute ``.pt`` path for a voice.

        This can be passed directly as ``ref_audio`` to the model's
        ``generate_voice_clone*`` methods.
        """
        entry = self.get(name)
        if entry is None:
            raise KeyError(f"Voice '{name}' not found or expired")
        return self.storage_dir / entry.pt_path

    # ── Internal ─────────────────────────────────────────────────

    def _remove_entry(self, name: str):
        """Remove entry + .pt file.  Caller must hold ``_lock``."""
        entry = self._registry.pop(name, None)
        if entry is None:
            return
        pt_path = self.storage_dir / entry.pt_path
        if pt_path.exists():
            pt_path.unlink()

    @staticmethod
    def _audio_cache_key(
        ref_audio_path: str, ref_text: str, xvec_only: bool, append_silence: bool,
    ) -> str:
        """Derive a cache key from audio file metadata + parameters.

        Uses path + file size + mtime instead of reading the entire file content,
        avoiding potentially hundreds of ms of blocking I/O on large audio files
        in the request hot path.
        """
        h = hashlib.sha256()
        try:
            st = os.stat(ref_audio_path)
            h.update(ref_audio_path.encode())
            h.update(struct.pack("<Qd", st.st_size, st.st_mtime))
        except OSError:
            h.update(ref_audio_path.encode())
        h.update(ref_text.encode())
        h.update(b"xvec" if xvec_only else b"icl")
        h.update(b"silence" if append_silence else b"no_silence")
        return h.hexdigest()[:16]
