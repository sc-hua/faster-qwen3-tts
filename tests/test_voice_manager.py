import importlib.util
import pickle
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _install_fake_torch():
    torch = types.ModuleType("torch")

    def save(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load(path, map_location=None, weights_only=False):
        with open(path, "rb") as f:
            return pickle.load(f)

    torch.save = save
    torch.load = load
    sys.modules["torch"] = torch


def _load_voice_manager_module():
    _install_fake_torch()
    spec = importlib.util.spec_from_file_location(
        "voice_manager_under_test",
        ROOT / "faster_qwen3_tts" / "voice_manager.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


voice_manager_module = _load_voice_manager_module()
VoiceEntry = voice_manager_module.VoiceEntry
VoiceManager = voice_manager_module.VoiceManager


class DummyModel:
    def __init__(self):
        self.device = "cpu"
        self.calls = 0

    def extract_voice_prompt(
        self,
        ref_audio,
        ref_text="",
        xvec_only=True,
        append_silence=True,
    ):
        self.calls += 1
        return {
            "mode": "xvec" if xvec_only else "icl",
            "ref_text": ref_text,
            "path": str(ref_audio),
            "append_silence": append_silence,
        }


def _make_manager(tmp_path, default_ttl=1.0):
    return VoiceManager(model=DummyModel(), storage_dir=tmp_path, default_ttl=default_ttl)


def _write_prompt(path):
    path.write_bytes(pickle.dumps({"mode": "xvec"}))


def test_get_or_create_uses_content_hash_for_same_audio(tmp_path):
    manager = _make_manager(tmp_path, default_ttl=3600.0)

    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "nested" / "b.wav"
    audio_b.parent.mkdir()

    payload = b"same-audio-bytes"
    audio_a.write_bytes(payload)
    audio_b.write_bytes(payload)

    first = manager.get_or_create(str(audio_a), ref_text="hello", xvec_only=True)
    second = manager.get_or_create(str(audio_b), ref_text="hello", xvec_only=True)

    assert first.name == second.name
    assert manager.get_pt_path(first.name).exists()
    assert manager._model.calls == 1


def test_startup_cleanup_clears_runtime_cache_but_keeps_persistent_voice(tmp_path):
    manager = _make_manager(tmp_path)

    persistent_path = tmp_path / "persistent" / "speaker.pt"
    runtime_path = tmp_path / "cache" / "runtime.pt"
    orphan_path = tmp_path / "cache" / "orphan.pt"

    _write_prompt(persistent_path)
    _write_prompt(runtime_path)
    _write_prompt(orphan_path)

    manager._registry["speaker"] = VoiceEntry(
        name="speaker",
        pt_path="persistent/speaker.pt",
        mode="xvec",
        persistent=True,
    )
    manager._registry["runtime"] = VoiceEntry(
        name="runtime",
        pt_path="cache/runtime.pt",
        mode="xvec",
        persistent=False,
        ttl=3600.0,
    )
    manager.flush()

    stats = manager.startup_cleanup(clear_runtime_cache=True)

    assert stats == {"removed_entries": 1, "removed_files": 1}
    assert "speaker" in manager._registry
    assert "runtime" not in manager._registry
    assert persistent_path.exists()
    assert not runtime_path.exists()
    assert not orphan_path.exists()


def test_startup_cleanup_removes_expired_runtime_cache_without_full_reset(tmp_path):
    manager = _make_manager(tmp_path)

    expired_path = tmp_path / "cache" / "expired.pt"
    active_path = tmp_path / "cache" / "active.pt"
    _write_prompt(expired_path)
    _write_prompt(active_path)

    now = voice_manager_module.time.time()
    manager._registry["expired"] = VoiceEntry(
        name="expired",
        pt_path="cache/expired.pt",
        mode="xvec",
        persistent=False,
        ttl=1.0,
        last_used_at=now - 10.0,
    )
    manager._registry["active"] = VoiceEntry(
        name="active",
        pt_path="cache/active.pt",
        mode="xvec",
        persistent=False,
        ttl=3600.0,
        last_used_at=now,
    )
    manager.flush()

    stats = manager.startup_cleanup(clear_runtime_cache=False)

    assert stats == {"removed_entries": 1, "removed_files": 0}
    assert "expired" not in manager._registry
    assert "active" in manager._registry
    assert not expired_path.exists()
    assert active_path.exists()
