import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _install_api_server_stubs():
    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return dict(self.__dict__)

    def Field(default=None, **kwargs):
        return default

    def model_validator(*args, **kwargs):
        return lambda fn: fn

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    pydantic.model_validator = model_validator
    sys.modules["pydantic"] = pydantic

    fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def __init__(self, *args, **kwargs):
            self.middleware = []

        def add_middleware(self, *args, **kwargs):
            self.middleware.append((args, kwargs))

        def post(self, *args, **kwargs):
            return lambda fn: fn

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def delete(self, *args, **kwargs):
            return lambda fn: fn

        def exception_handler(self, *args, **kwargs):
            return lambda fn: fn

    class UploadFile:
        filename = "upload.wav"

        async def read(self):
            return b""

    class Request:
        pass

    def File(default=None, **kwargs):
        return default

    def Form(default=None, **kwargs):
        return default

    fastapi.FastAPI = FastAPI
    fastapi.HTTPException = HTTPException
    fastapi.File = File
    fastapi.Form = Form
    fastapi.Request = Request
    fastapi.UploadFile = UploadFile
    sys.modules["fastapi"] = fastapi

    responses = types.ModuleType("fastapi.responses")

    class Response:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class JSONResponse(Response):
        pass

    class StreamingResponse(Response):
        pass

    responses.Response = Response
    responses.JSONResponse = JSONResponse
    responses.StreamingResponse = StreamingResponse
    sys.modules["fastapi.responses"] = responses

    middleware = types.ModuleType("fastapi.middleware")
    cors = types.ModuleType("fastapi.middleware.cors")

    class CORSMiddleware:
        pass

    cors.CORSMiddleware = CORSMiddleware
    sys.modules["fastapi.middleware"] = middleware
    sys.modules["fastapi.middleware.cors"] = cors

    soundfile = types.ModuleType("soundfile")
    soundfile.info = lambda path: types.SimpleNamespace(duration=2.0)
    sys.modules["soundfile"] = soundfile

    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: None
    sys.modules["uvicorn"] = uvicorn

    audio_utils = types.ModuleType("faster_qwen3_tts.audio_utils")
    audio_utils.audio_to_pcm16_bytes = lambda audio: b""
    audio_utils.create_wav_header = lambda sr: b""
    audio_utils.encode_audio = lambda audio, sr, fmt: b""
    audio_utils.media_type = lambda fmt: "audio/wav"

    package = types.ModuleType("faster_qwen3_tts")
    package.audio_utils = audio_utils
    sys.modules["faster_qwen3_tts"] = package
    sys.modules["faster_qwen3_tts.audio_utils"] = audio_utils


def _load_api_server_module():
    _install_api_server_stubs()
    spec = importlib.util.spec_from_file_location(
        "api_server_under_test",
        ROOT / "api_server.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


api_server = _load_api_server_module()


class DummyVoiceManager:
    def __init__(self):
        self.calls = []

    def startup_cleanup(self, clear_runtime_cache=False):
        self.calls.append(clear_runtime_cache)
        return {"removed_entries": 2, "removed_files": 3}


class DummyModel:
    def __init__(self, warmed_up=False):
        self._warmed_up = warmed_up
        self.prefill_lens = []

    def _warmup(self, prefill_len):
        self.prefill_lens.append(prefill_len)
        self._warmed_up = True


def test_prepare_server_runtime_cleans_runtime_cache_and_warms_model():
    voice_manager = DummyVoiceManager()
    model = DummyModel()

    api_server._voice_manager = voice_manager
    api_server._model = model

    api_server._prepare_server_runtime()

    assert voice_manager.calls == [True]
    assert model.prefill_lens == [100]


def test_prepare_server_runtime_skips_duplicate_warmup():
    voice_manager = DummyVoiceManager()
    model = DummyModel(warmed_up=True)

    api_server._voice_manager = voice_manager
    api_server._model = model

    api_server._prepare_server_runtime()

    assert voice_manager.calls == [True]
    assert model.prefill_lens == []
