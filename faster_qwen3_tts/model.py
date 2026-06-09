"""
FasterQwen3TTS: Real-time TTS using CUDA graph capture.

Wrapper class that provides a Qwen3-TTS API while using
CUDA graphs for 6-10x speedup.
"""
import collections
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from .utils import suppress_flash_attn_warning
from .voice_manager import MODE_ICL, MODE_XVEC

logger = logging.getLogger(__name__)


def _audio_to_numpy(audio) -> np.ndarray:
    """Convert codec decoder output to a flat numpy array.

    The speech tokenizer may return either a torch Tensor or a numpy array
    depending on the backend.  This helper normalises both to a flat 1-D
    numpy array so callers don't need a per-element type check.
    """
    if hasattr(audio, "cpu"):
        a = audio.flatten().cpu().numpy()
    else:
        a = np.asarray(audio).flatten()
    if not np.all(np.isfinite(a)):
        logger.warning("NaN/inf detected in decoded audio; replacing with silence")
        np.nan_to_num(a, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return a




class FasterQwen3TTS:
    """
    Qwen3-TTS model with CUDA graphs for real-time inference.
    
    Compatible API with Qwen3TTSModel, but uses CUDA graph
    capture for 6-10x speedup on NVIDIA GPUs.
    """
    
    def __init__(
        self,
        base_model,
        predictor_graph,
        talker_graph,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 2048,
    ):
        self.model = base_model  # The qwen-tts Qwen3TTSModel instance
        self.predictor_graph = predictor_graph
        self.talker_graph = talker_graph
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.sample_rate = self._infer_sample_rate(base_model)
        self._warmed_up = False
        # LRU cache for extracted voice prompts (GPU tensors).
        # Avoids re-extracting the same reference audio on repeated calls.
        self._voice_prompt_cache = collections.OrderedDict()
        # 64 entries ≈ a few MB of GPU memory; keeps the common "handful of
        # voices in rotation" use case entirely cache-hot.
        self._voice_prompt_cache_max = 64

    def _cache_voice_prompt(self, key, value):
        """Insert into LRU cache, evicting oldest entry if over limit."""
        self._voice_prompt_cache[key] = value
        while len(self._voice_prompt_cache) > self._voice_prompt_cache_max:
            self._voice_prompt_cache.popitem(last=False)

    @staticmethod
    def _voice_prompt_cache_key(
        ref_audio: Union[str, Path],
        ref_text: str,
        xvec_only: bool,
        append_silence: bool,
    ):
        path = Path(ref_audio)
        if path.suffix == ".pt":
            stat = path.stat()
            return (
                str(path.resolve()),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                ref_text,
            )
        return (str(ref_audio), ref_text, xvec_only, append_silence)

    def _prompt_dict_to_vcp(
        self,
        prompt_dict: dict,
        input_ids,
        ref_text: str = "",
    ) -> Tuple[Dict[str, Any], list, bool]:
        """Convert an ``extract_voice_prompt`` dict into runtime VCP format.

        This is the single conversion point used by both ``_load_voice_prompt_pt``
        and ``_resolve_voice_clone_prompt_from_reference`` so the two paths cannot
        diverge silently.

        Returns ``(vcp, ref_ids, using_icl_mode)``.
        """
        mode = prompt_dict.get("mode", MODE_XVEC)
        spk_emb = prompt_dict["ref_spk_embedding"]
        if spk_emb.device != torch.device(self.device) or spk_emb.dtype != self.dtype:
            spk_emb = spk_emb.to(device=self.device, dtype=self.dtype)

        if mode == MODE_XVEC:
            vcp = dict(
                ref_code=[None],
                ref_spk_embedding=[spk_emb],
                x_vector_only_mode=[True],
                icl_mode=[False],
            )
            return vcp, [None] * len(input_ids), False

        # ICL mode
        ref_code = prompt_dict.get("ref_code")
        if ref_code is None:
            raise ValueError("ICL-mode prompt dict has no ref_code")
        ref_code = ref_code.to(device=self.device)

        effective_ref_text = ref_text or prompt_dict.get("ref_text", "")
        if not effective_ref_text:
            raise ValueError(
                "ref_text is required for ICL mode but neither the prompt dict "
                "nor the call-site provided one."
            )

        vcp = dict(
            ref_code=[ref_code],
            ref_spk_embedding=[spk_emb],
            x_vector_only_mode=[False],
            icl_mode=[True],
        )
        ref_id = self.model._tokenize_texts(
            [self.model._build_ref_text(effective_ref_text)]
        )[0]
        return vcp, [ref_id], True

    def _decode_and_log(
        self,
        codec_ids,
        speech_tokenizer,
        timing: dict,
        ref_codes=None,
    ) -> Tuple[list, int]:
        """Decode codec tokens to audio waveform and log timing stats.

        Shared by the three non-streaming generate methods to avoid
        duplicating the decode → numpy → trim → log boilerplate.
        """
        if codec_ids is None:
            logger.warning("Generation returned no tokens")
            return [np.zeros(1, dtype=np.float32)], self.sample_rate

        if ref_codes is not None:
            codes_for_decode = torch.cat(
                [ref_codes.to(codec_ids.device), codec_ids], dim=0
            )
        else:
            codes_for_decode = codec_ids

        audio_list, sr = speech_tokenizer.decode(
            {"audio_codes": codes_for_decode.unsqueeze(0)}
        )

        ref_len = ref_codes.shape[0] if ref_codes is not None else 0
        total_len = codes_for_decode.shape[0]
        audio_arrays = []
        for a in audio_list:
            a = _audio_to_numpy(a)
            if ref_len > 0:
                cut = int(ref_len / max(total_len, 1) * len(a))
                a = a[cut:]
            audio_arrays.append(a)

        n_steps = timing["steps"]
        audio_duration = n_steps / 12.0  # 12 Hz codec
        total_time = timing["prefill_ms"] / 1000 + timing["decode_s"]
        rtf = audio_duration / total_time if total_time > 0 else 0
        logger.info(
            f"Generated {audio_duration:.2f}s audio in {total_time:.2f}s "
            f"({timing['ms_per_step']:.1f}ms/step, RTF: {rtf:.2f})"
        )

        return audio_arrays, sr

    @staticmethod
    def _streaming_decode_chunks(
        speech_tokenizer,
        codec_stream,
        chunk_size: int,
        ref_codes=None,
    ) -> "Generator[Tuple[np.ndarray, int, dict], None, None]":
        """Shared streaming decode loop for all three streaming generate methods.

        Two-phase hybrid strategy:
          Phase 1 (accumulated): decode all codes so far to get exact audio, while
          calibrating the samples-per-frame ratio.
          Phase 2 (sliding window): decode only the new chunk plus a fixed left
          context, using the calibrated ratio to trim context audio — keeps decode
          cost bounded regardless of total length.

        When *ref_codes* is provided (ICL voice-clone mode), reference codes are
        prepended during Phase 1 so the codec decoder has proper acoustic context,
        then the reference audio portion is trimmed from the output.
        """
        context_frames = 25
        min_calibration_frames = max(context_frames, chunk_size)
        all_codes = []
        prev_gen_audio_len = 0
        samples_per_frame = None

        for codec_chunk, timing in codec_stream:
            all_codes.append(codec_chunk)
            n_new = codec_chunk.shape[0]
            all_flat = torch.cat(all_codes, dim=0)
            n_total = all_flat.shape[0]

            # --- DEBUG: log codec token stats for this chunk ---
            cmin = codec_chunk.min().item()
            cmax = codec_chunk.max().item()
            if cmax >= 2048 or cmin < 0:
                logger.warning(
                    "OOB codec tokens in chunk %d: shape=%s min=%d max=%d",
                    timing.get('chunk_index', -1), codec_chunk.shape, cmin, cmax,
                )

            if samples_per_frame is None:
                # Phase 1: accumulated decode until we can calibrate.
                if ref_codes is not None:
                    codes_input = torch.cat(
                        [ref_codes.to(all_flat.device), all_flat], dim=0
                    )
                else:
                    codes_input = all_flat
                audio_list, sr = speech_tokenizer.decode(
                    {"audio_codes": codes_input.unsqueeze(0)}
                )
                audio = _audio_to_numpy(audio_list[0])

                # Trim reference audio portion if present
                if ref_codes is not None:
                    ref_len = ref_codes.shape[0]
                    total_len = codes_input.shape[0]
                    ref_audio_cut = int(ref_len / max(total_len, 1) * len(audio))
                    gen_audio = audio[ref_audio_cut:]
                else:
                    gen_audio = audio

                new_audio = gen_audio[prev_gen_audio_len:]
                prev_gen_audio_len = len(gen_audio)

                # --- DEBUG: log audio stats after Phase 1 decode ---
                if len(new_audio) > 0:
                    amax = float(np.max(np.abs(new_audio)))
                    nan_count = int(np.count_nonzero(~np.isfinite(new_audio)))
                    if nan_count > 0 or amax > 1.0:
                        logger.warning(
                            "Phase1 audio stats: len=%d max_abs=%.4f nan_count=%d",
                            len(new_audio), amax, nan_count,
                        )

                if n_total >= min_calibration_frames:
                    samples_per_frame = len(gen_audio) / n_total
            else:
                # Phase 2: sliding window with bounded context
                ctx_start = max(0, n_total - n_new - context_frames)
                window = all_flat[ctx_start:]
                n_ctx = window.shape[0] - n_new

                audio_list, sr = speech_tokenizer.decode(
                    {"audio_codes": window.unsqueeze(0)}
                )
                audio = _audio_to_numpy(audio_list[0])

                if n_ctx > 0:
                    ctx_samples = int(round(n_ctx * samples_per_frame))
                    new_audio = audio[ctx_samples:]
                else:
                    new_audio = audio

                # --- DEBUG: log audio stats after Phase 2 decode ---
                if len(new_audio) > 0:
                    amax = float(np.max(np.abs(new_audio)))
                    nan_count = int(np.count_nonzero(~np.isfinite(new_audio)))
                    if nan_count > 0 or amax > 1.0:
                        logger.warning(
                            "Phase2 audio stats: len=%d max_abs=%.4f nan_count=%d window=%d ctx=%d",
                            len(new_audio), amax, nan_count, window.shape[0], n_ctx,
                        )

            yield new_audio, sr, timing

    @staticmethod
    def _get_speech_tokenizer(base_model):
        """Return the nested qwen-tts speech tokenizer when available."""
        return getattr(getattr(base_model, "model", None), "speech_tokenizer", None)

    @property
    def speech_tokenizer(self):
        """Expose the codec decoder on the wrapper's public surface."""
        speech_tokenizer = self._get_speech_tokenizer(self.model)
        if speech_tokenizer is None:
            raise AttributeError("Underlying model does not expose a speech_tokenizer")
        return speech_tokenizer

    @staticmethod
    def _infer_sample_rate(base_model) -> int:
        """Infer output audio sample rate from qwen-tts internals."""
        # Qwen3-TTS model IDs include "12Hz", but that is codec frame-rate (tokens/s),
        # not waveform sampling rate. Generated audio is 24kHz.
        sample_rate = None

        speech_tokenizer = FasterQwen3TTS._get_speech_tokenizer(base_model)
        if speech_tokenizer is not None:
            sample_rate = getattr(speech_tokenizer, "sample_rate", None)

        if sample_rate is None:
            sample_rate = getattr(base_model, "sample_rate", None)

        if sample_rate is None:
            logger.warning(
                "Could not infer sample rate from base model; defaulting to 24000 Hz."
            )
            return 24000

        return int(sample_rate)
        
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = torch.bfloat16,
        attn_implementation: str = "sdpa",
        max_seq_len: int = 2048,
    ):
        """
        Load Qwen3-TTS model and prepare CUDA graphs.

        Args:
            model_name: Model path or HuggingFace Hub ID
            device: Device to use ("cuda" or "cpu")
            dtype: Data type for inference
            attn_implementation: Attention implementation ("sdpa" or "flash_attention_2")
            max_seq_len: Maximum sequence length for static cache
            
        Returns:
            FasterQwen3TTS instance
        """
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("CUDA graphs require CUDA device")
        
        logger.info(f"Loading Qwen3-TTS model: {model_name}")
        
        # Import here to avoid dependency issues (and suppress flash-attn warning)
        with suppress_flash_attn_warning():
            from .qwen_tts import Qwen3TTSModel
        from .predictor_graph import PredictorGraph
        from .talker_graph import TalkerGraph
        # Load base model using qwen-tts library
        base_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        
        talker = base_model.model.talker
        talker_config = base_model.model.config.talker_config

        # Extract predictor config from loaded model
        predictor = talker.code_predictor
        pred_config = predictor.model.config
        talker_hidden = talker_config.hidden_size

        # Build CUDA graphs
        logger.info("Building CUDA graphs...")
        predictor_graph = PredictorGraph(
            predictor,
            pred_config,
            talker_hidden,
            device=device,
            dtype=dtype,
            do_sample=True,
            top_k=50,
            temperature=0.9,
        )
        
        talker_graph = TalkerGraph(
            talker.model,
            talker_config,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )
        
        logger.info("CUDA graphs initialized (will capture on first run)")
        
        return cls(
            base_model=base_model,
            predictor_graph=predictor_graph,
            talker_graph=talker_graph,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )
    
    def _warmup(self, prefill_len: int):
        """Warm up and capture CUDA graphs with given prefill length."""
        if self._warmed_up:
            return
            
        logger.info("Warming up CUDA graphs...")
        self.predictor_graph.capture(num_warmup=3)
        self.talker_graph.capture(prefill_len=prefill_len, num_warmup=3)

        self._warmed_up = True
        logger.info("CUDA graphs captured and ready")
    
    def generate(
        self,
        text: str,
        language: str = "English",
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        """
        Generate speech from text using default voice.
        
        Not yet implemented - use generate_voice_clone() instead.
        """
        raise NotImplementedError(
            "Default voice generation not yet implemented. "
            "Use generate_voice_clone() with reference audio."
        )
    
    def _load_ref_audio_with_silence(self, ref_audio: Union[str, Path], silence_secs: float = 0.5) -> Tuple[np.ndarray, int]:
        """Load reference audio and optionally append trailing silence.

        The ICL voice-cloning prompt ends with the last codec token of the reference
        audio, so the model's first generated token is conditioned on whatever phoneme
        the reference ends with. Appending a short silence makes the last tokens
        encode silence instead, preventing that phoneme from bleeding into the start
        of the generated speech. Set silence_secs=0 to disable this behavior.
        """
        audio, sr = sf.read(str(ref_audio), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # convert to mono
        if silence_secs > 0:
            silence = np.zeros(int(silence_secs * sr), dtype=np.float32)
            audio = np.concatenate([audio, silence])
        return audio, sr

    def extract_voice_prompt(
        self,
        ref_audio: Union[str, Path],
        ref_text: str = "",
        xvec_only: bool = True,
        append_silence: bool = True,
    ) -> dict:
        """Extract a reusable voice prompt dict from reference audio.

        The returned dict can be saved with ``torch.save()`` and later loaded
        to skip the speaker encoder / audio tokenizer at inference time.

        Args:
            ref_audio: Path to reference audio file.
            ref_text: Transcription of the reference audio.  Required when
                ``xvec_only=False`` (ICL mode).
            xvec_only: When *True* extract only the speaker embedding
                (x-vector).  When *False* also extract codec tokens
                (``ref_code``) for full ICL voice cloning.
            append_silence: Append trailing silence before encoding
                (ICL only, prevents phoneme bleed-through).

        Returns:
            A dict ready for ``torch.save``:

            * xvec mode:  ``{version, mode, ref_spk_embedding}``
            * ICL mode:   ``{version, mode, ref_spk_embedding, ref_code, ref_text}``
        """
        if not xvec_only and not ref_text:
            raise ValueError("ref_text is required for ICL mode (xvec_only=False)")

        if xvec_only:
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text="",
                x_vector_only_mode=True,
            )
            return {
                "version": 1,
                "mode": MODE_XVEC,
                "ref_spk_embedding": prompt_items[0].ref_spk_embedding.cpu(),
            }

        # ICL mode
        silence_secs = 0.5 if append_silence else 0.0
        ref_audio_input = self._load_ref_audio_with_silence(ref_audio, silence_secs=silence_secs)
        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio_input,
            ref_text=ref_text,
        )
        item = prompt_items[0]
        return {
            "version": 1,
            "mode": MODE_ICL,
            "ref_spk_embedding": item.ref_spk_embedding.cpu(),
            "ref_code": item.ref_code.cpu(),
            "ref_text": item.ref_text or ref_text,
        }

    def _resolve_voice_clone_prompt(
        self,
        input_ids,
        ref_audio: Optional[Union[str, Path]],
        ref_text: str,
        xvec_only: bool,
        append_silence: bool,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[Any]]],
    ) -> Tuple[Dict[str, Any], list, bool]:
        """Resolve voice clone prompt data and return (prompt, ref_ids, using_icl_mode)."""
        if voice_clone_prompt is not None:
            return self._resolve_precomputed_voice_clone_prompt(
                input_ids=input_ids,
                ref_text=ref_text,
                voice_clone_prompt=voice_clone_prompt,
            )
        if ref_audio is None:
            raise ValueError("ref_audio is required when voice_clone_prompt is not provided")

        return self._resolve_voice_clone_prompt_from_reference(
            input_ids=input_ids,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
        )

    def _resolve_precomputed_voice_clone_prompt(
        self,
        input_ids,
        ref_text: str,
        voice_clone_prompt: Union[Dict[str, Any], List[Any]],
    ) -> Tuple[Dict[str, Any], list, bool]:
        if isinstance(voice_clone_prompt, list):
            if len(voice_clone_prompt) != len(input_ids):
                raise ValueError(
                    f"voice_clone_prompt must have length {len(input_ids)}, got {len(voice_clone_prompt)}"
                )

            vcp = self.model._prompt_items_to_voice_clone_prompt(voice_clone_prompt)
            ref_ids = []
            for item in voice_clone_prompt:
                if bool(item.icl_mode):
                    item_ref_text = item.ref_text if item.ref_text else ref_text
                    if not item_ref_text:
                        raise ValueError(
                            "ref_text is required when voice_clone_prompt uses ICL mode."
                        )
                    ref_id = self.model._tokenize_texts(
                        [self.model._build_ref_text(item_ref_text)]
                    )[0]
                    ref_ids.append(ref_id)
                else:
                    ref_ids.append(None)

            return vcp, ref_ids, any(vcp["icl_mode"])

        required_keys = ("ref_spk_embedding",)
        missing = [k for k in required_keys if k not in voice_clone_prompt]
        if missing:
            raise ValueError(
                f"voice_clone_prompt missing required keys: {missing}. "
                f"Expected keys: {list(required_keys)}"
            )

        list_keys = ("ref_spk_embedding", "x_vector_only_mode", "icl_mode", "ref_code")
        for key in list_keys:
            if key not in voice_clone_prompt:
                continue
            value = voice_clone_prompt[key]
            if not isinstance(value, list) or len(value) != len(input_ids):
                raise ValueError(
                    f"voice_clone_prompt[{key!r}] must be a list with length {len(input_ids)}"
                )

        xvec_modes = voice_clone_prompt.get("x_vector_only_mode", [True] * len(input_ids))
        if "icl_mode" in voice_clone_prompt:
            icl_modes = [bool(v) for v in voice_clone_prompt["icl_mode"]]
            for i, (xvec_mode, icl_mode) in enumerate(zip(xvec_modes, icl_modes)):
                if bool(xvec_mode) == bool(icl_mode):
                    raise ValueError(
                        f"voice_clone_prompt has inconsistent mode flags at index {i}: "
                        "x_vector_only_mode and icl_mode must be opposites"
                    )
        else:
            icl_modes = [not bool(v) for v in xvec_modes]

        ref_codes = voice_clone_prompt.get("ref_code", [None] * len(input_ids))
        for i, (xvec_mode, icl_mode, ref_code) in enumerate(zip(xvec_modes, icl_modes, ref_codes)):
            if bool(xvec_mode) and ref_code is not None:
                raise ValueError(
                    f"voice_clone_prompt index {i}: ref_code must be None in x_vector_only mode"
                )
            if bool(icl_mode) and ref_code is None:
                raise ValueError(
                    f"voice_clone_prompt index {i}: ref_code is required in ICL mode"
                )

        vcp = dict(
            ref_code=ref_codes,
            ref_spk_embedding=voice_clone_prompt["ref_spk_embedding"],
            x_vector_only_mode=[bool(v) for v in xvec_modes],
            icl_mode=[bool(v) for v in icl_modes],
        )
        using_icl_mode = any(vcp["icl_mode"])

        if using_icl_mode:
            if not ref_text:
                raise ValueError(
                    "ref_text is required when voice_clone_prompt uses ICL mode."
                )
            ref_texts = [self.model._build_ref_text(ref_text)]
            # NOTE: single ref_text is shared across all ICL items in the batch.
            ref_id = self.model._tokenize_texts(ref_texts)[0]
            ref_ids = [ref_id if is_icl else None for is_icl in vcp["icl_mode"]]
        else:
            ref_ids = [None] * len(input_ids)

        return vcp, ref_ids, using_icl_mode

    def _load_voice_prompt_pt(
        self,
        pt_path: Union[str, Path],
        input_ids,
        ref_text: str,
    ) -> Tuple[Dict[str, Any], list, bool]:
        """Load a .pt voice prompt file (unified or legacy format).

        Supports:
        - **Unified dict** (version 1): ``{version, mode, ref_spk_embedding, ...}``
        - **Legacy tensor**: a bare ``ref_spk_embedding`` tensor (treated as xvec).

        Returns ``(vcp, ref_ids, using_icl_mode)``.
        """
        # weights_only=True prevents arbitrary code execution via pickle
        data = torch.load(str(pt_path), map_location=self.device, weights_only=True)

        # Normalise legacy bare-tensor format into a dict
        if isinstance(data, torch.Tensor):
            data = {"mode": MODE_XVEC, "ref_spk_embedding": data}
        elif not isinstance(data, dict):
            raise ValueError(f"Unsupported .pt format: expected dict or Tensor, got {type(data).__name__}")

        return self._prompt_dict_to_vcp(data, input_ids, ref_text)

    def _resolve_voice_clone_prompt_from_reference(
        self,
        input_ids,
        ref_audio: Union[str, Path],
        ref_text: str,
        xvec_only: bool,
        append_silence: bool,
    ) -> Tuple[Dict[str, Any], list, bool]:
        cache_key = self._voice_prompt_cache_key(
            ref_audio,
            ref_text,
            xvec_only,
            append_silence,
        )
        if cache_key in self._voice_prompt_cache:
            self._voice_prompt_cache.move_to_end(cache_key)
            vcp, ref_ids = self._voice_prompt_cache[cache_key]
            using_icl_mode = any(vcp.get("icl_mode", [False]))
            return vcp, ref_ids, using_icl_mode

        # .pt files carry their own mode — ignore xvec_only flag
        if str(ref_audio).endswith('.pt'):
            vcp, ref_ids, using_icl_mode = self._load_voice_prompt_pt(
                pt_path=ref_audio, input_ids=input_ids, ref_text=ref_text,
            )
            self._cache_voice_prompt(cache_key, (vcp, ref_ids))
            return vcp, ref_ids, using_icl_mode

        # Extract voice prompt then convert to runtime VCP format.
        # Both steps are routed through single-point helpers so the
        # extraction logic cannot diverge from extract_voice_prompt().
        prompt_dict = self.extract_voice_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
        )
        vcp, ref_ids, using_icl_mode = self._prompt_dict_to_vcp(
            prompt_dict, input_ids, ref_text,
        )
        self._cache_voice_prompt(cache_key, (vcp, ref_ids))
        return vcp, ref_ids, using_icl_mode

    def _prepare_generation(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        language: str = "English",
        xvec_only: bool = False,
        non_streaming_mode: bool = False,
        append_silence: bool = True,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[Any]]] = None,
        instruct: Optional[str] = None,
    ):
        """Prepare inputs for generation (shared by streaming and non-streaming).

        Args:
            xvec_only: When True, use only the speaker embedding (x-vector) for voice
                cloning instead of the full ICL acoustic prompt. This prevents the model from
                continuing the reference audio's last phoneme and allows natural language switching.
                Default False to match upstream ICL behavior, where the full reference
                audio codec tokens are included in context.
            voice_clone_prompt: Optional precomputed prompt dict from
                `create_voice_clone_prompt`/`_prompt_items_to_voice_clone_prompt`.
                When provided, `xvec_only` is ignored. This path supports both:
                x-vector-only prompts (`ref_spk_embedding` only) and ICL prompts
                (`ref_spk_embedding` + `ref_code` + mode flags). `ref_text` is ignored
                for x-vector-only and required for ICL.
            instruct: Optional instruction string to guide generation style/language (e.g.
                "请用纯正广东话朗读"). Prepended as a user turn before the assistant TTS turn.
        """
        input_texts = [self.model._build_assistant_text(text)]
        input_ids = self.model._tokenize_texts(input_texts)

        instruct_ids = [None]
        if instruct:
            instruct_ids = [self.model._tokenize_texts([self.model._build_instruct_text(instruct)])[0]]

        vcp, ref_ids, using_icl_mode = self._resolve_voice_clone_prompt(
            input_ids=input_ids,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
            voice_clone_prompt=voice_clone_prompt,
        )

        if instruct and not using_icl_mode:
            logger.warning(
                "Base-model instruct with x-vector-only voice cloning is experimental. "
                "Upstream Qwen3-TTS itself does not follow instructions reliably in this "
                "mode. Prefer xvec_only=False (ICL mode) when using instruct for voice "
                "cloning."
            )

        m = self.model.model

        tie, tam, tth, tpe = self._build_talker_inputs_local(
            m=m,
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=vcp,
            languages=[language] if language is not None else ["Auto"],
            speakers=None,
            non_streaming_mode=non_streaming_mode,
            instruct_ids=instruct_ids,
        )

        if not self._warmed_up:
            self._warmup(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None

        # For ICL mode: return ref_codes so the decoder can use them as acoustic context
        ref_codes = None
        if using_icl_mode and vcp.get("ref_code") and vcp["ref_code"][0] is not None:
            ref_codes = vcp["ref_code"][0]

        return m, talker, config, tie, tam, tth, tpe, ref_codes

    def _prepare_generation_custom(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        instruct: Optional[str] = None,
        non_streaming_mode: bool = True,
    ):
        input_texts = [self.model._build_assistant_text(text)]
        input_ids = self.model._tokenize_texts(input_texts)

        instruct_ids = []
        if instruct is None or instruct == "":
            instruct_ids.append(None)
        else:
            instruct_ids.append(self.model._tokenize_texts([self.model._build_instruct_text(instruct)])[0])

        m = self.model.model
        tie, tam, tth, tpe = self._build_talker_inputs_local(
            m=m,
            input_ids=input_ids,
            ref_ids=[None],
            voice_clone_prompt=None,
            languages=[language] if language is not None else ["Auto"],
            speakers=[speaker],
            non_streaming_mode=non_streaming_mode,
            instruct_ids=instruct_ids,
        )

        if not self._warmed_up:
            self._warmup(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None

        return m, talker, config, tie, tam, tth, tpe

    def _build_talker_inputs_local(
        self,
        m,
        input_ids,
        ref_ids,
        voice_clone_prompt,
        languages,
        speakers,
        non_streaming_mode: bool,
        instruct_ids=None,
    ):
        """Local copy of upstream talker input building for qwen-tts main repo."""
        talker_input_embeds = [[] for _ in range(len(input_ids))]

        voice_clone_spk_embeds = None
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = m.generate_speaker_prompt(voice_clone_prompt)

        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(
                        m.talker.text_projection(m.talker.get_text_embeddings()(instruct_id))
                    )

        if speakers is None:
            speakers = [None] * len(input_ids)

        trailing_text_hiddens = []
        tts_pad_embed = None

        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:
                    speaker_embed = None
                else:
                    if speaker.lower() not in m.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    spk_id = m.config.talker_config.spk_id[speaker.lower()]
                    speaker_embed = m.talker.get_input_embeddings()(
                        torch.tensor(spk_id, device=m.talker.device, dtype=input_id.dtype)
                    )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            assert language is not None
            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in m.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                language_id = m.config.talker_config.codec_language_id[language.lower()]

            if (
                language.lower() in ["chinese", "auto"]
                and speaker not in ("", None)
                and m.config.talker_config.spk_is_dialect[speaker.lower()]
            ):
                dialect = m.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = m.config.talker_config.codec_language_id[dialect]

            tts_bos_embed, tts_eos_embed, tts_pad_embed = m.talker.text_projection(
                m.talker.get_text_embeddings()(
                    torch.tensor(
                        [[m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id]],
                        device=m.talker.device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)

            if language_id is None:
                codec_prefill_list = [[
                    m.config.talker_config.codec_nothink_id,
                    m.config.talker_config.codec_think_bos_id,
                    m.config.talker_config.codec_think_eos_id,
                ]]
            else:
                codec_prefill_list = [[
                    m.config.talker_config.codec_think_id,
                    m.config.talker_config.codec_think_bos_id,
                    language_id,
                    m.config.talker_config.codec_think_eos_id,
                ]]

            codec_input_emebdding_0 = m.talker.get_input_embeddings()(
                torch.tensor(codec_prefill_list, device=m.talker.device, dtype=input_id.dtype)
            )
            codec_input_emebdding_1 = m.talker.get_input_embeddings()(
                torch.tensor(
                    [[m.config.talker_config.codec_pad_id, m.config.talker_config.codec_bos_id]],
                    device=m.talker.device,
                    dtype=input_id.dtype,
                )
            )
            if speaker_embed is None:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
            else:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, speaker_embed.view(1, 1, -1), codec_input_emebdding_1], dim=1)

            _talker_input_embed_role = m.talker.text_projection(
                m.talker.get_text_embeddings()(input_id[:, :3])
            )
            _talker_input_embed = torch.cat(
                (
                    tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
                    tts_bos_embed,
                ),
                dim=1,
            ) + codec_input_emebdding[:, :-1]

            talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

            if (
                voice_clone_prompt is not None
                and voice_clone_prompt.get("ref_code", None) is not None
                and voice_clone_prompt["icl_mode"][index]
            ):
                icl_input_embed, trailing_text_hidden = m.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(m.talker.device).clone(),  # escape inference_mode context
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        m.talker.text_projection(
                            m.talker.get_text_embeddings()(input_id[:, 3:4])
                        )
                        + codec_input_emebdding[:, -1:],
                    ],
                    dim=1,
                )
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            torch.cat(
                                (
                                    m.talker.text_projection(
                                        m.talker.get_text_embeddings()(input_id[:, 3:-5])
                                    ),
                                    tts_eos_embed,
                                ),
                                dim=1,
                            )
                            + m.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[m.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                                    device=m.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                            tts_pad_embed
                            + m.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[m.config.talker_config.codec_bos_id]],
                                    device=m.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    trailing_text_hidden = torch.cat(
                        (
                            m.talker.text_projection(
                                m.talker.get_text_embeddings()(input_id[:, 4:-5])
                            ),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )

            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed,
            batch_first=True,
            padding_value=0.0,
        )
        talker_input_embeds = padded_reversed.flip(dims=[1])

        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)

        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(
            sequences_to_pad,
            batch_first=True,
            padding_value=0.0,
        )
        arange_tensor = torch.arange(max(trailing_text_original_lengths), device=padded_hiddens.device).expand(
            len(trailing_text_original_lengths), -1
        )
        lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens

        return talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed

    @torch.inference_mode()
    def generate_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        xvec_only: bool = False,
        non_streaming_mode: bool = False,
        append_silence: bool = True,
        instruct: Optional[str] = None,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[Any]]] = None,
        eos_logit_bias: float = 0.0,
    ) -> Tuple[list, int]:
        """
        Generate speech with voice cloning using reference audio.

        Args:
            text: Text to synthesize
            language: Target language
            ref_audio: Path to reference audio file. Required when `voice_clone_prompt` is not provided.
            ref_text: Transcription of reference audio.
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens before EOS is allowed
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample
            repetition_penalty: Repetition penalty
            xvec_only: When True, use only the speaker embedding for voice cloning.
                This prevents phoneme bleed-through from the reference and allows clean
                language switching. Default False to match upstream ICL behavior
                (reference audio in context).
            non_streaming_mode: Match upstream text-feeding layout. Default False to match
                upstream step-by-step text feeding during decode.
            voice_clone_prompt: Optional precomputed voice clone prompt dict. When provided,
                `xvec_only` is ignored and prompt extraction from `ref_audio` is skipped.
                This path supports x-vector-only prompts (`ref_spk_embedding` only)
                and ICL prompts (`ref_spk_embedding` + `ref_code` + mode flags).
                `ref_text` is ignored for x-vector-only and required for ICL.
            instruct: Optional instruction to guide generation style/dialect (e.g.
                "请用纯正广东话朗读"). Prepended as a user turn before the TTS assistant turn.
                Experimental for x-vector-only voice cloning; prefer `xvec_only=False`.

        Returns:
            Tuple of ([audio_waveform], sample_rate)
        """
        from .generation import fast_generate

        m, talker, config, tie, tam, tth, tpe, ref_codes = self._prepare_generation(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            non_streaming_mode=non_streaming_mode,
            append_silence=append_silence,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
        )

        codec_ids, timing = fast_generate(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            eos_logit_bias=eos_logit_bias,
        )

        return self._decode_and_log(
            codec_ids, m.speech_tokenizer, timing, ref_codes=ref_codes,
        )

    @torch.inference_mode()
    def generate_voice_clone_streaming(
        self,
        text: str,
        language: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
        xvec_only: bool = False,
        non_streaming_mode: bool = False,
        append_silence: bool = True,
        instruct: Optional[str] = None,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[Any]]] = None,
        eos_logit_bias: float = 0.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        """
        Stream voice-cloned speech generation, yielding audio chunks.

        Same as generate_voice_clone() but yields (audio_chunk, sample_rate, timing)
        tuples every chunk_size codec steps (~chunk_size/12 seconds of audio).

        Args:
            text: Text to synthesize
            language: Target language
            ref_audio: Path to reference audio file. Required when `voice_clone_prompt` is not provided.
            ref_text: Transcription of reference audio.
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens before EOS is allowed
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample
            repetition_penalty: Repetition penalty
            chunk_size: Codec steps per chunk (12 = ~1 second)
            xvec_only: When True, use only the speaker embedding for voice cloning.
                This prevents phoneme bleed-through from the reference and allows clean
                language switching. Default False to match upstream ICL behavior
                (reference audio in context).
            non_streaming_mode: Default False to match upstream text feeding during decode.
                Set to True to prefill the full target text before streaming decode.
            voice_clone_prompt: Optional precomputed voice clone prompt dict. When provided,
                `xvec_only` is ignored and prompt extraction from `ref_audio` is skipped.
                This path supports x-vector-only prompts (`ref_spk_embedding` only)
                and ICL prompts (`ref_spk_embedding` + `ref_code` + mode flags).
                `ref_text` is ignored for x-vector-only and required for ICL.
            instruct: Optional instruction to guide generation style/dialect (e.g.
                "请用纯正广东话朗读"). Prepended as a user turn before the TTS assistant turn.
                Experimental for x-vector-only voice cloning; prefer `xvec_only=False`.

        Yields:
            Tuple of (audio_chunk_numpy, sample_rate, timing_dict)
        """
        from .generation import fast_generate_streaming

        import time as _time
        _t_prep_start = _time.monotonic()
        m, talker, config, tie, tam, tth, tpe, ref_codes = self._prepare_generation(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            non_streaming_mode=non_streaming_mode,
            append_silence=append_silence,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
        )
        _t_prep = _time.monotonic() - _t_prep_start

        speech_tokenizer = m.speech_tokenizer
        logger.debug("prepare_generation: %.1fms | tie=%s tth=%s",
                      _t_prep * 1000, tie.shape, tth.shape)

        if cancel_event is not None and cancel_event.is_set():
            return

        yield from self._streaming_decode_chunks(
            speech_tokenizer,
            fast_generate_streaming(
                talker=talker,
                talker_input_embeds=tie,
                attention_mask=tam,
                trailing_text_hiddens=tth,
                tts_pad_embed=tpe,
                config=config,
                predictor_graph=self.predictor_graph,
                talker_graph=self.talker_graph,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                chunk_size=chunk_size,
                eos_logit_bias=eos_logit_bias,
                cancel_event=cancel_event,
            ),
            chunk_size,
            ref_codes=ref_codes,
        )

    # TODO: remove custom voice task.
    @torch.inference_mode()
    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        non_streaming_mode: bool = True,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        if self.model.model.tts_model_type != "custom_voice":
            raise ValueError("Loaded model does not support custom voice generation")

        self.model._validate_languages([language])
        self.model._validate_speakers([speaker])

        if self.model.model.tts_model_size in "0b6":
            instruct = None

        from .generation import fast_generate

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation_custom(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
        )

        codec_ids, timing = fast_generate(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

        return self._decode_and_log(codec_ids, m.speech_tokenizer, timing)

    # TODO: remove custom voice task.
    @torch.inference_mode()
    def generate_custom_voice_streaming(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        non_streaming_mode: bool = True,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        if self.model.model.tts_model_type != "custom_voice":
            raise ValueError("Loaded model does not support custom voice generation")

        self.model._validate_languages([language])
        self.model._validate_speakers([speaker])

        if self.model.model.tts_model_size in "0b6":
            instruct = None

        from .generation import fast_generate_streaming

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation_custom(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
        )

        speech_tokenizer = m.speech_tokenizer

        yield from self._streaming_decode_chunks(
            speech_tokenizer,
            fast_generate_streaming(
                talker=talker,
                talker_input_embeds=tie,
                attention_mask=tam,
                trailing_text_hiddens=tth,
                tts_pad_embed=tpe,
                config=config,
                predictor_graph=self.predictor_graph,
                talker_graph=self.talker_graph,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                chunk_size=chunk_size,
            ),
            chunk_size,
        )

    # TODO: remove voice design task.
    @torch.inference_mode()
    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str,
        non_streaming_mode: bool = True,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        if self.model.model.tts_model_type != "voice_design":
            raise ValueError("Loaded model does not support voice design generation")

        self.model._validate_languages([language])

        from .generation import fast_generate

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation_custom(
            text=text,
            language=language,
            speaker=None,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
        )

        codec_ids, timing = fast_generate(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

        return self._decode_and_log(codec_ids, m.speech_tokenizer, timing)

    # TODO: remove voice design task.
    @torch.inference_mode()
    def generate_voice_design_streaming(
        self,
        text: str,
        instruct: str,
        language: str,
        non_streaming_mode: bool = True,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        if self.model.model.tts_model_type != "voice_design":
            raise ValueError("Loaded model does not support voice design generation")

        self.model._validate_languages([language])

        from .generation import fast_generate_streaming

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation_custom(
            text=text,
            language=language,
            speaker=None,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
        )

        speech_tokenizer = m.speech_tokenizer

        yield from self._streaming_decode_chunks(
            speech_tokenizer,
            fast_generate_streaming(
                talker=talker,
                talker_input_embeds=tie,
                attention_mask=tam,
                trailing_text_hiddens=tth,
                tts_pad_embed=tpe,
                config=config,
                predictor_graph=self.predictor_graph,
                talker_graph=self.talker_graph,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                chunk_size=chunk_size,
            ),
            chunk_size,
        )
