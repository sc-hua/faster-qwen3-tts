import inspect
import threading
import types

import torch

from faster_qwen3_tts import generation_loop


def _state():
    return types.SimpleNamespace(prefill_ms=12.5)


def _frames(count):
    for index in range(count):
        yield torch.tensor([index, index + 1], dtype=torch.long)


def test_production_generation_api_has_no_parity_mode():
    assert (
        "parity_mode"
        not in inspect.signature(generation_loop.fast_generate).parameters
    )
    assert (
        "parity_mode"
        not in inspect.signature(generation_loop.fast_generate_streaming).parameters
    )


def test_non_streaming_collects_shared_codec_frames(monkeypatch):
    monkeypatch.setattr(
        generation_loop,
        "prepare_fast_generation",
        lambda **kwargs: _state(),
    )
    monkeypatch.setattr(
        generation_loop,
        "iter_fast_codec_frames",
        lambda state, **kwargs: _frames(3),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    codec_ids, timing = generation_loop.fast_generate(
        talker=None,
        talker_input_embeds=torch.zeros(1),
        attention_mask=torch.zeros(1),
        trailing_text_hiddens=torch.zeros(1),
        tts_pad_embed=torch.zeros(1),
        config=None,
        predictor_graph=None,
        talker_graph=None,
    )

    assert codec_ids.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert timing["prefill_ms"] == 12.5
    assert timing["steps"] == 3


def test_streaming_chunks_shared_codec_frames(monkeypatch):
    monkeypatch.setattr(
        generation_loop,
        "prepare_fast_generation",
        lambda **kwargs: _state(),
    )
    monkeypatch.setattr(
        generation_loop,
        "iter_fast_codec_frames",
        lambda state, **kwargs: _frames(5),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    chunks = list(
        generation_loop.fast_generate_streaming(
            talker=None,
            talker_input_embeds=torch.zeros(1),
            attention_mask=torch.zeros(1),
            trailing_text_hiddens=torch.zeros(1),
            tts_pad_embed=torch.zeros(1),
            config=None,
            predictor_graph=None,
            talker_graph=None,
            chunk_size=2,
        )
    )

    assert [chunk.tolist() for chunk, _ in chunks] == [
        [[0, 1], [1, 2]],
        [[2, 3], [3, 4]],
        [[4, 5]],
    ]
    assert [info["is_final"] for _, info in chunks] == [False, False, True]
    assert chunks[0][1]["prefill_ms"] == 12.5
    assert chunks[-1][1]["total_steps_so_far"] == 5


def test_streaming_skips_prefill_when_already_cancelled(monkeypatch):
    called = False

    def _prepare(**kwargs):
        nonlocal called
        called = True
        return _state()

    monkeypatch.setattr(generation_loop, "prepare_fast_generation", _prepare)
    cancel_event = threading.Event()
    cancel_event.set()

    chunks = list(
        generation_loop.fast_generate_streaming(
            talker=None,
            talker_input_embeds=torch.zeros(1),
            attention_mask=torch.zeros(1),
            trailing_text_hiddens=torch.zeros(1),
            tts_pad_embed=torch.zeros(1),
            config=None,
            predictor_graph=None,
            talker_graph=None,
            cancel_event=cancel_event,
        )
    )

    assert chunks == []
    assert not called
