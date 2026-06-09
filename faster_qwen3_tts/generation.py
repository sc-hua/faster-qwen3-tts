"""Shared CUDA Graph generation state machine."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Generator, Optional, Tuple

import torch

from .predictor_graph import PredictorGraph
from .sampling import (
    apply_repetition_penalty,
    build_codec_suppress_mask,
    sample_logits,
    validate_sampling_config,
)
from .talker_graph import TalkerGraph


@dataclass
class FastGenerationState:
    talker_codec_embed: object
    talker_codec_head: object
    predictor_codec_embeds: object
    predictor_graph: PredictorGraph
    talker_graph: TalkerGraph
    trailing_text_hiddens: torch.Tensor
    tts_pad_embed: torch.Tensor
    suppress_mask: torch.Tensor
    eos_id: int
    codebook_vocab_size: int
    min_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    do_sample: bool
    repetition_penalty: float
    eos_logit_bias: float
    token: torch.Tensor
    past_hidden: torch.Tensor
    gen_step: int
    prefill_len: int
    prefill_ms: float
    first_tokens: list[torch.Tensor] = field(default_factory=list)


def prepare_fast_generation(
    *,
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: PredictorGraph,
    talker_graph: TalkerGraph,
    min_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
    eos_logit_bias: float,
) -> FastGenerationState:
    """Run prefill and initialize the shared CUDA Graph decode state."""
    validate_sampling_config(
        predictor_graph,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
    )

    eos_id = config.codec_eos_token_id
    codebook_vocab_size = config.code_predictor_config.vocab_size
    suppress_mask = build_codec_suppress_mask(
        config.vocab_size,
        codebook_vocab_size,
        eos_id,
        talker_input_embeds.device,
    )
    predictor = talker.code_predictor

    started_at = time.time()
    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
    )
    token = sample_logits(
        out.logits[:, -1, :],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        suppress_mask=suppress_mask,
        suppress_tokens=[eos_id] if min_new_tokens > 0 else None,
        eos_logit_bias=eos_logit_bias,
        eos_id=eos_id,
    )

    prefill_len = talker_graph.prefill_kv(out.past_key_values)
    talker_graph.set_generation_state(
        attention_mask,
        getattr(talker, "rope_deltas", None),
    )
    torch.cuda.synchronize()

    return FastGenerationState(
        talker_codec_embed=talker.get_input_embeddings(),
        talker_codec_head=talker.codec_head,
        predictor_codec_embeds=predictor.get_input_embeddings(),
        predictor_graph=predictor_graph,
        talker_graph=talker_graph,
        trailing_text_hiddens=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        suppress_mask=suppress_mask,
        eos_id=eos_id,
        codebook_vocab_size=codebook_vocab_size,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        eos_logit_bias=eos_logit_bias,
        token=token,
        past_hidden=out.past_hidden,
        gen_step=out.generation_step,
        prefill_len=prefill_len,
        prefill_ms=(time.time() - started_at) * 1000,
    )


def iter_fast_codec_frames(
    state: FastGenerationState,
    *,
    max_new_tokens: int,
    cancel_event: Optional[threading.Event] = None,
) -> Generator[torch.Tensor, None, None]:
    """Yield one complete codec frame per CUDA Graph decode step."""
    for step_idx in range(max_new_tokens):
        if state.token.item() == state.eos_id:
            return
        if cancel_event is not None and cancel_event.is_set():
            return

        last_id_hidden = state.talker_codec_embed(state.token.unsqueeze(1))
        pred_input = torch.cat((state.past_hidden, last_id_hidden), dim=1)
        codebook_token_ids = state.predictor_graph.run(pred_input)

        codec_frame = torch.cat([state.token.view(1), codebook_token_ids])
        if not (1 <= state.token.item() < state.codebook_vocab_size):
            codec_frame.zero_()
        codec_frame = codec_frame.detach()
        state.first_tokens.append(state.token.detach())

        current_pos = state.prefill_len + step_idx
        if current_pos >= state.talker_graph.max_seq_len - 1:
            yield codec_frame
            return

        codec_hiddens = [last_id_hidden]
        for index, embedding in enumerate(state.predictor_codec_embeds):
            codec_hiddens.append(
                embedding(codebook_token_ids[index].unsqueeze(0).unsqueeze(0))
            )
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
        if state.gen_step < state.trailing_text_hiddens.shape[1]:
            inputs_embeds = (
                inputs_embeds
                + state.trailing_text_hiddens[:, state.gen_step].unsqueeze(1)
            )
        else:
            inputs_embeds = inputs_embeds + state.tts_pad_embed

        hidden_states = state.talker_graph.run(inputs_embeds, position=current_pos)
        logits = state.talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
        if state.repetition_penalty != 1.0:
            history = torch.stack(state.first_tokens)
            logits = apply_repetition_penalty(
                logits,
                history,
                state.repetition_penalty,
            )

        state.token = sample_logits(
            logits.squeeze(0),
            temperature=state.temperature,
            top_k=state.top_k,
            top_p=state.top_p,
            do_sample=state.do_sample,
            suppress_mask=state.suppress_mask,
            suppress_tokens=(
                [state.eos_id]
                if len(state.first_tokens) < state.min_new_tokens
                else None
            ),
            eos_logit_bias=state.eos_logit_bias,
            eos_id=state.eos_id,
        )
        state.past_hidden = hidden_states[:, -1:, :].clone()
        state.gen_step += 1
        yield codec_frame


@torch.inference_mode()
def fast_generate(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: PredictorGraph,
    talker_graph: TalkerGraph,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    eos_logit_bias: float = 0.0,
) -> Tuple[Optional[torch.Tensor], dict]:
    """Generate and collect all codec frames."""
    state = prepare_fast_generation(
        talker=talker,
        talker_input_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        trailing_text_hiddens=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        config=config,
        predictor_graph=predictor_graph,
        talker_graph=talker_graph,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        eos_logit_bias=eos_logit_bias,
    )

    started_at = time.time()
    frames = list(
        iter_fast_codec_frames(
            state,
            max_new_tokens=max_new_tokens,
        )
    )
    torch.cuda.synchronize()
    decode_s = time.time() - started_at
    steps = len(frames)
    timing = {
        "prefill_ms": state.prefill_ms,
        "decode_s": decode_s,
        "steps": steps,
        "ms_per_step": (decode_s / steps * 1000) if steps else 0,
        "steps_per_s": (steps / decode_s) if decode_s > 0 else 0,
    }
    return (torch.stack(frames) if frames else None), timing


@torch.inference_mode()
def fast_generate_streaming(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: PredictorGraph,
    talker_graph: TalkerGraph,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    chunk_size: int = 12,
    eos_logit_bias: float = 0.0,
    cancel_event: Optional[threading.Event] = None,
) -> Generator[Tuple[torch.Tensor, dict], None, None]:
    """Yield generated codec frames in fixed-size chunks."""
    if cancel_event is not None and cancel_event.is_set():
        return

    state = prepare_fast_generation(
        talker=talker,
        talker_input_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        trailing_text_hiddens=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        config=config,
        predictor_graph=predictor_graph,
        talker_graph=talker_graph,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        eos_logit_bias=eos_logit_bias,
    )

    buffer = []
    total_steps = 0
    chunk_index = 0
    chunk_started_at = time.time()
    for frame in iter_fast_codec_frames(
        state,
        max_new_tokens=max_new_tokens,
        cancel_event=cancel_event,
    ):
        buffer.append(frame)
        if len(buffer) < chunk_size:
            continue

        torch.cuda.synchronize()
        total_steps += len(buffer)
        yield torch.stack(buffer), {
            "chunk_index": chunk_index,
            "chunk_steps": len(buffer),
            "prefill_ms": state.prefill_ms if chunk_index == 0 else 0,
            "decode_ms": (time.time() - chunk_started_at) * 1000,
            "total_steps_so_far": total_steps,
            "is_final": False,
        }
        buffer = []
        chunk_index += 1
        chunk_started_at = time.time()

    if buffer:
        torch.cuda.synchronize()
        total_steps += len(buffer)
        yield torch.stack(buffer), {
            "chunk_index": chunk_index,
            "chunk_steps": len(buffer),
            "prefill_ms": state.prefill_ms if chunk_index == 0 else 0,
            "decode_ms": (time.time() - chunk_started_at) * 1000,
            "total_steps_so_far": total_steps,
            "is_final": True,
        }
