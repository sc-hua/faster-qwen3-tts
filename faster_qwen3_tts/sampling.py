"""Shared sampling helpers for talker and predictor generation."""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F


def validate_sampling_config(
    predictor_graph,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
) -> None:
    """Reject sampling settings that differ from the captured predictor graph."""
    expected = predictor_graph.sampling_config
    actual = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
    }
    if actual != expected:
        raise ValueError(
            "Sampling parameters are fixed when the predictor CUDA graph is captured; "
            f"requested {actual}, captured {expected}."
        )


def build_codec_suppress_mask(
    vocab_size: int,
    codebook_vocab_size: int,
    eos_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a boolean mask that suppresses all tokens outside [1, codebook_vocab_size) ∪ {eos_id}."""
    allowed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    hi = min(codebook_vocab_size, vocab_size)
    if hi > 1:
        allowed[1:hi] = True
    if 0 <= eos_id < vocab_size:
        allowed[eos_id] = True
    return ~allowed


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits in-place and return them.

    Args:
        logits: Tensor shaped [1, 1, vocab] or [1, vocab].
        token_history: 1-D tensor of previously generated token ids.
        repetition_penalty: HF-style repetition penalty (>1.0).
    """
    if repetition_penalty == 1.0 or token_history.numel() == 0:
        return logits
    unique_toks = token_history.unique()
    tok_logits = logits[..., unique_toks]
    logits[..., unique_toks] = torch.where(
        tok_logits > 0, tok_logits / repetition_penalty, tok_logits * repetition_penalty
    )
    return logits


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    suppress_mask: Optional[torch.Tensor] = None,
    suppress_tokens: Optional[Iterable[int]] = None,
    eos_logit_bias: float = 0.0,
    eos_id: int = -1,
) -> torch.Tensor:
    """Sample a token from logits.

    Mirrors HF order: suppress -> eos_bias -> temperature -> top-k -> top-p -> sample.
    """
    logits = logits.clone()
    if suppress_mask is not None:
        logits[..., suppress_mask] = float("-inf")
    if suppress_tokens:
        logits[..., list(suppress_tokens)] = float("-inf")
    if eos_logit_bias != 0.0 and 0 <= eos_id < logits.shape[-1]:
        logits[..., eos_id] = logits[..., eos_id] + eos_logit_bias
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0 when do_sample=True")
    logits = logits / temperature
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < topk_vals[..., -1:], torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
