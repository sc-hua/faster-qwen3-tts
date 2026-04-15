"""CUDA Graph wrapper for codec decoder (speech_tokenizer)."""
import torch
from torch.cuda import CUDAGraph

import logging
logger = logging.getLogger(__name__)


class CodecGraphDecoder:
    """Wraps the Qwen3TTSTokenizerV2Decoder with a single CUDA graph.

    Uses ONE capture at the largest expected size and zero-pads smaller
    inputs.  This avoids cross-graph memory aliasing that can occur when
    multiple CUDA graphs are captured for the same model (different sizes
    share intermediate buffer addresses, and replaying one graph corrupts
    another's state permanently).
    """

    def __init__(self, decoder: torch.nn.Module, max_size: int = 50):
        self.decoder = decoder
        self.num_quantizers = decoder.config.num_quantizers
        self.total_upsample = decoder.total_upsample
        self.max_size = max_size
        self.graph: CUDAGraph | None = None
        self.static_input: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None

    @torch.inference_mode()
    def capture(self, device: torch.device, num_warmup: int = 3):
        self.decoder.eval()
        size = self.max_size
        dummy = torch.zeros(1, self.num_quantizers, size, dtype=torch.long, device=device)
        for _ in range(num_warmup):
            self.decoder(dummy)
        torch.cuda.synchronize(device)

        static_input = torch.zeros(1, self.num_quantizers, size, dtype=torch.long, device=device)
        self.decoder(static_input)
        torch.cuda.synchronize(device)

        graph = CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.decoder(static_input)

        self.graph = graph
        self.static_input = static_input
        self.static_output = static_output
        logger.info("Captured single codec graph for max_size=%d", size)

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        actual = codes.shape[-1]

        if self.graph is None or actual > self.max_size:
            return self.decoder(codes)

        self.static_input.zero_()
        self.static_input[:, :, :actual] = codes
        self.graph.replay()

        actual_len = actual * self.total_upsample
        return self.static_output[..., :actual_len].clone()

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            ctx = min(left_context_size, start_index)
            chunk = codes[..., start_index - ctx:end_index]
            wav_chunk = self(chunk)
            wavs.append(wav_chunk[..., ctx * self.total_upsample:])
            start_index = end_index
        return torch.cat(wavs, dim=-1)
