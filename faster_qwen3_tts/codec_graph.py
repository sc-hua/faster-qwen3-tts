"""CUDA Graph wrapper for codec decoder (speech_tokenizer)."""
import torch
from torch.cuda import CUDAGraph

import logging
logger = logging.getLogger(__name__)


class CodecGraphDecoder:
    """Wraps the Qwen3TTSTokenizerV2Decoder with CUDA graphs for fixed-size inputs."""

    def __init__(self, decoder: torch.nn.Module, capture_sizes: list[int] | None = None):
        self.decoder = decoder
        self.num_quantizers = decoder.config.num_quantizers
        self.total_upsample = decoder.total_upsample
        self.capture_sizes = capture_sizes or [4, 8, 12, 25, 29, 37, 50]
        self.graphs: dict[int, CUDAGraph] = {}
        self.static_inputs: dict[int, torch.Tensor] = {}
        self.static_outputs: dict[int, torch.Tensor] = {}

    def _get_padded_size(self, actual: int) -> int | None:
        for s in self.capture_sizes:
            if actual <= s:
                return s
        return None

    @torch.inference_mode()
    def capture(self, device: torch.device, num_warmup: int = 3):
        self.decoder.eval()
        for size in self.capture_sizes:
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

            self.graphs[size] = graph
            self.static_inputs[size] = static_input
            self.static_outputs[size] = static_output
            logger.info("Captured codec graph for size=%d", size)

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        actual = codes.shape[-1]
        padded = self._get_padded_size(actual)

        if padded is None or padded not in self.graphs:
            return self.decoder(codes)

        self.static_inputs[padded].zero_()
        self.static_inputs[padded][:, :, :actual] = codes
        self.graphs[padded].replay()

        actual_len = actual * self.total_upsample
        return self.static_outputs[padded][..., :actual_len].clone()

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
