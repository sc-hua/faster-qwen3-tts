#!/usr/bin/env python3
"""
Real-time TTS with CUDA graphs using a precomputed speaker embedding (x-vector mode).

This mode uses only the speaker embedding for voice identity, without the full
acoustic prompt from reference audio. Benefits:
- No accent bleed from reference audio into other languages
- Shorter prefill (10 tokens vs ~80+ in ICL mode) = lower TTFT
- Speaker embedding can be precomputed and cached (4KB file)

Usage:
    # First extract the speaker embedding (one-time) via Python:
    #   model = FasterQwen3TTS.from_pretrained(...)
    #   prompt = model.extract_voice_prompt(ref_audio="voice.wav", xvec_only=True)
    #   torch.save(prompt["ref_spk_embedding"], "speaker.pt")

    # Then generate with CUDA graphs:
    python examples/generate_with_embedding.py --speaker speaker.pt --text "Hello world" --language English --output out.wav
    python examples/generate_with_embedding.py --speaker speaker.pt --text "Bonjour le monde" --language French --output out.wav
"""
import argparse
import torch
import sys

sys.path.insert(0, '.')


def main():
    parser = argparse.ArgumentParser(description="CUDA-graphed TTS with precomputed speaker embedding")
    parser.add_argument("--speaker", required=True, help="Path to speaker embedding (.pt)")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", default="Auto", help="Language (English, French, German, Spanish, ...)")
    parser.add_argument("--output", default="output.wav", help="Output wav path")
    parser.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Model path")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()

    import soundfile as sf
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"Loading model from {args.model_path}...")
    model = FasterQwen3TTS.from_pretrained(
        args.model_path,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Generate using .pt file directly — model auto-detects format
    print(f"Generating with speaker from {args.speaker}...")
    audio_list, sr = model.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=args.speaker,  # .pt path works directly
        xvec_only=True,
    )

    sf.write(args.output, audio_list[0], sr)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
