"""
infer.py

Arabic-to-English speech translation using the fine-tuned SeamlessM4T-v2 + LoRA model.
Model is loaded directly from Hugging Face. TTS is handled locally by Kokoro.

Pipeline:
    Arabic audio -> [Fine-tuned SeamlessM4T-v2 S2TT] -> English text -> [Kokoro TTS] -> English speech

Usage:
    python scripts/infer.py --audio path/to/arabic_audio.wav
    python scripts/infer.py --audio path/to/arabic_audio.wav --output english.wav
    python scripts/infer.py --audio path/to/arabic_audio.wav --play

Arguments:
    --audio     Path to input Arabic audio file (WAV or MP3)
    --output    Path to save English speech output (default: output.wav)
    --play      Play the output audio after generation (requires sounddevice)
    --device    cuda or cpu (default: auto-detect)

Example:
    python scripts/infer.py --audio demo/sample_audio/example_ar.wav
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from peft import PeftModel


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
HF_MODEL_ID    = "hams-chadi/seamless-m4t-v2-arabic-english-lora"
BASE_MODEL_ID  = "facebook/seamless-m4t-v2-large"
TGT_LANG       = "eng"
SAMPLE_RATE    = 16_000
MAX_NEW_TOKENS = 128
NUM_BEAMS      = 5


# ------------------------------------------------------------
# Audio loading
# ------------------------------------------------------------
def load_audio(path: str) -> np.ndarray:
    """Load an audio file and resample to 16 kHz mono float32."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
def load_model(device: str):
    """Load the fine-tuned model and processor from Hugging Face."""
    print("Loading processor from Hugging Face ...")
    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)

    print("Loading base model ...")
    base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float32,
    )

    print("Loading LoRA adapter from Hugging Face ...")
    model = PeftModel.from_pretrained(base, HF_MODEL_ID)
    model = model.merge_and_unload().to(device).eval()

    print(f"Model loaded on {device}.")
    return model, processor


# ------------------------------------------------------------
# TTS (Kokoro)
# ------------------------------------------------------------
def load_tts():
    """Load Kokoro TTS pipeline."""
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code="a")
        print("Kokoro TTS loaded.")
        return pipeline
    except Exception as e:
        print(f"Warning: Kokoro unavailable ({e}). Falling back to pyttsx3.")
        return None


def synthesize(text: str, output_path: str, kokoro_pipeline, tts_sr: int = 24000):
    """Convert English text to speech. Returns (waveform, sample_rate)."""
    text = (text or "").strip()
    if not text:
        wav = np.zeros(tts_sr, dtype=np.float32)
        sf.write(output_path, wav, tts_sr)
        return wav, tts_sr

    if kokoro_pipeline is not None:
        chunks = []
        for _, _, audio in kokoro_pipeline(text, voice="af_heart", speed=1.0):
            if audio is not None and len(audio) > 0:
                chunks.append(np.asarray(audio, dtype=np.float32))
        wav = np.concatenate(chunks) if chunks else np.zeros(tts_sr, dtype=np.float32)
        sf.write(output_path, wav, tts_sr)
        return wav, tts_sr

    # pyttsx3 fallback
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    wav, sr = sf.read(output_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), sr


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
def translate(audio_path: str, output_path: str, model, processor, kokoro):
    """
    Full pipeline: Arabic audio -> English text -> English speech.
    Returns a dict with translated_text, output_audio, and latency_ms.
    """
    audio = load_audio(audio_path)
    duration_s = len(audio) / SAMPLE_RATE

    # Step 1: S2TT
    t0 = time.perf_counter()
    inputs = processor(
        audios=audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).to(next(model.parameters()).device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            tgt_lang=TGT_LANG,
            num_beams=NUM_BEAMS,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    text  = processor.batch_decode(tokens, skip_special_tokens=True)[0]
    t_s2tt = (time.perf_counter() - t0) * 1000

    # Step 2: TTS
    t0 = time.perf_counter()
    wav, tts_sr = synthesize(text, output_path, kokoro)
    t_tts = (time.perf_counter() - t0) * 1000

    total = t_s2tt + t_tts
    rtf   = (total / 1000) / duration_s if duration_s > 0 else 0

    return {
        "source_audio"    : audio_path,
        "translated_text" : text,
        "output_audio"    : output_path,
        "latency_ms": {
            "time_to_first_text" : round(t_s2tt, 1),
            "time_to_first_audio": round(total, 1),
            "total_end_to_end"   : round(total, 1),
        },
        "audio_duration_s" : round(duration_s, 2),
        "real_time_factor"  : round(rtf, 3),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Arabic-to-English speech translation"
    )
    parser.add_argument(
        "--audio", required=True,
        help="Path to input Arabic audio file (WAV or MP3)"
    )
    parser.add_argument(
        "--output", default="output.wav",
        help="Path to save English speech output (default: output.wav)"
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play the output audio after generation"
    )
    parser.add_argument(
        "--device", default=None,
        help="cuda or cpu (default: auto-detect)"
    )
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"Error: audio file not found: {args.audio}")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nArabic-to-English Speech Translation")
    print(f"  Input  : {args.audio}")
    print(f"  Output : {args.output}")
    print(f"  Device : {device}")
    print(f"  Model  : {HF_MODEL_ID}")
    print()

    model, processor = load_model(device)
    kokoro = load_tts()

    result = translate(args.audio, args.output, model, processor, kokoro)

    print()
    print("=" * 50)
    print(json.dumps(result, indent=2))

    if args.play:
        try:
            import sounddevice as sd
            wav, sr = sf.read(args.output)
            print("\nPlaying output ...")
            sd.play(wav, sr)
            sd.wait()
        except ImportError:
            print("\nInstall sounddevice: pip install sounddevice")


if __name__ == "__main__":
    main()