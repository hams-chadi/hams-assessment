"""
benchmark_latency.py

Measure end-to-end latency for the fine-tuned Arabic-to-English speech translation
model. Latency covers the full pipeline (audio preprocessing, model inference, and
decoding), not just the model forward pass. Reports mean, p50, p95, max latency and
the real-time factor (RTF = latency / audio duration).

Usage:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --n-runs 100 --warmup 5
    python scripts/benchmark_latency.py --num-beams 1     # greedy, lower latency

Inputs:
    arabic_en_translation/test_set.json
    arabic_en_translation/checkpoints/lora_s2tt/best_adapter
    arabic_en_translation/checkpoints/lora_s2tt/processor

Outputs:
    arabic_en_translation/results/latency_report.json
    arabic_en_translation/results/latency_report.txt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from peft import PeftModel


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_MODEL    = "facebook/seamless-m4t-v2-large"
TGT_LANG      = "eng"
SAMPLE_RATE   = 16_000
MAX_LABEL_LEN = 128

ROOT        = Path(".")
CKPT_DIR    = ROOT / "checkpoints" / "lora_s2tt"
ADAPTER_DIR = CKPT_DIR / "best_adapter"
PROC_DIR    = CKPT_DIR / "processor"
RESULTS_DIR = ROOT / "results"
TEST_SET    = ROOT / "test_set.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path):
    """Load an audio file (MP3 or WAV), resample to 16 kHz mono float32."""
    import librosa
    path = str(path)
    # Strip leading 'arabic_en_translation' prefix if present
    # (test_set.json was saved from the notebook with the full path)
    if path.startswith("arabic_en_translation"):
        path = path[len("arabic_en_translation"):].lstrip("\\/")
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Number of measured runs")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup runs (not measured)")
    parser.add_argument("--num-beams", type=int, default=5,
                        help="Beam search width (1 = greedy, lower latency)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_SET.exists():
        raise FileNotFoundError(
            f"{TEST_SET} not found. Run the export cell in the notebook first."
        )
    with open(TEST_SET, encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # Load model
    processor = AutoProcessor.from_pretrained(str(PROC_DIR))
    base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model = model.merge_and_unload().to(DEVICE).eval()

    # Pre-load the audio we will use
    n_total = min(args.n_runs + args.warmup, len(test_data))
    samples = []
    for ex in test_data[:n_total]:
        arr = load_audio(ex["audio_path"])
        samples.append(arr)

    # Benchmark
    latencies, rtfs = [], []
    for i, audio in enumerate(tqdm(samples, desc="Latency benchmark")):
        dur = len(audio) / SAMPLE_RATE

        # Warmup runs are not recorded
        if i < args.warmup:
            inputs = processor(audios=audio, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                model.generate(**inputs, tgt_lang=TGT_LANG,
                               num_beams=args.num_beams, max_new_tokens=MAX_LABEL_LEN)
            continue

        # Measure the FULL pipeline: preprocess -> inference -> decode
        t0 = time.perf_counter()
        inputs = processor(audios=audio, sampling_rate=SAMPLE_RATE,
                           return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            tokens = model.generate(**inputs, tgt_lang=TGT_LANG,
                                    num_beams=args.num_beams, max_new_tokens=MAX_LABEL_LEN)
        processor.batch_decode(tokens, skip_special_tokens=True)
        total_ms = (time.perf_counter() - t0) * 1000

        latencies.append(total_ms)
        rtfs.append((total_ms / 1000) / dur if dur > 0 else 0)

    lat = np.array(latencies)
    report = {
        "n_samples": len(lat),
        "num_beams": args.num_beams,
        "mean_latency_ms": round(float(lat.mean()), 1),
        "p50_latency_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 1),
        "max_latency_ms": round(float(lat.max()), 1),
        "mean_rtf": round(float(np.mean(rtfs)), 3),
        "p95_rtf": round(float(np.percentile(rtfs, 95)), 3),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
        "batch_size": 1,
    }

    # Save
    with open(RESULTS_DIR / "latency_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(RESULTS_DIR / "latency_report.txt", "w", encoding="utf-8") as f:
        f.write("Latency Benchmark (full pipeline)\n")
        f.write("=" * 50 + "\n\n")
        for k, v in report.items():
            f.write(f"{k:<20}: {v}\n")
        f.write("\nLatency covers preprocessing + inference + decoding, not just the\n")
        f.write("model forward pass. RTF below 1.0 means faster than real time.\n")

    print("\n" + "=" * 50)
    print("Latency Benchmark Results")
    print("=" * 50)
    for k, v in report.items():
        print(f"  {k:<20}: {v}")
    print(f"\nSaved to {RESULTS_DIR / 'latency_report.json'}")


if __name__ == "__main__":
    main()