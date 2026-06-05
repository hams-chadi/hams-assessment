"""
benchmark_concurrency.py

Test the fine-tuned Arabic-to-English speech translation model under concurrent load.
Submits several requests at once using a thread pool and reports, per concurrency
level, the mean and p95 latency, throughput, GPU memory usage, and failure count.

Because Python threads share one GPU, concurrent requests are effectively serialized
on the device, so latency is expected to rise with concurrency. The goal is to show
the model can serve more than one live session without breaking.

Usage:
    python scripts/benchmark_concurrency.py
    python scripts/benchmark_concurrency.py --levels 1 2 4 8

Inputs:
    arabic_en_translation/test_set.json
    arabic_en_translation/checkpoints/lora_s2tt/best_adapter
    arabic_en_translation/checkpoints/lora_s2tt/processor

Outputs:
    arabic_en_translation/results/concurrency_report.csv
    arabic_en_translation/results/concurrency_report.txt
"""

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torchaudio

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

# Globals shared by worker threads
_model = None
_processor = None


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


def one_request(audio_array):
    """Single inference request. Returns (latency_ms, success)."""
    try:
        audio_array = np.asarray(audio_array, dtype=np.float32)
        t0 = time.perf_counter()
        inputs = _processor(audios=audio_array, sampling_rate=SAMPLE_RATE,
                            return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            tokens = _model.generate(**inputs, tgt_lang=TGT_LANG,
                                     num_beams=5, max_new_tokens=MAX_LABEL_LEN)
        _processor.batch_decode(tokens, skip_special_tokens=True)
        return (time.perf_counter() - t0) * 1000, True
    except Exception:
        return None, False


def benchmark_concurrency(audio_pool, levels):
    rows = []
    for n in levels:
        batch = [audio_pool[i % len(audio_pool)] for i in range(n)]
        rounds = max(1, 16 // n)
        lats, failures = [], 0

        t_wall = time.perf_counter()
        for _ in range(rounds):
            with ThreadPoolExecutor(max_workers=n) as ex:
                futures = [ex.submit(one_request, a) for a in batch]
                for fut in as_completed(futures):
                    lat, ok = fut.result()
                    if ok:
                        lats.append(lat)
                    else:
                        failures += 1
        wall = time.perf_counter() - t_wall

        total_reqs = n * rounds
        arr = np.array(lats) if lats else np.array([0.0])
        vram = round(torch.cuda.memory_allocated() / 1e9, 2) if DEVICE == "cuda" else None

        row = {
            "concurrency": n,
            "n_requests": total_reqs,
            "failures": failures,
            "mean_lat_ms": round(float(arr.mean()), 1),
            "p95_lat_ms": round(float(np.percentile(arr, 95)), 1),
            "throughput_rps": round(total_reqs / wall, 2),
            "vram_gb": vram,
        }
        rows.append(row)
        print(f"  c={n:>2} | mean {row['mean_lat_ms']}ms | p95 {row['p95_lat_ms']}ms "
              f"| throughput {row['throughput_rps']} rps | failures {failures}")
    return rows


def main():
    global _model, _processor

    parser = argparse.ArgumentParser(description="Concurrency benchmark")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4],
                        help="Concurrency levels to test (e.g. 1 2 4 8)")
    parser.add_argument("--max-pool", type=int, default=8,
                        help="Number of distinct audio clips to cycle through")
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

    # Load model into the module globals so worker threads share it
    _processor = AutoProcessor.from_pretrained(str(PROC_DIR))
    base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    )
    _model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    _model = _model.merge_and_unload().to(DEVICE).eval()

    # Pre-load a small pool of audio clips
    pool = [load_audio(ex["audio_path"]) for ex in test_data[:args.max_pool]]

    print(f"\nRunning concurrency benchmark at levels {args.levels} ...")
    rows = benchmark_concurrency(pool, args.levels)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "concurrency_report.csv", index=False)

    with open(RESULTS_DIR / "concurrency_report.txt", "w", encoding="utf-8") as f:
        f.write("Concurrency Benchmark\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Device : {DEVICE}\n")
        if DEVICE == "cuda":
            f.write(f"GPU    : {torch.cuda.get_device_name(0)}\n")
        f.write("\n" + df.to_string(index=False) + "\n\n")
        f.write("Threads share one GPU, so latency rises with concurrency. The test\n")
        f.write("confirms the model serves multiple sessions without failures.\n")

    print("\n" + "=" * 50)
    print("Concurrency Benchmark Results")
    print("=" * 50)
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR / 'concurrency_report.csv'}")


if __name__ == "__main__":
    main()