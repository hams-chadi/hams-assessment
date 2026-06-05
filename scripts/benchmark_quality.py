"""
benchmark_quality.py

Evaluate Arabic-to-English speech translation quality on the held-out test set.
Computes BLEU, chrF, and COMET for the fine-tuned model, and (optionally) the
zero-shot baseline for a before/after comparison.

Usage:
    python scripts/benchmark_quality.py
    python scripts/benchmark_quality.py --compare-zeroshot
    python scripts/benchmark_quality.py --max-samples 100

Inputs:
    arabic_en_translation/test_set.json          (exported by the notebook)
    arabic_en_translation/checkpoints/lora_s2tt/best_adapter
    arabic_en_translation/checkpoints/lora_s2tt/processor

Outputs:
    arabic_en_translation/results/quality_report.json
    arabic_en_translation/results/quality_report.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_MODEL  = "facebook/seamless-m4t-v2-large"
TGT_LANG    = "eng"
SAMPLE_RATE = 16_000
MAX_LABEL_LEN = 128

ROOT        = Path(".")
CKPT_DIR    = ROOT / "checkpoints" / "lora_s2tt"
ADAPTER_DIR = CKPT_DIR / "best_adapter"
PROC_DIR    = CKPT_DIR / "processor"
RESULTS_DIR = ROOT / "results"
TEST_SET    = ROOT / "test_set.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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


def translate(audio_array, processor, model, num_beams=5):
    audio_array = np.asarray(audio_array, dtype=np.float32)
    inputs = processor(
        audios=audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        tokens = model.generate(
            **inputs, tgt_lang=TGT_LANG, num_beams=num_beams,
            max_new_tokens=MAX_LABEL_LEN,
        )
    return processor.batch_decode(tokens, skip_special_tokens=True)[0]


def run_inference(test_data, processor, model, desc):
    hyps, refs, srcs = [], [], []
    for ex in tqdm(test_data, desc=desc):
        audio = load_audio(ex["audio_path"])
        hyps.append(translate(audio, processor, model))
        refs.append(ex["translation"])
        srcs.append(ex["sentence"])
    return hyps, refs, srcs


def score_bleu_chrf(hyps, refs):
    bleu = BLEU()
    chrf = CHRF()
    return (
        round(bleu.corpus_score(hyps, [refs]).score, 2),
        round(chrf.corpus_score(hyps, [refs]).score, 2),
    )


def score_comet(hyps, refs, srcs):
    """COMET semantic score. Returns None if COMET is unavailable."""
    try:
        from comet import download_model, load_from_checkpoint
        print("Loading COMET model (downloads on first run) ...")
        path = download_model("Unbabel/wmt22-comet-da")
        comet = load_from_checkpoint(path)
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        out = comet.predict(data, batch_size=8, gpus=1 if DEVICE == "cuda" else 0)
        return round(out["system_score"], 4)
    except Exception as e:
        print("COMET unavailable:", str(e)[:200])
        return None


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Translation quality benchmark")
    parser.add_argument("--compare-zeroshot", action="store_true",
                        help="Also evaluate the zero-shot base model for before/after")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit the number of test examples")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test set
    if not TEST_SET.exists():
        raise FileNotFoundError(
            f"{TEST_SET} not found. Run the export cell in the notebook first."
        )
    with open(TEST_SET, encoding="utf-8") as f:
        test_data = json.load(f)
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    print(f"Loaded {len(test_data)} test examples.")
    print(f"Device: {DEVICE}")

    # Load processor
    processor = AutoProcessor.from_pretrained(str(PROC_DIR))

    # ---- Fine-tuned model ----
    print("\nLoading fine-tuned model ...")
    base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    )
    ft_model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    ft_model = ft_model.merge_and_unload().to(DEVICE).eval()

    ft_hyps, refs, srcs = run_inference(test_data, processor, ft_model, "Fine-tuned")
    ft_bleu, ft_chrf = score_bleu_chrf(ft_hyps, refs)
    ft_comet = score_comet(ft_hyps, refs, srcs)

    del ft_model, base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    report = {
        "n_samples": len(test_data),
        "device": DEVICE,
        "fine_tuned": {"BLEU": ft_bleu, "chrF": ft_chrf, "COMET": ft_comet},
    }

    # ---- Optional zero-shot baseline ----
    if args.compare_zeroshot:
        print("\nLoading zero-shot base model ...")
        zs_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32
        ).to(DEVICE).eval()

        zs_hyps, _, _ = run_inference(test_data, processor, zs_model, "Zero-shot")
        zs_bleu, zs_chrf = score_bleu_chrf(zs_hyps, refs)
        zs_comet = score_comet(zs_hyps, refs, srcs)

        del zs_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        report["zero_shot"] = {"BLEU": zs_bleu, "chrF": zs_chrf, "COMET": zs_comet}

    # ---- Save reports ----
    with open(RESULTS_DIR / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(RESULTS_DIR / "quality_report.txt", "w", encoding="utf-8") as f:
        f.write("Translation Quality Benchmark\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test examples : {report['n_samples']}\n")
        f.write(f"Device        : {report['device']}\n\n")
        if "zero_shot" in report:
            f.write(f"{'Metric':<10}{'Before (zero-shot)':<22}{'After (fine-tuned)':<22}\n")
            f.write("-" * 54 + "\n")
            for m in ("BLEU", "chrF", "COMET"):
                zs = report["zero_shot"][m]
                ft = report["fine_tuned"][m]
                f.write(f"{m:<10}{str(zs):<22}{str(ft):<22}\n")
        else:
            f.write(f"{'Metric':<10}{'Fine-tuned':<15}\n")
            f.write("-" * 25 + "\n")
            for m in ("BLEU", "chrF", "COMET"):
                f.write(f"{m:<10}{str(report['fine_tuned'][m]):<15}\n")

    # ---- Print to console ----
    print("\n" + "=" * 50)
    print("Quality Benchmark Results")
    print("=" * 50)
    if "zero_shot" in report:
        print(f"{'Metric':<10}{'Before':<15}{'After':<15}")
        for m in ("BLEU", "chrF", "COMET"):
            print(f"{m:<10}{str(report['zero_shot'][m]):<15}{str(report['fine_tuned'][m]):<15}")
    else:
        for m in ("BLEU", "chrF", "COMET"):
            print(f"{m:<10}{report['fine_tuned'][m]}")
    print(f"\nSaved to {RESULTS_DIR / 'quality_report.json'}")
    print(f"Saved to {RESULTS_DIR / 'quality_report.txt'}")


if __name__ == "__main__":
    main()