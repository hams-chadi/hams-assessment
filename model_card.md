# Model Card: Arabic-to-English Speech Translation (SeamlessM4T-v2 + LoRA)

## Summary
Fine-tuned Modern Standard Arabic to English speech translation. The system takes Arabic
speech and produces English text (via a fine-tuned SeamlessM4T-v2 model) and then English
speech (via Kokoro TTS, a local open-source TTS engine).

## Model
- Base model: facebook/seamless-m4t-v2-large (CC BY-NC 4.0)
- Fine-tuning: LoRA (rank 16, alpha 32, dropout 0.05)
- Target modules: q_proj, k_proj, v_proj, out_proj
- Task: MSA Arabic speech -> English text (S2TT), then Kokoro TTS for English speech
- TTS backend: Kokoro (local, Apache 2.0)

## Training
- Dataset: CoVoST-2 (ar_en), CC0
- Train / validation / test: 1113 / 300 / 300
- Epochs: 20 (10 epochs at lr=1e-5, then 10 epochs at lr=5e-6)
- Effective batch size: 16 (4 x 4 gradient accumulation)
- Optimizer: AdamW, weight decay 0.01
- Scheduler: Cosine with warmup
- Precision: fp16 on GPU
- Hardware: GPU with 17 GB VRAM
- Target deployment hardware: NVIDIA L4 (24 GB VRAM)
- Best validation loss: 1.86

## Results (held-out test set, 300 examples)

| Metric            | Before (zero-shot) | After (fine-tuned) |
|-------------------|--------------------|--------------------|
| BLEU              | 47.91              | 49.20              |
| chrF              | 65.45              | 65.89              |
| COMET             | 0.8864             | 0.8858             |
| Mean latency (ms) | 283                | 274                |
| p95 latency (ms)  | 363                | 363                |

Note: COMET stayed flat after fine-tuning. BLEU and chrF improved, indicating
better wording and fluency rather than a change in semantic meaning.

## Licenses
- SeamlessM4T-v2: CC BY-NC 4.0 (non-commercial)
- CoVoST-2: CC0 (commercial use allowed)
- TTS (Kokoro): Apache 2.0
- PEFT: Apache 2.0

The base model is non-commercial. A commercial deployment would replace it with a
permissively licensed speech-translation model.

## Hardware
- Inference: fits within 17 GB VRAM, batch size 1
- Assessment target: NVIDIA L4 (24 GB VRAM)
- Fine-tuning: LoRA keeps training within 24 GB on an L4
- CPU: possible but several times slower

## Known Limitations
- Trained on MSA; dialectal Arabic (Saudi, Egyptian) will degrade quality.
- Long utterances beyond about 30 seconds may truncate.
- Heavy background noise reduces accuracy.
- Code-switching (mixed Arabic and English) can leave some words untranslated.
- Non-streaming: time to first text equals full translation time.
- CC BY-NC 4.0 base model blocks commercial use as-is.

## Possible Improvements
- Train on more data (full CoVoST-2 is the largest available public Arabic S2T corpus).
- QLoRA (4-bit) to reduce memory further.
- ONNX or TensorRT for lower inference latency.
- SeamlessStreaming for partial output before the utterance ends.
- Reduce beam count for lower latency where quality permits.