"""
Gradio web interface for Arabic-to-English speech translation.
Two input modes:
  1. Choose a sample from the test set by ID
  2. Upload your own Arabic audio file

 pip install gradio
python demo/app.py
# Then open http://localhost:7860

Model loaded from Hugging Face on startup.
"""

import json
import time
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import gradio as gr
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
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Paths — relative to arabic_en_translation/
ROOT      = Path(__file__).parent.parent
TEST_SET  = ROOT / "test_set.json"
AUDIO_DIR = ROOT / "audio_outputs"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load model once at startup
# ------------------------------------------------------------
print(f"Device: {DEVICE}")
print("Loading model from Hugging Face ...")

processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
base      = SeamlessM4Tv2ForSpeechToText.from_pretrained(
    BASE_MODEL_ID, dtype=torch.float32
)
peft_model = PeftModel.from_pretrained(base, HF_MODEL_ID)
model      = peft_model.merge_and_unload().to(DEVICE).eval()
print("Model ready.")

# Load Kokoro TTS
print("Loading Kokoro TTS ...")
try:
    from kokoro import KPipeline
    kokoro = KPipeline(lang_code="a")
    TTS_SR = 24000
    print("Kokoro ready.")
except Exception as e:
    print(f"Kokoro unavailable: {e}")
    kokoro = None
    TTS_SR = 24000

# Load test set
test_data = []
if TEST_SET.exists():
    with open(TEST_SET, encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Test set loaded: {len(test_data)} examples")

# Warmup
print("Running warmup ...")
dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
inp = processor(audios=dummy, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    model.generate(**inp, tgt_lang=TGT_LANG, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS)
print("Warmup done. Ready.")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_audio_file(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def synthesize(text: str) -> tuple:
    if not text.strip() or kokoro is None:
        return np.zeros(TTS_SR, dtype=np.float32), TTS_SR
    chunks = []
    for _, _, audio in kokoro(text, voice="af_heart", speed=1.0):
        if audio is not None and len(audio) > 0:
            chunks.append(np.asarray(audio, dtype=np.float32))
    wav = np.concatenate(chunks) if chunks else np.zeros(TTS_SR, dtype=np.float32)
    return wav, TTS_SR


def run_translation(audio_array: np.ndarray):
    """Translate Arabic audio array to English text and speech."""
    duration_s = len(audio_array) / SAMPLE_RATE

    # S2TT
    t0 = time.perf_counter()
    inputs = processor(
        audios=audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        tokens = model.generate(
            **inputs, tgt_lang=TGT_LANG,
            num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS,
        )
    text   = processor.batch_decode(tokens, skip_special_tokens=True)[0]
    t_s2tt = (time.perf_counter() - t0) * 1000

    # TTS
    t0 = time.perf_counter()
    wav, tts_sr = synthesize(text)
    t_tts = (time.perf_counter() - t0) * 1000

    total = t_s2tt + t_tts
    rtf   = (total / 1000) / duration_s if duration_s > 0 else 0

    latency_info = (
        f"S2TT (translation) : {t_s2tt:.0f} ms\n"
        f"TTS  (speech)      : {t_tts:.0f} ms\n"
        f"Total end-to-end   : {total:.0f} ms\n"
        f"Audio duration     : {duration_s:.2f} s\n"
        f"Real-time factor   : {rtf:.3f}"
    )

    # Save output wav to temp file for Gradio audio player
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wav, tts_sr)

    return text, tmp.name, latency_info


# ------------------------------------------------------------
# Tab 1: Choose from dataset
# ------------------------------------------------------------
def translate_from_dataset(sample_id: int):
    if not test_data:
        return "", None, "", "test_set.json not found.", ""

    idx = int(sample_id) - 1
    if idx < 0 or idx >= len(test_data):
        return "", None, "", f"ID must be between 1 and {len(test_data)}.", ""

    ex = test_data[idx]

    # Fix path prefix
    audio_path = str(ex["audio_path"])
    for prefix in ["arabic_en_translation\\", "arabic_en_translation/"]:
        if audio_path.startswith(prefix):
            audio_path = str(ROOT / audio_path[len(prefix):])
            break
    if not Path(audio_path).is_absolute():
        audio_path = str(ROOT / audio_path)

    try:
        audio_array = load_audio_file(audio_path)
    except Exception as e:
        return "", None, "", f"Could not load audio: {e}", ""

    arabic_text  = ex.get("sentence", "")
    reference    = ex.get("translation", "")

    english_text, output_wav, latency_info = run_translation(audio_array)

    return (
        arabic_text,
        (SAMPLE_RATE, audio_array),
        reference,
        english_text,
        output_wav,
        latency_info,
    )


# ------------------------------------------------------------
# Tab 2: Upload your own audio
# ------------------------------------------------------------
def translate_uploaded(audio_path):
    if audio_path is None:
        return "", None, "Please upload an audio file."

    try:
        audio_array = load_audio_file(audio_path)
    except Exception as e:
        return "", None, f"Could not load audio: {e}"

    english_text, output_wav, latency_info = run_translation(audio_array)
    return english_text, output_wav, latency_info


# ------------------------------------------------------------
# Gradio interface
# ------------------------------------------------------------
with gr.Blocks(title="Arabic-to-English Speech Translation") as demo:
    gr.Markdown("""
# Arabic-to-English Speech Translation
Fine-tuned SeamlessM4T-v2 + LoRA — Modern Standard Arabic to English

**Model**: [hams-chadi/seamless-m4t-v2-arabic-english-lora](https://huggingface.co/hams-chadi/seamless-m4t-v2-arabic-english-lora)

Pipeline: Arabic speech → SeamlessM4T-v2 (S2TT) → English text → Kokoro TTS → English speech
    """)

    with gr.Tabs():

        # ---- Tab 1: Dataset sample ----
        with gr.Tab("Choose from Test Set"):
            gr.Markdown(f"Select a sample ID from the test set (1 to {len(test_data)}).")

            with gr.Row():
                sample_id = gr.Number(
                    label="Sample ID",
                    value=1,
                    minimum=1,
                    maximum=len(test_data) if test_data else 300,
                    step=1,
                    precision=0,
                )
                btn1 = gr.Button("Translate", variant="primary")

            with gr.Row():
                with gr.Column():
                    arabic_text  = gr.Textbox(label="Arabic source", interactive=False)
                    arabic_audio = gr.Audio(label="Arabic input audio", interactive=False)
                    reference    = gr.Textbox(label="English reference", interactive=False)
                with gr.Column():
                    english_text  = gr.Textbox(label="English translation", interactive=False)
                    english_audio = gr.Audio(label="English output speech", interactive=False)
                    latency1      = gr.Textbox(label="Latency breakdown", interactive=False, lines=5)

            btn1.click(
                fn=translate_from_dataset,
                inputs=[sample_id],
                outputs=[arabic_text, arabic_audio, reference, english_text, english_audio, latency1],
            )

        # ---- Tab 2: Upload audio ----
        with gr.Tab("Upload Your Own Audio"):
            gr.Markdown("Upload any Arabic speech audio file (WAV or MP3).")

            with gr.Row():
                upload = gr.Audio(label="Upload Arabic audio", type="filepath")
                btn2   = gr.Button("Translate", variant="primary")

            with gr.Row():
                with gr.Column():
                    english_text2  = gr.Textbox(label="English translation", interactive=False)
                with gr.Column():
                    english_audio2 = gr.Audio(label="English output speech", interactive=False)
                    latency2       = gr.Textbox(label="Latency breakdown", interactive=False, lines=5)

            btn2.click(
                fn=translate_uploaded,
                inputs=[upload],
                outputs=[english_text2, english_audio2, latency2],
            )

    gr.Markdown("""
---
**Note:** First request may be slower due to CUDA kernel initialization.
Subsequent requests will be faster.
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )