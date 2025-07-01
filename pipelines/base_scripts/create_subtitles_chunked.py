import os
import re
import argparse
import pandas as pd
import torch
import whisperx
from datetime import datetime

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description="Generate subtitles using WhisperX with 4-word groups")
parser.add_argument("--csv", type=str, help="Path to input CSV", default=os.getenv("CSV_PATH", "pipelines/input.csv"))
args = parser.parse_args()

# ===== CONFIG =====
CSV_PATH = args.csv
LANGUAGE = None  # None = auto-detect, or use "en", "hi", etc.
MODEL_SIZE = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_COL = "AudioPath"
ID_COL = "ID"
STATUS_COL = "SubtitleStatus"
SRT_COL = "SubtitlePath"

AUDIO_DIR = "audio/narrations"
SRT_OUTPUT_DIR = "subtitles/srt"
os.makedirs(SRT_OUTPUT_DIR, exist_ok=True)

# ===== Logging =====
def print_status(msg, status="info"):
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
    print(f"{datetime.now().strftime('%H:%M:%S')} {icons.get(status, '')} {msg}")

# ===== Sanitizer =====
def sanitize_subtitle_text(text):
    text = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF]', '', text)  # remove control chars
    text = text.replace('"', "")
    text = text.replace('\\', '')
    text = text.replace('%', '')
    return text.strip()

# ===== Group words into 4-word segments =====
def regroup_segments_to_four_words(aligned_segments):
    new_segments = []
    all_words = []
    for seg in aligned_segments:
        if "words" in seg:
            all_words.extend(seg["words"])

    chunk = []
    for word in all_words:
        if word["word"].strip():
            chunk.append(word)
            if len(chunk) == 4:
                new_segments.append({
                    "start": chunk[0]["start"],
                    "end": chunk[-1]["end"],
                    "text": " ".join(w["word"] for w in chunk),
                })
                chunk = []

    if chunk:
        new_segments.append({
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": " ".join(w["word"] for w in chunk),
        })

    return new_segments

# ===== SRT Writer =====
def write_srt_manual(segments, output_path):
    def format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = sanitize_subtitle_text(seg['text'])
            if not text:
                continue
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

# ===== Load CSV =====
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print_status(f"Loaded CSV with {len(df)} rows", "success")
except Exception as e:
    print_status(f"Failed to read CSV: {str(e)}", "error")
    exit()

# ===== Init WhisperX =====
print_status(f"Loading WhisperX model ({MODEL_SIZE}, device={DEVICE})...", "progress")
model = whisperx.load_model(MODEL_SIZE, device=DEVICE, language=LANGUAGE)

# ===== Main Loop =====
for i, row in df.iterrows():
    if str(row.get(STATUS_COL, "")).lower() == "completed":
        continue

    audio_path = row.get(AUDIO_COL)
    if not isinstance(audio_path, str) or not os.path.exists(audio_path):
        print_status(f"Skipping missing audio: {audio_path}", "warning")
        continue

    try:
        print_status(f"🎤 Transcribing ID {row[ID_COL]}", "progress")
        result = model.transcribe(audio_path)

        print_status("⏱️ Aligning words...", "progress")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
        aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, DEVICE)

        print_status("✂️ Re-grouping segments to 4-word groups...", "progress")
        new_segments = regroup_segments_to_four_words(aligned["segments"])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"story_{row[ID_COL]}_{timestamp}.srt"
        srt_path = os.path.join(SRT_OUTPUT_DIR, filename)
        write_srt_manual(new_segments, srt_path)

        df.at[i, SRT_COL] = srt_path
        df.at[i, STATUS_COL] = "completed"
        print_status(f"✅ Subtitle saved: {srt_path}", "success")

    except Exception as e:
        df.at[i, STATUS_COL] = f"error: {str(e)}"
        print_status(f"❌ Error with ID {row[ID_COL]}: {e}", "error")

# ===== Save CSV =====
try:
    df.to_csv(CSV_PATH, index=False)
    print_status("✅ CSV updated with subtitle paths and status.", "success")
except Exception as e:
    print_status(f"❌ Failed to save CSV: {str(e)}", "error")
