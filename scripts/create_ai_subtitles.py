# import os
# import pandas as pd
# import torch
# import whisperx
# from datetime import datetime

# # ===== CONFIG =====
# CSV_PATH = "data/input.csv"
# LANGUAGE = None  # Set to None for auto-detect, or use "en" or "hi"
# MODEL_SIZE = "medium"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# AUDIO_COL = "AudioPath"
# ID_COL = "ID"
# STATUS_COL = "SubtitleStatus"
# SRT_COL = "SubtitlePath"

# AUDIO_DIR = "audio/narrations"
# SRT_OUTPUT_DIR = "subtitles/srt"
# os.makedirs(SRT_OUTPUT_DIR, exist_ok=True)

# # ===== Logging =====
# def print_status(msg, status="info"):
#     icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
#     print(f"{datetime.now().strftime('%H:%M:%S')} {icons.get(status, '')} {msg}")

# # ===== SRT Writer (Manual) =====
# def write_srt_manual(segments, output_path):
#     def format_timestamp(seconds):
#         h = int(seconds // 3600)
#         m = int((seconds % 3600) // 60)
#         s = int(seconds % 60)
#         ms = int((seconds - int(seconds)) * 1000)
#         return f"{h:02}:{m:02}:{s:02},{ms:03}"

#     with open(output_path, "w", encoding="utf-8") as f:
#         for idx, seg in enumerate(segments, 1):
#             start = format_timestamp(seg['start'])
#             end = format_timestamp(seg['end'])
#             text = seg['text'].strip()
#             f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

# # ===== LOAD CSV =====
# try:
#     df = pd.read_csv(CSV_PATH)
#     print_status(f"Loaded CSV with {len(df)} rows", "success")
# except Exception as e:
#     print_status(f"Failed to read CSV: {str(e)}", "error")
#     exit()

# # ===== INIT MODEL =====
# print_status(f"Loading WhisperX model ({MODEL_SIZE}, device={DEVICE})...", "progress")
# model = whisperx.load_model(MODEL_SIZE, device=DEVICE, language=LANGUAGE)

# # ===== LOOP =====
# for i, row in df.iterrows():
#     if str(row.get(STATUS_COL, "")).lower() == "completed":
#         continue

#     audio_path = row.get(AUDIO_COL)
#     if not isinstance(audio_path, str) or not os.path.exists(audio_path):
#         print_status(f"Skipping missing audio: {audio_path}", "warning")
#         continue

#     try:
#         print_status(f"🎤 Transcribing ID {row[ID_COL]}", "progress")
#         result = model.transcribe(audio_path)

#         print_status("⏱️ Aligning words...", "progress")
#         model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
#         aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, DEVICE)

#         srt_path = os.path.join(SRT_OUTPUT_DIR, f"story_{row[ID_COL]}.srt")
#         write_srt_manual(aligned["segments"], srt_path)

#         df.at[i, SRT_COL] = srt_path
#         df.at[i, STATUS_COL] = "completed"
#         print_status(f"✅ Subtitle saved: {srt_path}", "success")

#     except Exception as e:
#         df.at[i, STATUS_COL] = f"error: {str(e)}"
#         print_status(f"❌ Error with ID {row[ID_COL]}: {e}", "error")

# # ===== SAVE CSV =====
# try:
#     df.to_csv(CSV_PATH, index=False)
#     print_status("✅ CSV updated with subtitle paths and status.", "success")
# except Exception as e:
#     print_status(f"❌ Failed to save CSV: {str(e)}", "error")


import os
import pandas as pd
import torch
import whisperx
from datetime import datetime
import re

# ===== CONFIG =====
CSV_PATH = "data/input.csv"
LANGUAGE = None  # Set to None for auto-detect, or use "en" or "hi"
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
    text = text.replace("'", "’")  # safer apostrophe
    text = text.replace('"', "")   # remove quotes
    text = text.replace('\\', '')  # remove backslashes
    text = text.replace('%', '')   # remove format symbols
    return text.strip()

# ===== SRT Writer (Safe) =====
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
                continue  # skip empty segments
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

# ===== LOAD CSV =====
try:
    df = pd.read_csv(CSV_PATH)
    print_status(f"Loaded CSV with {len(df)} rows", "success")
except Exception as e:
    print_status(f"Failed to read CSV: {str(e)}", "error")
    exit()

# ===== INIT MODEL =====
print_status(f"Loading WhisperX model ({MODEL_SIZE}, device={DEVICE})...", "progress")
model = whisperx.load_model(MODEL_SIZE, device=DEVICE, language=LANGUAGE)

# ===== LOOP =====
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

        srt_path = os.path.join(SRT_OUTPUT_DIR, f"story_{row[ID_COL]}.srt")
        write_srt_manual(aligned["segments"], srt_path)

        df.at[i, SRT_COL] = srt_path
        df.at[i, STATUS_COL] = "completed"
        print_status(f"✅ Subtitle saved: {srt_path}", "success")

    except Exception as e:
        df.at[i, STATUS_COL] = f"error: {str(e)}"
        print_status(f"❌ Error with ID {row[ID_COL]}: {e}", "error")

# ===== SAVE CSV =====
try:
    df.to_csv(CSV_PATH, index=False)
    print_status("✅ CSV updated with subtitle paths and status.", "success")
except Exception as e:
    print_status(f"❌ Failed to save CSV: {str(e)}", "error")
