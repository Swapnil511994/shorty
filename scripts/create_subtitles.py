import os
import re
import pandas as pd
from moviepy.editor import AudioFileClip

# Paths
CSV_PATH = "data/input.csv"
STORY_DIR = "stories/generated"
SUBTITLE_DIR = "subtitles/srt"
os.makedirs(SUBTITLE_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH)

# Helper: Split long sentence into short phrases
def split_text_into_chunks(text, max_words=10):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# Function: Generate SRT from story and audio duration
def story_to_srt(story_text, audio_duration):
    # Break story into sentences
    sentences = re.split(r'(?<=[.!?]) +', story_text.strip())
    phrases = []

    for sentence in sentences:
        sentence = sentence.replace('\n', ' ').strip()
        if not sentence:
            continue
        phrases.extend(split_text_into_chunks(sentence, max_words=10))

    if not phrases:
        return ""

    chunk_duration = audio_duration / len(phrases)
    srt_lines = []
    idx = 1
    start_sec = 0

    for phrase in phrases:
        end_sec = start_sec + chunk_duration
        start_time = f"{int(start_sec//3600):02}:{int((start_sec%3600)//60):02}:{int(start_sec%60):02},000"
        end_time = f"{int(end_sec//3600):02}:{int((end_sec%3600)//60):02}:{int(end_sec%60):02},000"
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(phrase.strip())
        srt_lines.append("")
        start_sec = end_sec
        idx += 1

    return "\n".join(srt_lines)

# Process rows
print("📜 Starting subtitle generation...\n")

for _, row in df.iterrows():
    story_path = row.get("StoryPath")
    audio_path = row.get("AudioPath")
    story_id = row.get("ID")

    if not isinstance(story_path, str) or not os.path.exists(story_path):
        continue
    if not isinstance(audio_path, str) or not os.path.exists(audio_path):
        continue

    try:
        # Get audio duration
        duration = AudioFileClip(audio_path).duration

        with open(story_path, "r", encoding="utf-8") as f:
            story_text = f.read()

        srt_content = story_to_srt(story_text, duration)
        srt_output_path = os.path.join(SUBTITLE_DIR, f"story_{story_id}.srt")

        with open(srt_output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        df.loc[df["ID"] == story_id, "SubtitlePath"] = srt_output_path
        df.loc[df["ID"] == story_id, "SubtitleStatus"] = "completed"
        print(f"✅ Created: {srt_output_path}")

    except Exception as e:
        df.loc[df["ID"] == story_id, "SubtitleStatus"] = f"error: {e}"
        print(f"❌ Error for ID {story_id}: {e}")

# Save updated CSV
df.to_csv(CSV_PATH, index=False)
print("\n📦 Subtitle generation complete and CSV updated.")

# import os
# import pandas as pd
# import torch
# import whisperx

# # ===== CONFIG =====
# CSV_PATH = "data/input.csv"
# LANGUAGE = "hi"  # "en" for English, "hi" for Hindi
# MODEL = "medium"  # Options: tiny, base, small, medium, large
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# AUDIO_DIR = "audio/narrations"
# SRT_OUTPUT_DIR = "subtitles/srt"
# os.makedirs(SRT_OUTPUT_DIR, exist_ok=True)

# # ===== LOAD CSV =====
# df = pd.read_csv(CSV_PATH)

# # ===== INIT MODEL =====
# print(f"🔄 Loading WhisperX model ({MODEL}, lang={LANGUAGE}, device={DEVICE})...")
# model = whisperx.load_model(MODEL, device=DEVICE, language=LANGUAGE)

# # ===== LOOP =====
# for i, row in df.iterrows():
#     if row.get("AudioStatus", "").lower() != "completed":
#         continue
#     if row.get("SubtitleStatus", "").lower() == "completed":
#         continue

#     audio_path = row.get("AudioPath")
#     if not isinstance(audio_path, str) or not os.path.exists(audio_path):
#         print(f"⚠️ Skipping missing audio: {audio_path}")
#         continue

#     try:
#         print(f"\n🎙️ Transcribing ID {row['ID']}...")

#         # Step 1: Transcribe
#         result = model.transcribe(audio_path)

#         # Step 2: Align
#         print("⏱️ Aligning...")
#         model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)
#         aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, DEVICE)

#         # Step 3: Write to SRT
#         srt_path = os.path.join(SRT_OUTPUT_DIR, f"story_{row['ID']}.srt")
#         whisperx.utils.write_srt(aligned["segments"], srt_path)

#         df.at[i, "SubtitlePath"] = srt_path
#         df.at[i, "SubtitleStatus"] = "completed"
#         print(f"✅ Subtitle saved: {srt_path}")

#     except Exception as e:
#         df.at[i, "SubtitleStatus"] = f"error: {str(e)}"
#         print(f"❌ Error with ID {row['ID']}: {e}")

# # ===== SAVE CSV =====
# df.to_csv(CSV_PATH, index=False)
# print("📦 Subtitle generation complete and CSV updated.")
