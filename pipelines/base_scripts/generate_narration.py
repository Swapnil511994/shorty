import os
import argparse
import pandas as pd
from TTS.api import TTS
from datetime import datetime
# from pydub import AudioSegment  # Uncomment if converting MP3 to WAV

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description="Generate voice narration from stories using XTTS.")
parser.add_argument("--csv", type=str, help="Path to input CSV", default=os.getenv("CSV_PATH", "data/input.csv"))
parser.add_argument("--sample", type=str, help="Path to input sample audio", default=os.getenv("AUDIO_PATH", "audio/sample/sample.wav"))
args = parser.parse_args()

# ===== Config =====
CSV_PATH = args.csv
STORY_DIR = "stories/generated"
OUTPUT_DIR = "audio/narrations"
SAMPLE_PATH = args.sample
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "en"
DEVICE = "cuda"
FP16 = True
MIN_TEXT_LENGTH = 10

# ===== Convert MP3 to WAV (XTTS requires 16kHz mono WAV) =====
# def convert_to_wav(input_path):
#     output_path = os.path.splitext(input_path)[0] + ".wav"
#     if not os.path.exists(output_path):
#         audio = AudioSegment.from_file(input_path)
#         audio = audio.set_frame_rate(16000).set_channels(1)
#         audio.export(output_path, format="wav")
#     return output_path

# ===== Initialize =====
os.makedirs(OUTPUT_DIR, exist_ok=True)
# SAMPLE_PATH = convert_to_wav(SAMPLE_PATH)
# print(f"🎙️ Using voice sample: {SAMPLE_PATH}")

print("🔄 Loading TTS model...")
tts = TTS(
    model_name=MODEL_NAME,
    progress_bar=True,
    gpu=True if DEVICE == "cuda" else False,
).to(DEVICE)

# ===== Load CSV =====
df = pd.read_csv(CSV_PATH)
print(f"📊 Loaded {len(df)} records from {CSV_PATH}")

# ===== Process Narrations =====
for idx, row in df.iterrows():
    if str(row.get("StoryStatus", "")).lower() != "completed":
        continue
    if str(row.get("AudioStatus", "")).startswith("completed"):
        continue

    story_id = row["ID"]
    story_path = row.get("StoryPath")
    # if not isinstance(story_path, str) or not story_path.strip():
    #     story_path = os.path.join(STORY_DIR, f"story_{story_id}.txt")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"story_{story_id}_{timestamp}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        with open(story_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if len(text) < MIN_TEXT_LENGTH:
            raise ValueError(f"Text too short ({len(text)} chars). Min {MIN_TEXT_LENGTH} required.")

        print(f"🎙️ Processing ID {story_id} ({len(text)} chars)...")

        # Clean and trim text
        text = "\n".join([line.strip() for line in text.strip().splitlines() if line.strip()])

        tts.tts_to_file(
            text=text,
            speaker_wav=SAMPLE_PATH,
            language=LANGUAGE,
            file_path=output_path,
            split_sentences=True,
            speed=1.0,
            # pitch=0.85,
        )

        df.at[idx, "AudioPath"] = output_path
        df.at[idx, "AudioStatus"] = "completed"
        print(f"✅ Saved: {output_path}")

    except Exception as e:
        df.at[idx, "AudioStatus"] = f"error: {str(e)}"
        print(f"❌ Failed ID {story_id}: {str(e)}")

# ===== Save CSV =====
backup_path = CSV_PATH.replace(".csv", "_backup.csv")
df.to_csv(backup_path, index=False)
df.to_csv(CSV_PATH, index=False)
print(f"📦 CSV updated. Backup: {backup_path}")


# import os
# import argparse
# import pandas as pd
# from TTS.api import TTS
# from datetime import datetime
# from langdetect import detect

# # ===== Argument Parser =====
# parser = argparse.ArgumentParser(description="Generate voice narration from stories using XTTS.")
# parser.add_argument("--csv", type=str, help="Path to input CSV", default=os.getenv("CSV_PATH", "data/input.csv"))
# args = parser.parse_args()

# # ===== Config =====
# CSV_PATH = args.csv
# STORY_DIR = "stories/generated"
# OUTPUT_DIR = "audio/narrations"
# SAMPLE_PATH = "audio/sample/sample.wav"
# MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
# LANGUAGE = "en"
# DEVICE = "cuda"
# FP16 = True
# MIN_TEXT_LENGTH = 10

# # ===== Initialize =====
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# print("🔄 Loading TTS model...")
# tts = TTS(
#     model_name=MODEL_NAME,
#     progress_bar=True,
#     gpu=True if DEVICE == "cuda" else False,
# ).to(DEVICE)

# # ===== Helper: Add <lang=xx> Tags to Sentences =====
# def tag_multilingual_text(text):
#     lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
#     tagged_lines = []
#     for line in lines:
#         try:
#             lang = detect(line)
#             if lang.startswith("hi"):
#                 tagged_lines.append(f"<lang=hi>{line}</lang>")
#             else:
#                 tagged_lines.append(f"<lang=en>{line}</lang>")
#         except:
#             tagged_lines.append(f"<lang={LANGUAGE}>{line}</lang>")
#     return "\n".join(tagged_lines)

# # ===== Load CSV =====
# df = pd.read_csv(CSV_PATH)
# print(f"📊 Loaded {len(df)} records from {CSV_PATH}")

# # ===== Process Narrations =====
# for idx, row in df.iterrows():
#     if str(row.get("StoryStatus", "")).lower() != "completed":
#         continue
#     if str(row.get("AudioStatus", "")).startswith("completed"):
#         continue

#     story_id = row["ID"]
#     story_path = row.get("StoryPath")

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"story_{story_id}_{timestamp}.wav"
#     output_path = os.path.join(OUTPUT_DIR, filename)

#     try:
#         with open(story_path, "r", encoding="utf-8") as f:
#             text = f.read().strip()

#         if len(text) < MIN_TEXT_LENGTH:
#             raise ValueError(f"Text too short ({len(text)} chars). Min {MIN_TEXT_LENGTH} required.")

#         print(f"🎙️ Processing ID {story_id} ({len(text)} chars)...")

#         # Add language tags
#         text = tag_multilingual_text(text)

#         tts.tts_to_file(
#             text=text,
#             speaker_wav=SAMPLE_PATH,
#             language=LANGUAGE,
#             file_path=output_path,
#             split_sentences=True,
#             speed=1.0
#         )

#         df.at[idx, "AudioPath"] = output_path
#         df.at[idx, "AudioStatus"] = "completed"
#         print(f"✅ Saved: {output_path}")

#     except Exception as e:
#         df.at[idx, "AudioStatus"] = f"error: {str(e)}"
#         print(f"❌ Failed ID {story_id}: {str(e)}")

# # ===== Save CSV =====
# backup_path = CSV_PATH.replace(".csv", "_backup.csv")
# df.to_csv(backup_path, index=False)
# df.to_csv(CSV_PATH, index=False)
# print(f"📦 CSV updated. Backup: {backup_path}")
