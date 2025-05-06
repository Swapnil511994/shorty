# This script generates audio narrations for stories using TTS models.
# from TTS.api import TTS

# # Init
# tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

# # Generate narration to file
# tts.tts_to_file(
#     text="This is a sample narration using a single speaker model. can you hear me? choot ka chakkar maut ki takkar hai.",
#     file_path="output.wav"
# )



#working code for tortoise v2
# from TTS.api import TTS

# # Choose Tortoise v2 model
# tts = TTS(model_name="tts_models/en/multi-dataset/tortoise-v2").to("cuda")  # or "cpu"

# # Available speaker IDs: "tom", "william", "daniel" are deeper voices
# tts.tts_to_file(
#     text="This is a deep male voice generated using Tortoise.",
#     # speaker="tom",  # or try "william"
#     file_path="output.wav"
# )

import os
import pandas as pd
from TTS.api import TTS

# ===== Config =====
CSV_PATH = "data/input.csv"
STORY_DIR = "stories/generated"
OUTPUT_DIR = "audio/narrations"
# MODEL_NAME = "tts_models/en/multi-dataset/tortoise-v2"
MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
# MODEL_NAME = "tts_models/en/ljspeech/glow-tts"
# MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC_ph"
# MODEL_NAME = "tts_models/en/vctk/vits"
# MODEL_NAME = "tts_models/en/ljspeech/fast_pitch"
# MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER = "p225"  # Options: "tom", "william", "daniel"
DEVICE = "cuda"  # Use "cpu" if no GPU

# ===== Init =====
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("🔄 Loading TTS model...")
tts = TTS(model_name=MODEL_NAME).to(DEVICE)

# ===== Load CSV =====
df = pd.read_csv(CSV_PATH)
print(f"📊 Loaded {len(df)} records from {CSV_PATH}")

# ===== Process Narrations =====
for idx, row in df.iterrows():
    if row.get("StoryStatus", "pending").strip().lower() != "completed":
        continue
    if row.get("AudioStatus", "pending").startswith("completed"):
        continue

    story_id = row["ID"]
    story_path = row["StoryPath"]
    output_path = os.path.join(OUTPUT_DIR, f"story_{story_id}.wav")

    try:
        with open(story_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            raise ValueError("Empty story text")

        print(f"🎙️ Generating narration for ID {story_id} with speaker '{SPEAKER}'...")
        print(f"📄 Text snippet: {text[:150]}...")

        tts.tts_to_file(
            text=text,
            # speaker=SPEAKER,
            file_path=output_path
        )

        df.at[idx, "AudioPath"] = output_path
        df.at[idx, "AudioStatus"] = "completed"
        print(f"✅ Audio saved: {output_path}")

    except Exception as e:
        df.at[idx, "AudioStatus"] = f"error: {str(e)}"
        print(f"❌ Error for ID {story_id}: {e}")

# ===== Save Updated CSV =====
try:
    df.to_csv(CSV_PATH, index=False)
    print(f"📦 Narration process complete. CSV updated at {CSV_PATH}")
except PermissionError:
    print(f"⚠️ Failed to save CSV. Please close '{CSV_PATH}' if it's open in another program.")
