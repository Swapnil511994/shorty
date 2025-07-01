import os
import argparse
import pandas as pd
from datetime import datetime
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description="Generate voice narration using ChatterboxTTS.")
parser.add_argument("--csv", type=str, help="Path to input CSV", default=os.getenv("CSV_PATH", "data/input.csv"))
parser.add_argument("--sample", type=str, help="Path to input sample audio", default=os.getenv("AUDIO_PATH", "audio/sample/sample.wav"))
args = parser.parse_args()

# ===== Config =====
CSV_PATH = args.csv
STORY_DIR = "stories/generated"
OUTPUT_DIR = "audio/narrations"
SAMPLE_PATH = args.sample
DEVICE = "cuda"
MIN_TEXT_LENGTH = 10

# ===== Initialize =====
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔄 Loading Chatterbox TTS model...")
model = ChatterboxTTS.from_pretrained(device=DEVICE)

# ===== Load CSV =====
df = pd.read_csv(CSV_PATH, encoding="utf-8")
print(f"📊 Loaded {len(df)} records from {CSV_PATH}")

# ===== Process Narrations =====
for idx, row in df.iterrows():
    if str(row.get("StoryStatus", "")).lower() != "completed":
        continue
    if str(row.get("AudioStatus", "")).startswith("completed"):
        continue

    story_id = row["ID"]
    story_path = row.get("StoryPath")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"story_{story_id}_{timestamp}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        with open(story_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if len(text) < MIN_TEXT_LENGTH:
            raise ValueError(f"Text too short ({len(text)} chars). Min {MIN_TEXT_LENGTH} required.")

        print(f"🎙️ Processing ID {story_id} ({len(text)} chars)...")

        # Clean text
        text = "\n".join([line.strip() for line in text.strip().splitlines() if line.strip()])

        # Generate narration using cloned voice
        wav = model.generate(text, audio_prompt_path=SAMPLE_PATH)

        ta.save(output_path, wav, model.sr)

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
