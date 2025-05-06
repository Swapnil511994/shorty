import os
import pandas as pd
from TTS.api import TTS
# from pydub import AudioSegment  # For audio format conversion

# ===== Config =====
CSV_PATH = "data/input.csv"              # Path to your CSV
STORY_DIR = "stories/generated"          # Where text stories are stored
OUTPUT_DIR = "audio/narrations"          # Where to save audio
SAMPLE_PATH = "audio/sample/sample.wav"         # Your deep male voice sample
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "en"                          # en, es, fr, etc.
DEVICE = "cuda"                          # "cuda" or "cpu"
FP16 = True                              # Faster on RTX 4090
MIN_TEXT_LENGTH = 10                     # Skip texts shorter than this

# ===== Convert MP3 to WAV (XTTS requires 16kHz mono WAV) =====
def convert_to_wav(input_path):
    output_path = os.path.splitext(input_path)[0] + ".wav"
    if not os.path.exists(output_path):
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
    return output_path

# ===== Initialize =====
os.makedirs(OUTPUT_DIR, exist_ok=True)
# speaker_wav = convert_to_wav(SAMPLE_PATH)  # Auto-convert MP3->WAV
# print(f"🎙️ Using voice sample: {speaker_wav}")

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
    story_path = os.path.join(STORY_DIR, f"story_{story_id}.txt")
    output_path = os.path.join(OUTPUT_DIR, f"story_{story_id}.wav")

    try:
        with open(story_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        if len(text) < MIN_TEXT_LENGTH:
            raise ValueError(f"Text too short ({len(text)} chars). Min {MIN_TEXT_LENGTH} required.")
        
        print(f"🎙️ Processing ID {story_id} ({len(text)} chars)...")
        
        tts.tts_to_file(
            text=text,
            speaker_wav=SAMPLE_PATH,  # Your deep male voice
            language=LANGUAGE,
            file_path=output_path,
            split_sentences=True,     # Better pacing
            speed=1.0,                # Normal speed
            # pitch=0.85,               # Slightly deeper (0.5-1.5)
        )

        df.at[idx, "AudioPath"] = output_path
        df.at[idx, "AudioStatus"] = "completed"
        print(f"✅ Saved: {output_path}")

    except Exception as e:
        df.at[idx, "AudioStatus"] = f"error: {str(e)}"
        print(f"❌ Failed ID {story_id}: {str(e)}")

# ===== Save CSV =====
backup_path = CSV_PATH.replace(".csv", "_backup.csv")
df.to_csv(backup_path, index=False)  # Safety backup
df.to_csv(CSV_PATH, index=False)
print(f"📦 CSV updated. Backup: {backup_path}")