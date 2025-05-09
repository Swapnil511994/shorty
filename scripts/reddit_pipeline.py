import subprocess
import os

# Define Python executables for each env
PY_VENV = os.path.abspath("venv/Scripts/python.exe")
XTTS_VENV = os.path.abspath("xtts_env/Scripts/python.exe")
PY_VIDEO = os.path.abspath("video_env/Scripts/python.exe")

steps = [
    ("Generating Reddit Stories", PY_VIDEO, "scripts/generate_reddit_story.py"),
    ("Generating Metadata", PY_VIDEO, "scripts/generate_metadata_from_story.py"),
    ("Generating Audio", XTTS_VENV, "scripts/generate_xtts_narration.py"),
    ("Generating Subtitles", PY_VIDEO, "scripts/create_ai_subtitles.py"),
    ("Generating Videos", PY_VIDEO, "scripts/generate_video.py"),
    ("Uploading Videos", PY_VIDEO, "scripts/upload_youtube_shorts.py"),
]

for label, python_exec, script_path in steps:
    print(f"\n🚀 {label}...")
    result = subprocess.run([python_exec, script_path])
    if result.returncode != 0:
        print(f"❌ Failed at step: {label}")
        break
else:
    print("\n🎉 All steps completed successfully!")
