import subprocess
import os

# Define Python executables for each env
PY_VENV = os.path.abspath("venv/Scripts/python.exe")
PY_VIDEO = os.path.abspath("video_env/Scripts/python.exe")

steps = [
    ("Generating Stories", PY_VIDEO, "scripts/generate_story.py"),
    ("Generating Audio", PY_VENV, "scripts/generate_narration.py"),
    ("Generating Subtitles", PY_VIDEO, "scripts/create_subtitles.py"),
    ("Generating Videos", PY_VIDEO, "scripts/generate_video.py"),
]

for label, python_exec, script_path in steps:
    print(f"\n🚀 {label}...")
    result = subprocess.run([python_exec, script_path])
    if result.returncode != 0:
        print(f"❌ Failed at step: {label}")
        break
else:
    print("\n🎉 All steps completed successfully!")
