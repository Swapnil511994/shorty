import argparse
import subprocess
import os

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Run the YouTube Shorts generation pipeline.")
parser.add_argument("--csv", type=str, default="pipelines/datum.csv", help="Path to input CSV")
parser.add_argument("--news_limit", type=str, default="10", help="Number of news items to fetch")
parser.add_argument("--regions", type=str, default="in", help="Comma-separated country codes")
parser.add_argument("--query", type=str, default="general", help="Search query for GNews")
parser.add_argument("--upload", action="store_true", help="Include this flag to upload videos to YouTube")
args = parser.parse_args()

# === Configurable Flags ===
CSV_PATH = args.csv
NEWS_LIMIT = args.news_limit
REGIONS = args.regions
QUERY = args.query

# === Python Executables ===
PY_VENV = os.path.abspath("venv/Scripts/python.exe")               # General purpose
XTTS_VENV = os.path.abspath("xtts_env/Scripts/python.exe")         # Text-to-Speech (XTTS)
WHISPERX_ENV = os.path.abspath("whisperx_env/Scripts/python.exe")  # WhisperX subtitles
PY_VIDEO = os.path.abspath("video_env/Scripts/python.exe")         # Video generation & upload
GENAI_VENV = os.path.abspath("genai_env/Scripts/python.exe")       # GenAI for image generation
HIDREAM_ENV = os.path.abspath("hidream_env/Scripts/python.exe")       # GenAI for image generation

# === Step-by-step Commands ===
pipeline_steps = [
    # (
    #     "📰 Fetching News & Generating Stories",
    #     PY_VENV,
    #     "pipelines/custom_scripts/news.py",
    #     ["--csv", CSV_PATH, "--news_limit", NEWS_LIMIT, "--regions", REGIONS, "--query", QUERY]
    # ),
    # (
    #     "🧠 Generating Metadata",
    #     PY_VENV,
    #     "pipelines/base_scripts/generate_metadata.py",
    #     ["--csv", CSV_PATH]
    # ),
    # (
    #     "🎙️ Generating Narration",
    #     XTTS_VENV,
    #     "pipelines/base_scripts//generate_narration.py",
    #     ["--csv", CSV_PATH]
    # ),
    # (
    #     "📝 Creating Subtitles",
    #     WHISPERX_ENV,
    #     "pipelines/base_scripts/create_subtitles.py",
    #     ["--csv", CSV_PATH]
    # ),
    (
        "🎬 Generating Videos",
        HIDREAM_ENV,
        "pipelines/base_scripts/generate_visuals.py",
        ["--csv", CSV_PATH]
    ),
]


# === Run Pipeline ===
for label, python_exec, script_path, script_args in pipeline_steps:
    print(f"\n🚀 {label}")
    result = subprocess.run([python_exec, script_path] + script_args,encoding='utf-8')
    if result.returncode != 0:
        print(f"❌ Failed at step: {label}")
        break
else:
    print("\n🎉 All steps completed successfully!")
