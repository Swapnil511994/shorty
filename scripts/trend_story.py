import os
import subprocess
import pandas as pd
from datetime import datetime
import requests
import re

# ===== Logging =====
def print_status(msg, status="info"):
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
    print(f"{datetime.now().strftime('%H:%M:%S')} {icons.get(status, '')} {msg}")

# === Config ===
CSV_PATH = "data/input.csv"
STORY_DIR = "stories/generated"
MODEL_NAME = "mistral"

# Load API key securely
try:
    with open("gnews_api.txt", "r") as f:
        GNEWS_API_KEY = f.read().strip()
except Exception as e:
    print_status(f"❌ Failed to read GNews API key: {e}", "error")
    GNEWS_API_KEY = ""

GNEWS_ENDPOINT = "https://gnews.io/api/v4/top-headlines"
os.makedirs(STORY_DIR, exist_ok=True)

# ===== Fetch US Headlines =====
def fetch_us_headlines(max_results=10):
    params = {
        "lang": "en",
        "country": "us",
        "token": GNEWS_API_KEY,
        "max": max_results
    }
    try:
        response = requests.get(GNEWS_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": article["title"],
                "description": article.get("description", "")
            }
            for article in data.get("articles", [])
        ]
    except Exception as e:
        print_status(f"GNews fetch error: {e}", "error")
        return []

# ===== Clean Story =====
def clean_story(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text, flags=re.UNICODE)  # Emojis
    text = re.sub(r'#\w+', '', text)  # Hashtags
    text = re.sub(r'[\[\(].*?[\]\)]', '', text)  # [Music], (Narrator)
    text = re.sub(r'^(title|script|heading)[:\-–]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = '\n'.join(line.strip() for line in text.splitlines())  # Trim each line
    return text.strip()


# ===== Story Generator =====
def generate_story(prompt):
    system_prompt = (
        "You are an expert content creator who writes short, educational, and engaging scripts for YouTube Shorts.\n"
        "Your goal is to explain news topics clearly and in a way that drives algorithmic reach.\n"
        "Use the provided headline and description to:\n"
        "- Clearly explain what the news is about\n"
        "- Add value by saying what it means or why it matters\n"
        "- Use terms like 'explained', 'what it means', or 'did you know' to increase search discoverability\n"
        "- End the script with a CTA: 'Like, share, and subscribe!'\n\n"
        "Rules:\n"
        "- The script must be under 200 words\n"
        "- Write naturally for voice narration\n"
        "- Do NOT include any descriptors like [Music], (Narrator), no hashtags, etc.\n"
        "- Do NOT include any emojis in the output.\n"
        "- Do NOT start with Here's your YouTube Shorts script: or any other informative text.\n"
        "- Format output as clean, spoken text — no title, no headings, just the script"
    )
    try:
        result = subprocess.run(
            ['ollama', 'run', MODEL_NAME, system_prompt + "\n\nPROMPT: " + prompt],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            timeout=120
        )
        if result.returncode != 0:
            raise Exception(result.stderr.strip())
        return result.stdout.strip()
    except Exception as e:
        raise Exception(f"Model generation error: {e}")

# ===== Main Process =====
def process_csv():
    print_status("🚀 Starting Step 1: Content Generation", "progress")
    try:
        df = pd.read_csv(CSV_PATH)
        print_status(f"Loaded CSV with {len(df)} rows", "success")
    except Exception:
        df = pd.DataFrame(columns=["ID", "Prompt", "StoryText", "StoryStatus"])

    if df.empty or df["Prompt"].dropna().empty:
        print_status("No prompts found — fetching US news headlines from GNews", "info")
        headlines = fetch_us_headlines(max_results=10)
        df = pd.DataFrame([{
            "ID": i,
            "Prompt": f"{item['title']}\n\n{item['description']}",
            "StoryText": "",
            "StoryStatus": ""
        } for i, item in enumerate(headlines)])

    for idx, row in df.iterrows():
        if str(row.get("StoryStatus", "")).lower() == "completed":
            print_status(f"Skipping row {idx}: already completed", "info")
            continue

        raw_prompt = str(row.get("Prompt", "")).strip()
        story_id = row.get("ID", idx)

        print_status(f"📚 Generating script for ID {story_id}\n{raw_prompt}", "progress")
        try:
            story = generate_story(raw_prompt)
            cleaned_story = clean_story(story)

            path = os.path.join(STORY_DIR, f"story_{story_id}.txt")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(cleaned_story)

            df.at[idx, 'StoryText'] = cleaned_story
            df.at[idx, 'StoryStatus'] = "Completed"
            print_status(f"✅ Script generated and saved for ID {story_id}", "success")
        except Exception as e:
            df.at[idx, 'StoryStatus'] = f"Failed: {str(e)}"
            print_status(f"❌ Failed to generate script for ID {story_id}: {e}", "error")

    try:
        df.to_csv(CSV_PATH, index=False)
        print_status("✅ CSV updated", "success")
    except Exception as e:
        print_status(f"❌ Failed to save CSV: {str(e)}", "error")

if __name__ == "__main__":
    process_csv()
    print_status("🏁 Step 1 complete", "success")
