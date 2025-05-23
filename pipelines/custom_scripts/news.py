import os
import subprocess
import pandas as pd
from datetime import datetime,timedelta
import requests
import re
import argparse

# ===== Logging =====
def print_status(msg, status="info"):
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
    print(f"{datetime.now().strftime('%H:%M:%S')} {icons.get(status, '')} {msg}")

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description="Generate educational stories from news or prompts.")
parser.add_argument("--csv", type=str, help="Path to input CSV", default=os.getenv("CSV_PATH", "data/input.csv"))
parser.add_argument("--news_limit", type=int, help="Number of news items to fetch", default=10)
parser.add_argument("--regions", type=str, help="Comma-separated list of country codes (e.g., 'us,in')", default="in")
parser.add_argument("--query", type=str, help="Search query for GNews", default=None)
args = parser.parse_args()

CSV_PATH = args.csv
NEWS_LIMIT = args.news_limit
REGIONS = [r.strip() for r in args.regions.split(",") if r.strip()]
QUERY = args.query

# ===== Config =====
STORY_DIR = "stories/generated"
MODEL_NAME = "mistral"

# Load API key securely
try:
    with open("gnews_api.txt", "r") as f:
        GNEWS_API_KEY = f.read().strip()
except Exception as e:
    print_status(f"❌ Failed to read GNews API key: {e}", "error")
    GNEWS_API_KEY = ""

GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"
os.makedirs(STORY_DIR, exist_ok=True)

# ===== Fetch News =====
def fetch_news(country):
    if not QUERY:
        print_status("⚠️ QUERY is not set. Skipping news fetch.", "warning")
        return []
    twelve_hours_ago = datetime.utcnow() - timedelta(hours=6)
    from_timestamp = twelve_hours_ago.strftime("%Y-%m-%d%H:%M:%SZ")

    url = f"{GNEWS_ENDPOINT}?q={QUERY}&lang=en&country={country}&max={NEWS_LIMIT}&apikey={GNEWS_API_KEY}&from={from_timestamp}"
    print_status(f"🔍 Fetching news for region '{country}' via URL: {url}", "progress")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "image": article.get("image", ""),
                "url": article.get("url", "")
            } for article in data.get("articles", []) if article.get("title")
        ]
    except Exception as e:
        print_status(f"Failed to fetch news for {country}: {e}", "error")
        return []

# ===== Clean Story Text =====
def clean_story(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text, flags=re.UNICODE)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[\[\(].*?[\]\)]', '', text)
    text = re.sub(r'^(title|script|heading)[:\-–]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r' +', ' ', text)
    return '\n'.join(line.strip() for line in text.splitlines()).strip()

# ===== Generate Story =====
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
            timeout=120,
            errors="replace",
        )
        if result.returncode != 0:
            raise Exception(result.stderr.strip())
        return result.stdout.strip()
    except Exception as e:
        raise Exception(f"Model generation error: {e}")

# ===== Main Process =====
def process_news():
    print_status("🚀 Starting content generation from GNews", "progress")
    try:
        df = pd.read_csv(CSV_PATH,encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=["ID", "Prompt", "ImageURL", "StoryText", "StoryStatus"])

    for region in REGIONS:
        for article in fetch_news(region):
            prompt = f"{article['title']}\n\n{article['description']}"
            image = article.get("image", "")
            story_id = df["ID"].max() + 1 if not df.empty else 1
            df = pd.concat([
                df,
                pd.DataFrame([{ "ID": story_id, "Prompt": prompt, "ImageURL": image, "StoryText": "", "StoryStatus": "", "NewsUrl": article["url"] }])
            ], ignore_index=True)

    for idx, row in df.iterrows():
        if str(row.get("StoryStatus", "")).lower() == "completed":
            continue

        prompt = str(row.get("Prompt", "")).strip()
        if not prompt:
            continue

        story_id = row.get("ID", idx)
        try:
            story = generate_story(prompt)
            cleaned = clean_story(story)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"story_{story_id}_{timestamp}.txt"
            story_path = os.path.join(STORY_DIR, filename)
            with open(story_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)

            df.at[idx, "StoryText"] = cleaned
            df.at[idx, "StoryStatus"] = "Completed"
            df.at[idx, "StoryPath"] = story_path
            print_status(f"✅ Story {story_id} generated and saved", "success")
        except Exception as e:
            df.at[idx, "StoryStatus"] = f"Failed: {str(e)}"
            print_status(f"❌ Failed to generate story {story_id}: {e}", "error")

    try:
        df.to_csv(CSV_PATH, index=False)
        print_status("✅ CSV updated with new entries", "success")
    except Exception as e:
        print_status(f"❌ Failed to save CSV: {e}", "error")

if __name__ == "__main__":
    process_news()
    print_status("🌟 All news processed.", "success")
