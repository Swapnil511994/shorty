import os
import re
import subprocess
import requests
import pandas as pd
from datetime import datetime
import argparse

# ===== Logging =====
def print_status(msg, status="info"):
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
    print(f"{datetime.now().strftime('%H:%M:%S')} {icons.get(status, '')} {msg}")

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description="Generate educational stories from news or prompts.")
parser.add_argument("--csv", type=str, help="Path to input CSV", default=os.getenv("CSV_PATH", "data/input.csv"))
parser.add_argument("--news_limit", type=int, help="Number of news items to fetch", default=10)
parser.add_argument("--regions", type=str, help="Comma-separated list of country codes", default="us")
parser.add_argument("--query", type=str, help="Search query for GNews", default=None)
args = parser.parse_args()

CSV_PATH = args.csv
NEWS_LIMIT = args.news_limit
REGIONS = [r.strip() for r in args.regions.split(",") if r.strip()]
QUERY = args.query

# ===== Configuration =====
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

# ===== Fetch News Headlines =====
def fetch_news(country):
    if not GNEWS_API_KEY:
        print_status("❌ GNews API key missing - skipping news fetch", "error")
        return []
    
    params = {
        "lang": "en",
        "country": country,
        "token": GNEWS_API_KEY,
        "max": NEWS_LIMIT
    }
    if QUERY:
        params["q"] = QUERY
    
    try:
        response = requests.get(GNEWS_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": article["title"],
                "description": article.get("description", ""),
                "image": article.get("image", ""),
                "url": article.get("url", "")
            }
            for article in data.get("articles", [])
        ]
    except Exception as e:
        print_status(f"GNews fetch error for {country}: {e}", "error")
        return []

# ===== Clean Story =====
def clean_story(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text, flags=re.UNICODE)  # Emojis
    text = re.sub(r'#\w+', '', text)  # Hashtags
    text = re.sub(r'[\[\(].*?[\]\)]', '', text)  # [Music], (Narrator)
    text = re.sub(r'^(title|script|heading)[:\-–]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r' +', ' ', text)
    return '\n'.join(line.strip() for line in text.splitlines()).strip()

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
    
    # Try to load existing CSV or create new DataFrame
    try:
        df = pd.read_csv(CSV_PATH)
        print_status(f"Loaded CSV with {len(df)} existing rows", "success")
    except Exception:
        df = pd.DataFrame(columns=["ID", "Prompt", "ImageURL", "NewsUrl", "StoryText", "StoryStatus", "StoryPath"])
        print_status("Created new empty DataFrame", "info")

    # Only fetch news if DataFrame is empty or has no valid prompts
    if df.empty or df["Prompt"].dropna().empty:
        print_status("No existing prompts found - fetching news headlines", "info")
        news_articles = []
        for region in REGIONS:
            news_articles.extend(fetch_news(region))
        
        if news_articles:
            new_rows = [{
                "ID": len(df) + idx + 1,
                "Prompt": f"{article['title']}\n\n{article['description']}",
                "ImageURL": article.get("image", ""),
                "NewsUrl": article.get("url", ""),
                "StoryText": "",
                "StoryStatus": ""
            } for idx, article in enumerate(news_articles)]
            
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            print_status(f"Added {len(news_articles)} new prompts from news", "success")
        else:
            print_status("❌ No news articles found - nothing to process", "error")
            return

    # Process all rows (existing and newly added)
    for idx, row in df.iterrows():
        if str(row.get("StoryStatus", "")).lower() == "completed":
            print_status(f"Skipping row {idx}: already completed", "info")
            continue

        raw_prompt = str(row.get("Prompt", "")).strip()
        if not raw_prompt:
            print_status(f"Skipping row {idx}: empty prompt", "warning")
            continue

        story_id = row.get("ID", idx)
        print_status(f"📚 Generating script for ID {story_id}", "progress")
        
        try:
            story = generate_story(raw_prompt)
            cleaned_story = clean_story(story)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"story_{story_id}_{timestamp}.txt"
            story_path = os.path.join(STORY_DIR, filename)
            
            with open(story_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_story)

            df.at[idx, 'StoryText'] = cleaned_story
            df.at[idx, 'StoryStatus'] = "Completed"
            df.at[idx, 'StoryPath'] = story_path
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