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
parser.add_argument("--csv", type=str, help="Path to input CSV", default=os.getenv("CSV_PATH", "pipelines/input.csv"))
args = parser.parse_args()
CSV_PATH = args.csv

# ===== Configuration =====
STORY_DIR = "stories/generated"
MODEL_NAME = "gemma"

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
    return '\n'.join(line.strip() for line in text.splitlines()).strip()

# ===== Story Generator =====
def generate_story(prompt):
    system_prompt = (
        "आप एक विशेषज्ञ सामग्री निर्माता हैं जो YouTube Shorts के लिए छोटे, शैक्षिक और आकर्षक स्क्रिप्ट लिखते हैं।\n"
        "आपका लक्ष्य समाचार विषयों को स्पष्ट रूप से और एल्गोरिदम तक पहुंच को बढ़ावा देने के तरीके से समझाना है।\n"
        "प्रदान किए गए हेडलाइन और विवरण का उपयोग करके:\n"
        "- समाचार क्या है यह स्पष्ट करें\n"
        "- इसका क्या अर्थ है या यह क्यों महत्वपूर्ण है, यह बताएं\n"
        "- स्क्रिप्ट के अंत में CTA जोड़ें: 'लाइक करें, शेयर करें और सब्सक्राइब करें!'\n\n"
        "नियम:\n"
        "- स्क्रिप्ट 200 शब्दों से कम होनी चाहिए\n"
        "- इसे नेचुरल तरीके से लिखें जिससे वॉयसओवर अच्छा लगे\n"
        "- कोई हैशटैग, इमोजी या वर्णनात्मक लेबल शामिल न करें\n"
        "- आउटपुट में केवल बोलचाल की भाषा होनी चाहिए, शीर्षक या हेडिंग नहीं\n"
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
