import os
import requests
import pandas as pd
from datetime import datetime

# ===== CONFIG =====
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
CSV_PATH = "data/input.csv"
NEWS_LIMIT = 10
REGIONS = ["us", "in"]  # You can modify or extend this list
QUERY = "latest news"

# ===== LOGGING =====
def print_status(msg, status="info"):
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
    print(f"{datetime.now().strftime('%H:%M:%S')} {icons.get(status, '')} {msg}")

# ===== FETCH NEWS =====
def fetch_gnews(country):
    url = f"https://gnews.io/api/v4/search?q={QUERY}&lang=en&country={country}&max={NEWS_LIMIT}&apikey={GNEWS_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        return data.get("articles", [])
    except Exception as e:
        print_status(f"Failed to fetch news for {country}: {e}", "error")
        return []

# ===== PROCESS & SAVE =====
def update_csv(news_items):
    if not news_items:
        return

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=["ID", "Prompt", "ImageURL", "StoryStatus"])

    next_id = df["ID"].max() + 1 if not df.empty else 1

    for article in news_items:
        prompt = article.get("title", "")
        image = article.get("image", "")
        if not prompt:
            continue

        df = pd.concat([
            df,
            pd.DataFrame([{
                "ID": next_id,
                "Prompt": prompt,
                "ImageURL": image,
                "StoryStatus": "pending"
            }])
        ], ignore_index=True)
        next_id += 1

    df.to_csv(CSV_PATH, index=False)
    print_status(f"✅ CSV updated with {len(news_items)} new entries", "success")

# ===== MAIN =====
all_news = []
for region in REGIONS:
    print_status(f"🔎 Fetching news for region: {region}", "progress")
    region_news = fetch_gnews(region)
    all_news.extend(region_news)

update_csv(all_news)
print_status("🏁 GNews fetch complete.", "success")
