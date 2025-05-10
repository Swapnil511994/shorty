import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Setup headless Chrome
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)

try:
    print("🔍 Loading Google Trends page...")
    driver.get("https://trends.google.com/trending?geo=US&hl=en-US")
    time.sleep(5)  # Wait for JS to load

    elements = driver.find_elements(By.CSS_SELECTOR, 'div.feed-item span.title')
    top_trends = [el.text.strip() for el in elements if el.text.strip() != ""][:10]

    print("🔥 Top 10 Google Trends (via Selenium):")
    for i, trend in enumerate(top_trends, 1):
        print(f"{i}. {trend}")

    # Update CSV
    csv_path = 'data/input.csv'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            'ID', 'Prompt', 'StoryText', 'StoryStatus', 'Title',
            'Description', 'Tags', 'MetadataStatus', 'AudioPath',
            'AudioStatus', 'SubtitlePath', 'SubtitleStatus', 'VideoPath',
            'VideoStatus', 'YouTubeStatus', 'YouTubeLink', 'YouTubeID',
            'TrendDate', 'Source'
        ])

    # Filter new prompts
    existing_prompts = set(df['Prompt'].dropna())
    new_trends = [t for t in top_trends if t not in existing_prompts]

    start_id = df['ID'].max() + 1 if not df.empty else 0
    new_rows = []
    for i, trend in enumerate(new_trends):
        new_rows.append({
            'ID': start_id + i,
            'Prompt': trend,
            'StoryText': '',
            'StoryStatus': 'pending',
            'Title': '',
            'Description': '',
            'Tags': 'trending',
            'MetadataStatus': 'pending',
            'AudioPath': '',
            'AudioStatus': 'pending',
            'SubtitlePath': '',
            'SubtitleStatus': 'pending',
            'VideoPath': '',
            'VideoStatus': 'pending',
            'YouTubeStatus': 'pending',
            'YouTubeLink': '',
            'YouTubeID': '',
            'TrendDate': timestamp,
            'Source': 'Google Trends Page'
        })

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"✅ Appended {len(new_rows)} new trends to {csv_path}")
    else:
        print("ℹ️ No new trends to append (already present)")

finally:
    driver.quit()
