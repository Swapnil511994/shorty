import pandas as pd
import feedparser
import re
import argparse
from datetime import datetime
import urllib.parse
import os

def print_status(msg, level="info"):
    prefix = {
        "info": "🔄",
        "success": "✅",
        "error": "❌",
        "warn": "⚠️",
    }.get(level, "ℹ️")
    print(f"{datetime.now().strftime('%H:%M:%S')} {prefix} {msg}")

def sanitize_phrase(phrase):
    return re.sub(r"[^a-zA-Z\s]", "", phrase).strip()

def build_rss_query_string(trends):
    parts = []
    for trend in trends:
        for phrase in trend.split(","):
            clean = sanitize_phrase(phrase)
            if clean:
                parts.append(clean)
    return " ".join(parts)[:200]  # RSS uses space-based search

def fetch_articles_rss_batched(trends, batch_size=2, max_articles_per_batch=5):
    all_articles = []
    seen_titles = set()

    for i in range(0, len(trends), batch_size):
        batch = trends[i:i + batch_size]
        search_terms = []

        for trend in batch:
            for phrase in trend.split(","):
                clean = sanitize_phrase(phrase)
                if clean:
                    search_terms.append(clean)

        query = " ".join(search_terms)
        encoded_query = urllib.parse.quote_plus(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)

        for entry in feed.entries[:max_articles_per_batch]:
            if entry.title not in seen_titles:
                all_articles.append({
                    "headline": entry.get("title", ""),
                    "description": entry.get("summary", ""),
                    "content": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "combined": f"{entry.get('title', '')}\n\n{entry.get('summary', '')}"
                })
                seen_titles.add(entry.title)

    return all_articles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="pipelines/datum.csv", help="Output file")
    parser.add_argument("--trends_csv", type=str, default="pipelines/trends.csv", help="Trends file")
    parser.add_argument("--news_limit", type=int, default=10, help="Max articles to fetch")
    parser.add_argument("--regions", type=str, default="in", help="Comma-separated country codes")
    args = parser.parse_args()

    print_status("🚀 Starting trends-to-stories pipeline")

    try:
        df = pd.read_csv(args.trends_csv)
        if "Trend breakdown" not in df.columns:
            raise Exception("Missing 'Trend breakdown' column.")
        trends = df["Trend breakdown"].dropna().tolist()[:10]
        print_status(f"✅ Loaded top {len(trends)} trends", "success")
    except Exception as e:
        print_status(f"❌ Failed to load trends: {e}", "error")
        return

    query = build_rss_query_string(trends)
    print_status(f"🔍 Final Google News RSS Query: {query}", "info")

    articles = fetch_articles_rss_batched(trends, batch_size=1, max_articles_per_batch=args.news_limit)

    if not articles:
        print_status("⚠️ No articles found in RSS feed", "warn")
        return

    out_df = pd.DataFrame(articles)
    out_df.to_csv(args.csv, index=False)
    print_status(f"✅ Saved {len(articles)} articles to {args.csv}", "success")
    print_status("🏁 Pipeline complete!", "success")

if __name__ == "__main__":
    main()
