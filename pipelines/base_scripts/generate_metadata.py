import os
import re
import argparse
import subprocess
import pandas as pd
from datetime import datetime

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description="Generate video metadata (title, description, tags) using Ollama.")
parser.add_argument("--csv", type=str, help="Path to input CSV", default=os.getenv("CSV_PATH", "pipelines/input.csv"))
args = parser.parse_args()

# ===== Configuration =====
CSV_PATH = args.csv
METADATA_DIR = 'stories/metadata'
MODEL_NAME = 'mistral'
os.makedirs(METADATA_DIR, exist_ok=True)

# ===== Logging =====
def print_status(msg, status="info"):
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
    print(f"{datetime.now().strftime('%H:%M:%S')} {icons.get(status, '')} {msg}")

# ===== Content Generation =====
def generate_metadata(story_text):
    system_prompt = (
        "You are a YouTube Shorts strategist and SEO expert.\n"
        "Given a short video script, generate metadata optimized for YouTube and Google search.\n"
        "Return the result in exactly this format:\n\n"
        "[TITLE]\n"
        "A punchy, curiosity-driven title (max 15 words)\n"
        "[DESCRIPTION]\n"
        "A two-paragraph teaser that ends with a question mark and includes at least 1-5 keyword and relevant hashtags\n"
        "[TAGS]\n"
        "8-15 comma-separated lowercase keywords (no hashtags)\n\n"
        "Rules:\n"
        "- Do not skip or rename any section\n"
        "- Title should include keywords and trigger curiosity\n"
        "- Tags should include variations (e.g., 'nasa, space, astronomy, science news')\n"
        "- No formatting or markdown\n"
        "- Use plain text only"
    )

    result = subprocess.run(
        ['ollama', 'run', MODEL_NAME, system_prompt + "\n\nSTORY:\n" + story_text],
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

# ===== Metadata Validation =====
def validate_metadata(raw_text):
    try:
        normalized_text = '\n'.join(line.strip() for line in raw_text.splitlines())
        title = re.search(r'\[TITLE\]\s*(.+?)(?=\n\[DESCRIPTION\])', normalized_text, re.DOTALL).group(1).strip()
        description = re.search(r'\[DESCRIPTION\]\s*(.+?)(?=\n\[TAGS\])', normalized_text, re.DOTALL).group(1).strip()
        tags = re.search(r'\[TAGS\]\s*(.+)', normalized_text, re.DOTALL).group(1).strip()
        return {'title': title, 'description': description, 'tags': tags}
    except Exception as e:
        print_status(f"⚠️ Metadata validation failed: {e}", "warning")
        print_status(f"Raw metadata:\n{raw_text}", "info")
        return None

# ===== Main =====
def process_csv():
    print_status("🚀 Starting Step 2: Metadata Generation", "progress")
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
        print_status(f"Loaded CSV with {len(df)} rows", "success")
    except Exception as e:
        print_status(f"❌ Failed to read CSV: {str(e)}", "error")
        return

    for col in ['Title', 'Description', 'Tags', 'MetadataPath', 'MetadataStatus']:
        if col not in df.columns:
            df[col] = ""

    for idx, row in df.iterrows():
        if str(row.get("MetadataStatus", "")).lower() == "completed":
            print_status(f"Skipping row {idx}: metadata already completed", "info")
            continue

        story = str(row.get("StoryText", "")).strip()
        if not story:
            print_status(f"Skipping row {idx}: missing story", "warning")
            continue

        story_id = row.get("ID", idx)
        print_status(f"🎯 Generating metadata for ID {story_id}", "progress")

        try:
            metadata_raw = generate_metadata(story)
            metadata = validate_metadata(metadata_raw)
            if not metadata:
                df.at[idx, 'MetadataStatus'] = "Failed - Invalid format"
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metadata_{story_id}_{timestamp}.txt"
            metadata_path = os.path.join(METADATA_DIR, filename)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {metadata['title']}\n\n")
                f.write(f"Description: {metadata['description']}\n\n")
                f.write(f"Tags: {metadata['tags']}\n")

            df.at[idx, 'Title'] = metadata['title']
            df.at[idx, 'Description'] = metadata['description']
            df.at[idx, 'Tags'] = metadata['tags']
            df.at[idx, 'MetadataPath'] = metadata_path
            df.at[idx, 'MetadataStatus'] = "Completed"

            print_status(f"✅ Metadata saved: {metadata_path}", "success")

        except Exception as e:
            print_status(f"❌ Error on ID {story_id}: {e}", "error")
            df.at[idx, 'MetadataStatus'] = f"Failed - {str(e)}"

    try:
        df.to_csv(CSV_PATH, index=False)
        print_status("✅ CSV updated", "success")
    except Exception as e:
        print_status(f"❌ Failed to save CSV: {str(e)}", "error")

# ===== Entry Point =====
if __name__ == "__main__":
    process_csv()
    print_status("🏁 Step 2 complete", "success")
