import os
import subprocess
import pandas as pd
import re
from datetime import datetime

# ===== Configuration =====
CSV_PATH = 'data/input.csv'
METADATA_DIR = 'stories/metadata'
MODEL_NAME = 'mistral'

# ===== Setup Directories =====
os.makedirs(METADATA_DIR, exist_ok=True)

# ===== Logging =====
def print_status(msg, status="info"):
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
    print(f"{datetime.now().strftime('%H:%M:%S')} {icons.get(status, '')} {msg}")

# ===== Content Generation =====
def generate_metadata(story_text):
    system_prompt = (
        "You are a YouTube Shorts strategist.\n"
        "Generate metadata in this exact format:\n\n"
        "[TITLE]\n"
        "A catchy title (max 15 words)\n"
        "[DESCRIPTION]\n"
        "One-sentence teaser ending with or without a question mark\n"
        "[TAGS]\n"
        "5–8 comma-separated lowercase tags (no hashtags)\n\n"
        "Rules:\n"
        "- Use plain text only\n"
        "- Do not skip or rename any section\n"
        "- Keep title short and punchy\n"
        "- Description must end with a question mark"
    )

    result = subprocess.run(
        ['ollama', 'run', MODEL_NAME, system_prompt + "\n\nSTORY:\n" + story_text],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        timeout=120
    )

    if result.returncode != 0:
        raise Exception(result.stderr.strip())

    return result.stdout.strip()

# ===== Metadata Validation =====
def validate_metadata(raw_text):
    try:
        # title = re.search(r'\[TITLE\]\s*(.+?)(?=\n\[DESCRIPTION\])', raw_text, re.DOTALL).group(1).strip()
        # description = re.search(r'\[DESCRIPTION\]\s*(.+?)(?=\n\[TAGS\])', raw_text, re.DOTALL).group(1).strip()
        # tags = re.search(r'\[TAGS\]\s*(.+)', raw_text, re.DOTALL).group(1).strip()

        # Normalize header spacing
        normalized_text = '\n'.join(line.strip() for line in raw_text.splitlines())

        try:
            title = re.search(r'\[TITLE\]\s*(.+?)(?=\n\[DESCRIPTION\])', normalized_text, re.DOTALL).group(1).strip()
            description = re.search(r'\[DESCRIPTION\]\s*(.+?)(?=\n\[TAGS\])', normalized_text, re.DOTALL).group(1).strip()
            tags = re.search(r'\[TAGS\]\s*(.+)', normalized_text, re.DOTALL).group(1).strip()
        except AttributeError:
            raise ValueError("Metadata validation failed due to missing or malformed sections.")


        # if len(title.split()) > 8:
        #     raise Exception("Title too long")
        # if not description.endswith('?'):
        #     raise Exception("Description must end with '?'")
        # if not (5 <= len(tags.split(',')) <= 8):
        #     raise Exception("Tag count should be 5 to 8")

        return {
            'title': title,
            'description': description,
            'tags': tags
        }
    except Exception as e:
        print_status(f"⚠️ Metadata validation failed: {e}", "warning")
        print_status(f"Raw metadata:\n{raw_text}", "info")
        return None

# ===== Main =====
def process_csv():
    print_status("🚀 Starting Step 2: Metadata Generation", "progress")
    try:
        df = pd.read_csv(CSV_PATH)
        # df = pd.read_csv(CSV_PATH, encoding='utf-8', errors='replace')
        print_status(f"Loaded CSV with {len(df)} rows", "success")
    except Exception as e:
        print_status(f"Failed to read CSV: {str(e)}", "error")
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

            metadata_path = os.path.join(METADATA_DIR, f"metadata_{story_id}.txt")
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

# ===== Run =====
if __name__ == "__main__":
    process_csv()
    print_status("🏁 Step 2 complete", "success")
