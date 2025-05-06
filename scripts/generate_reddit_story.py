import os
import subprocess
import pandas as pd
import hashlib
from datetime import datetime

# ===== Configuration =====
CSV_PATH = 'data/input.csv'
STORY_DIR = 'stories/generated'
MODEL_NAME = 'mistral'
TARGET_WORD_COUNT = 120
WORD_COUNT_TOLERANCE = 20

# ===== Setup Directories =====
os.makedirs(STORY_DIR, exist_ok=True)

# ===== Logging =====
def print_status(msg, status="info"):
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
    print(f"{datetime.now().strftime('%H:%M:%S')} {icons.get(status, '')} {msg}")

# ===== Story Generation =====
def generate_story(prompt):
    hook_list = [
        "I never thought I'd be writing this, but...",
        "This isn't easy to admit, but...",
        "Was I wrong to...?",
        "I found something I wasn't supposed to see...",
        "My hands were shaking when...",
        "It started as an ordinary Tuesday...",
        "We've all been there - except my situation...",
        "The note said not to open it, but..."
    ]
    hook = hook_list[int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(hook_list)]

    system_prompt = (
        f"You are a popular Reddit storyteller known for short, emotional stories.\n"
        f"Write a story (100–200 words) that starts with: '{hook}'\n"
        "End with either:\n"
        "- 'What would you have done? Let me know in the comments!'\n"
        "- 'Like, share, and subscribe for more!'\n"
        "Focus on a single vivid moment. Keep it raw and honest."
    )

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

def validate_story(story):
    word_count = len(story.split())
    # if not (TARGET_WORD_COUNT - WORD_COUNT_TOLERANCE <= word_count <= TARGET_WORD_COUNT + WORD_COUNT_TOLERANCE):
    #     raise Exception(f"Invalid word count: {word_count}")
    # if not any(ending in story for ending in [
    #     "What would you have done? Let me know in the comments!",
    #     "Like, share, and subscribe for more!"
    # ]):
        # raise Exception("Story missing required ending")
    return True

# ===== Main =====
def process_csv():
    print_status("🚀 Starting Step 1: Story Generation", "progress")
    try:
        df = pd.read_csv(CSV_PATH)
        print_status(f"Loaded CSV with {len(df)} rows", "success")
    except Exception as e:
        print_status(f"Failed to read CSV: {str(e)}", "error")
        return

    for col in ['StoryPath', 'StoryText', 'StoryStatus']:
        if col not in df.columns:
            df[col] = ""

    for idx, row in df.iterrows():
        if str(row.get("StoryStatus", "")).lower() == "completed":
            print_status(f"Skipping row {idx}: already completed", "info")
            continue

        prompt = str(row.get("Prompt", "")).strip()
        story_id = row.get("ID", idx)

        print_status(f"✍️ Generating story for ID {story_id}", "progress")
        try:
            story = generate_story(prompt)
            validate_story(story)

            path = os.path.join(STORY_DIR, f"story_{story_id}.txt")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(story)

            df.at[idx, 'StoryText'] = story
            df.at[idx, 'StoryPath'] = path
            df.at[idx, 'StoryStatus'] = "Completed"

            print_status(f"✅ Story saved: {path}", "success")

        except Exception as e:
            print_status(f"❌ Failed to generate story for ID {story_id}: {e}", "error")
            df.at[idx, 'StoryStatus'] = f"Failed: {str(e)}"

    try:
        df.to_csv(CSV_PATH, index=False)
        print_status("✅ CSV updated", "success")
    except Exception as e:
        print_status(f"❌ Failed to save CSV: {str(e)}", "error")

# ===== Run =====
if __name__ == "__main__":
    process_csv()
    print_status("🏁 Step 1 complete", "success")
