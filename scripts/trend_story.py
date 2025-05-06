import os
import subprocess
import pandas as pd
from datetime import datetime

CSV_PATH = "data/input.csv"
STORY_DIR = "stories/generated"
MODEL_NAME = "mistral"

os.makedirs(STORY_DIR, exist_ok=True)

# ===== Logging =====
def print_status(msg, status="info"):
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
    print(f"{datetime.now().strftime('%H:%M:%S')} {icons.get(status, '')} {msg}")

# ===== Prompt Enhancer =====
PROMPT_ENHANCER = [
    ("teacher", "Explain the significance of Teacher Appreciation Day 2025 with key facts and why it's important."),
    ("lindor", "Provide a short biography and recent achievements of Francisco Lindor."),
    ("met gala", "Summarize highlights of the 2025 Met Gala after-party, including celebrities and fashion moments."),
    ("flagran", "Explain what a flagrant foul is in the NBA, and its impact during a Nuggets game."),
    ("hood canal", "Describe the Hood Canal Bridge closure in 2025, its causes, and traffic impact."),
    ("riot", "Give a factual summary of a recent protest that escalated into a riot, including causes and response."),
    ("blackpink", "Share some interesting facts about Lisa from BLACKPINK and her global influence."),
    ("nba", "Provide a quick update on recent surprising moments in the NBA."),
    ("mariners", "Update on the Seattle Mariners' current season, standings, and notable players."),
    ("trump", "Summarize recent public appearances and key news involving Barron Trump."),
    ("celeb", "Provide fun facts or recent updates about [PROMPT] as a celebrity."),
    ("weather", "Explain the weather trends and recent extreme weather alerts in Las Vegas.")
]

def enhance_prompt(raw_prompt):
    raw_lower = raw_prompt.lower()
    for keyword, template in PROMPT_ENHANCER:
        if keyword in raw_lower:
            return template.replace("[PROMPT]", raw_prompt)
    return f"Write an informative short video script based on: {raw_prompt}. Keep it concise, engaging, and under 100 words. End with a line encouraging viewers to like, share, or subscribe."

# ===== Story Generator using local Mistral =====
def generate_story(prompt):
    system_prompt = (
        "You are an expert content creator who writes short, informative, and engaging scripts for YouTube Shorts.\n"
        "Each script should clearly explain the topic, include interesting facts or context, and be understandable by a general audience.\n"
        "The generated out should not contain any descriptors like: [Upbeat Background Music], (Narrator Voice) or anything else of this sorts.\n"
        "Ensure the script can be narrated within 60 seconds and ends with either:\n"
        "- 'Subscribe for more!'\n"
        "- 'Like, share, and subscribe!'\n"
        "Format it for natural voiceover narration."

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

# ===== Main =====
def process_csv():
    print_status("🚀 Starting Step 1: Content Generation", "progress")
    try:
        df = pd.read_csv(CSV_PATH)
        print_status(f"Loaded CSV with {len(df)} rows", "success")
    except Exception as e:
        print_status(f"Failed to read CSV: {str(e)}", "error")
        return

    for col in ['StoryText', 'StoryStatus']:
        if col not in df.columns:
            df[col] = ""

    for idx, row in df.iterrows():
        if str(row.get("StoryStatus", "")).lower() == "completed":
            print_status(f"Skipping row {idx}: already completed", "info")
            continue

        raw_prompt = str(row.get("Prompt", "")).strip()
        story_id = row.get("ID", idx)
        enhanced_prompt = enhance_prompt(raw_prompt)

        print_status(f"📚 Generating script for ID {story_id}\n{enhanced_prompt}", "progress")
        try:
            story = generate_story(enhanced_prompt)

            path = os.path.join(STORY_DIR, f"story_{story_id}.txt")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(story)

            df.at[idx, 'StoryText'] = story
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
