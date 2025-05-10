# import os
# import subprocess
# import pandas as pd

# # ===== Config =====
# CSV_PATH = 'data/input.csv'
# STORY_DIR = 'stories/generated'
# MODEL_NAME = 'mistral'  # or llama2, gemma, etc.

# # ===== Ensure Output Directory Exists =====
# os.makedirs(STORY_DIR, exist_ok=True)

# def generate_story_with_ollama(prompt):
#     result = subprocess.run(
#         ['ollama', 'run', MODEL_NAME],
#         input=f"Write a short, engaging story under 60 seconds based on this prompt:\n\n{prompt}",
#         text=True,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         encoding='utf-8'
#     )
#     return result.stdout.strip()

# def process_csv():
#     df = pd.read_csv(CSV_PATH)

#     for i, row in df.iterrows():
#         # Skip if story is already marked as completed
#         if str(row.get('StoryStatus')).strip().lower() == 'completed':
#             continue

#         prompt = row.get('Prompt', '').strip()
#         story_id = row.get('ID')

#         if not prompt or pd.isna(story_id):
#             continue

#         try:
#             print(f"📝 Generating story for ID {story_id}: {prompt}")
#             story = generate_story_with_ollama(prompt)

#             story_filename = f"story_{story_id}.txt"
#             story_path = os.path.join(STORY_DIR, story_filename)
#             with open(story_path, "w", encoding="utf-8") as f:
#                 f.write(story)

#             df.at[i, 'StoryPath'] = story_path
#             df.at[i, 'StoryStatus'] = 'completed'
#             print(f"✅ Story saved at: {story_path}")

#         except Exception as e:
#             df.at[i, 'StoryStatus'] = f"error: {str(e)}"
#             print(f"❌ Error generating story for ID {story_id}: {e}")

#     # ===== Save Updated CSV =====
#     try:
#         df.to_csv(CSV_PATH, index=False)
#         print(f"📦 Story generation complete. CSV updated at {CSV_PATH}")
#     except PermissionError:
#         print(f"⚠️ Failed to save CSV. Please close '{CSV_PATH}' if it's open in another program.")

# if __name__ == "__main__":
#     process_csv()

import os
import subprocess
import pandas as pd
import re

CSV_PATH = 'data/input.csv'
STORY_DIR = 'stories/generated'
MODEL_NAME = 'mistral'

os.makedirs(STORY_DIR, exist_ok=True)

def generate_story_with_metadata(prompt):
    system_prompt = (
        "You are an expert YouTube storyteller and hook copywriter.\n"
        "Given a topic, you will:\n"
        "1. Write a compelling story in under 30 seconds with a proper ending.\n"
        "2. Generate an engaging YouTube title using pattern interrupts and power phrases.\n"
        "3. Generate a viral description using curiosity gaps and psychological hooks.\n"
        "4. Suggest comma-separated tags.\n"
        "Output MUST follow this exact format:\n"
        "### STORY\n"
        "[your short story here]\n\n"
        "### TITLE\n"
        "[your viral title]\n\n"
        "### DESCRIPTION\n"
        "[your optimized description]\n\n"
        "### TAGS\n"
        "[comma,separated,tags]"
    )

    full_prompt = f"{system_prompt}\n\n### TOPIC\n{prompt}"

    result = subprocess.run(
        ['ollama', 'run', MODEL_NAME],
        input=full_prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8'
    )
    return result.stdout.strip()

def extract_section(text, section):
    pattern = rf"### {section.upper()}\n(.*?)(?=\n### |\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def process_csv():
    # df = pd.read_csv(CSV_PATH)
    df = pd.read_csv(CSV_PATH, encoding='utf-8', errors='replace')

    for i, row in df.iterrows():
        if str(row.get("StoryStatus", "")).strip().lower() == "completed":
            continue

        prompt = row['Prompt']
        print(f"📝 Generating story for: {prompt}")
        response = generate_story_with_metadata(prompt)

        story = extract_section(response, "STORY")
        title = extract_section(response, "TITLE")
        description = extract_section(response, "DESCRIPTION")
        tags = extract_section(response, "TAGS")

        if not story:
            print(f"❌ Skipping ID {row['ID']}: No story generated.")
            continue

        story_filename = f"story_{row['ID']}.txt"
        story_path = os.path.join(STORY_DIR, story_filename)

        with open(story_path, "w", encoding="utf-8") as f:
            f.write(story)

        df.at[i, 'StoryPath'] = story_path
        df.at[i, 'StoryStatus'] = 'Completed'
        df.at[i, 'Title'] = title
        df.at[i, 'Description'] = description
        df.at[i, 'Tags'] = tags

    df.to_csv(CSV_PATH, index=False)
    print("✅ Story generation complete and CSV updated.")

if __name__ == "__main__":
    process_csv()
