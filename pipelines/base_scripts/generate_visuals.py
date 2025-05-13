
import os
import subprocess
import pandas as pd
from datetime import datetime
from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, TextClip
)
from moviepy.video.tools.subtitles import SubtitlesClip
import argparse
import srt
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ["IMAGEMAGICK_BINARY"] = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

# parser = argparse.ArgumentParser(description="Generate videos and queue them for upload.")
parser = argparse.ArgumentParser(description="Generate video visuals from narration and subtitles.")
parser.add_argument("--csv", default="pipelines/datum.csv", help="Path to input CSV")
parser.add_argument("--upload_queue", type=str, help="Path to upload queue CSV", default="data/upload_queue.csv")
parser.add_argument("--category_id", default="27", help="YouTube video category ID")
args = parser.parse_args()

CSV_PATH = args.csv
IMAGE_OUTPUT_DIR = "images/generated"
VIDEO_OUTPUT_DIR = "output/final_videos"
RESOLUTION = "1080x1920"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)


UPLOAD_QUEUE = args.upload_queue
CATEGORY_ID = args.category_id

def queue_video_for_upload(index, video_path, title, description, tags, category_id):
    if os.path.exists(UPLOAD_QUEUE):
        queue_df = pd.read_csv(UPLOAD_QUEUE)
    else:
        queue_df = pd.DataFrame(columns=["Index", "VideoPath", "Title", "Description", "Tags", "UploadStatus", "Timestamp", "CategoryId"])

    for col in ["Index", "VideoPath", "Title", "Description", "Tags", "UploadStatus", "Timestamp", "CategoryId"]:
        if col not in queue_df.columns:
            queue_df[col] = ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_df = pd.concat([queue_df, pd.DataFrame([{
        "Index": index,
        "VideoPath": video_path,
        "Title": title,
        "Description": description,
        "Tags": tags,
        "UploadStatus": "pending",
        "Timestamp": timestamp,
        "CategoryId": category_id
    }])], ignore_index=True)

    queue_df.to_csv(UPLOAD_QUEUE, index=False)

import os
import subprocess
import pandas as pd
from datetime import datetime
from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, TextClip
)
from moviepy.video.tools.subtitles import SubtitlesClip
import argparse
import srt
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ["IMAGEMAGICK_BINARY"] = r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"

parser = argparse.ArgumentParser(description="Generate videos and queue them for upload.")
parser.add_argument("--csv", default="pipelines/datum.csv", help="Path to input CSV")
parser.add_argument("--upload_queue", default="pipelines/upload_queue.csv", help="Path to upload queue CSV")
parser.add_argument("--category_id", default="27", help="YouTube video category ID")
args = parser.parse_args()

CSV_PATH = args.csv
UPLOAD_QUEUE = args.upload_queue
CATEGORY_ID = args.category_id

IMAGE_OUTPUT_DIR = "images/generated"
VIDEO_OUTPUT_DIR = "output/final_videos"
RESOLUTION = "1080x1920"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

def queue_video_for_upload(index, video_path, title, description, tags, category_id):
    if os.path.exists(UPLOAD_QUEUE):
        queue_df = pd.read_csv(UPLOAD_QUEUE)
    else:
        queue_df = pd.DataFrame(columns=["Index", "VideoPath", "Title", "Description", "Tags", "UploadStatus", "Timestamp", "CategoryId"])

    for col in ["Index", "VideoPath", "Title", "Description", "Tags", "UploadStatus", "Timestamp", "CategoryId"]:
        if col not in queue_df.columns:
            queue_df[col] = ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_df = pd.concat([queue_df, pd.DataFrame([{
        "Index": index,
        "VideoPath": video_path,
        "Title": title,
        "Description": description,
        "Tags": tags,
        "UploadStatus": "pending",
        "Timestamp": timestamp,
        "CategoryId": category_id
    }])], ignore_index=True)

    queue_df.to_csv(UPLOAD_QUEUE, index=False)

def parse_srt_to_dataframe(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        contents = f.read()
    parsed = list(srt.parse(contents))
    return pd.DataFrame([{
        "start": item.start.total_seconds(),
        "end": item.end.total_seconds(),
        "text": item.content.replace("\n", " ").strip()
    } for item in parsed])

def generate_prompts_from_subtitles(subtitle_path, story_text):
    df = parse_srt_to_dataframe(subtitle_path)
    print(f"📥 Loaded {len(df)} subtitle lines.")

    generic_pattern = r"(?i)(did you know|what do you think|comment below|stay tuned|"                       r"let us know|share your|subscribe|like and share|what's your opinion|"                       r"don't forget to|here's why|before we start|welcome back|today we|"                       r"without further ado)"

    filtering_prompt = f"""
ROLE: You're a strict video editor selecting ONLY visually rich scenes. REJECT ALL generic phrases!

STORY CONTEXT:
{story_text}

SUBTITLE LINES (EXACT TEXT):
{chr(10).join(df['text'].tolist())}

TASK:
1. FIRST FILTER OUT:
   - Any questions ("did you know", "what do you think")
   - Audience engagement phrases
   - Generic statements without concrete imagery
2. FROM REMAINING: Select 4-8 most visual lines
3. Return EXACT original lines only
4. No explanations, just the lines
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=filtering_prompt,
        capture_output=True, text=True, encoding='utf-8'
    )

    ai_lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()][:8]
    filtered_lines = []
    for line in ai_lines:
        if re.search(generic_pattern, line):
            continue
        original_text = next((t for t in df['text'] if t.strip().lower() == line.strip().lower()), None)
        if original_text and not re.search(generic_pattern, original_text):
            filtered_lines.append(line)

    if len(filtered_lines) < 4:
        print("⚠️ Activating advanced visual phrase detection...")
        df_filtered = df[~df['text'].str.contains(generic_pattern, case=False, regex=True)]
        if len(df_filtered) < 4:
            print("⚠️ Emergency fallback: Using weighted phrase scoring")
            tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            matrix = tfidf.fit_transform(df['text'])
            df['score'] = matrix.sum(axis=1).A1
            selected_df = df.nlargest(8, 'score')
        else:
            tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            matrix = tfidf.fit_transform(df_filtered['text'])
            df_filtered['score'] = matrix.sum(axis=1).A1
            selected_df = df_filtered.nlargest(8, 'score')
        selected_df = selected_df.sort_values('start').head(8)
    else:
        selected_lines = []
        for ai_line in filtered_lines[:8]:
            ai_clean = ai_line.strip().lower()
            for orig_line in df['text']:
                if ai_clean == orig_line.strip().lower():
                    selected_lines.append(orig_line)
                    break
        selected_df = df[df["text"].isin(selected_lines)].sort_values("start")

    selected_df = selected_df[~selected_df['text'].str.contains(generic_pattern, case=False, regex=True)]
    if len(selected_df) < 4:
        raise Exception("Failed to find enough visual scenes after final filtering")

    creative_prompt_request = f"""
ROLE: You're a creative director creating vivid image prompts. Be EXTREMELY imaginative!

INSTRUCTIONS:
1. For each story line below, create a detailed visual description
2. Add dramatic elements, metaphors, or symbolic representations
3. Include specific actions, settings, and visual metaphors
4. Make it visually striking and attention-grabbing

STORY LINES:
{chr(10).join(selected_df['text'].tolist())}

FORMAT:
- One prompt per line in the same order
- Start each with "VISUAL: "
- Keep under 200 characters
"""

    creative_result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=creative_prompt_request,
        capture_output=True, text=True, encoding='utf-8'
    )

    creative_prompts = []
    for line in creative_result.stdout.strip().split('\n'):
        if line.startswith("VISUAL: "):
            creative_prompts.append(line[8:].strip())
        elif creative_prompts:
            creative_prompts[-1] += " " + line.strip()

    if len(creative_prompts) != len(selected_df):
        creative_prompts = selected_df['text'].tolist()
        print("⚠️ Using original text for missing creative prompts")

    prompt_groups = []
    for i, (idx, row) in enumerate(selected_df.iterrows()):
        prompt_groups.append({
            "prompt": creative_prompts[i] if i < len(creative_prompts) else row['text'],
            "subtitle_indices": [idx]
        })

    print(f"🎬 Selected {len(prompt_groups)} scenes with creative prompts")
    return prompt_groups, selected_df

def generate_images(prompt_groups, story_id, df, row_idx):
    hidream_python = r"D:\\shorty\\hidream_env\\Scripts\\python.exe"
    image_paths = []
    subtitle_map = []

    for i, group in enumerate(prompt_groups):
        try:
            fname = f"story_{story_id}_scene_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            output_path = os.path.join(IMAGE_OUTPUT_DIR, fname)
            print(f"🎨 Generating image {i+1}/{len(prompt_groups)}")
            print(f"   Prompt: {group['prompt']}")
            
            subprocess.run([
                hidream_python, "-m", "hdi1", group['prompt'],
                "-m", "fast", "-r", RESOLUTION, "-o", output_path
            ], check=True)
            
            image_paths.append(output_path)
            subtitle_map.append(group["subtitle_indices"])
        except Exception as e:
            print(f"❌ Image gen failed: {e}")
            df.at[row_idx, "ImageStatus"] = f"error: {e}"
            return [], []

    df.at[row_idx, "ImageStatus"] = "completed"
    df.at[row_idx, "ImagePaths"] = "|".join(image_paths)
    df.at[row_idx, "ImageSubtitleMap"] = "|".join([".".join(map(str, group)) for group in subtitle_map])
    return image_paths, subtitle_map

def create_video(images, audio_path, subtitle_path, selected_subs, subtitle_map, output_dir, story_id):
    audio = AudioFileClip(audio_path)
    clips = []

    for img_path, indices in zip(images, subtitle_map):
        line_data = selected_subs.loc[indices[0]]
        clip = ImageClip(img_path).set_start(line_data['start']).set_duration(line_data['end'] - line_data['start'])
        clip = clip.resize(height=1920).set_position(('center', 'center'))
        clips.append(clip)

    endings_dir = os.path.join("images", "endings")
    cta_path = random.choice([os.path.join(endings_dir, f) for f in os.listdir(endings_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    cta_clip = ImageClip(cta_path).set_duration(5).set_start(audio.duration - 5).resize(height=1920)
    clips.append(cta_clip)

    video = CompositeVideoClip(clips).set_audio(audio)

    subtitle_generator = lambda txt: TextClip(
        txt, fontsize=70, font='Impact', color='white',
        stroke_color='black', stroke_width=3, method='caption',
        size=(1000, None), align='center'
    ).margin(top=20, opacity=0).set_position(('center', 0.75))

    subtitles = SubtitlesClip(subtitle_path, subtitle_generator)
    
    final = CompositeVideoClip([video, subtitles.set_position(('center', 'bottom'))])
    
    out_path = os.path.join(output_dir, f"story_{story_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    final.write_videofile(out_path, fps=30, codec="libx264", audio_codec="aac", threads=4, preset="ultrafast")
    return out_path

def process_stories():
    df = pd.read_csv(CSV_PATH)
    if "ImageStatus" not in df.columns:
        df["ImageStatus"] = ""
    if "ImagePaths" not in df.columns:
        df["ImagePaths"] = ""

    for idx, row in df.iterrows():
        if str(row.get("VideoStatus", "")).lower() == "completed":
            print(f"✅ Skipping ID {row['ID']}")
            continue

        story_id = row["ID"]
        try:
            print(f"\n🚀 Processing ID {story_id} — {row.get('Title', '')}")
            with open(row["StoryPath"], 'r', encoding='utf-8') as f:
                story_text = f.read()

            prompt_groups, selected_subs = generate_prompts_from_subtitles(row["SubtitlePath"], story_text)
            images, subtitle_map = generate_images(prompt_groups, story_id, df, idx)
            if not images:
                raise Exception("Image generation failed")

            video_path = create_video(images, row["AudioPath"], row["SubtitlePath"], selected_subs, subtitle_map, VIDEO_OUTPUT_DIR, story_id)
            df.at[idx, "VideoPath"] = video_path
            df.at[idx, "VideoStatus"] = "completed"

            queue_video_for_upload(story_id, video_path, row.get("Title", ""), row.get("Description", ""), row.get("Tags", ""), CATEGORY_ID)


            queue_video_for_upload(story_id, video_path, row.get("Title", ""), row.get("Description", ""), row.get("Tags", ""), CATEGORY_ID)
        except Exception as e:
            df.at[idx, "VideoStatus"] = f"error: {e}"
            print(f"❌ Error for ID {story_id}: {e}")

    df.to_csv(CSV_PATH, index=False)
    print("✅ CSV updated")

if __name__ == "__main__":
    process_stories()
