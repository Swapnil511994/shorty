# import os
# import random
# import pandas as pd
# from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
# from moviepy.video.tools.subtitles import SubtitlesClip
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from datetime import datetime
# import argparse

# # === Argument Parser ===
# parser = argparse.ArgumentParser(description="Generate videos and queue them for upload.")
# parser.add_argument("--csv", default="pipelines/input.csv", help="Path to input CSV")
# parser.add_argument("--upload_queue", default="pipelines/upload_queue.csv", help="Path to upload queue CSV")
# parser.add_argument("--category_id", default="27", help="YouTube video category ID")
# args = parser.parse_args()

# CSV_PATH = args.csv
# UPLOAD_QUEUE = args.upload_queue
# CATEGORY_ID = args.category_id

# VIDEO_OUTPUT_DIR = "output/final_videos"
# STOCK_VIDEO_DIR = "video/backgrounds"
# AUDIO_DIR = "audio/narrations"
# SUBTITLE_DIR = "subtitles/srt"
# TARGET_RESOLUTION = (1080, 1920)

# os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
# os.makedirs(os.path.dirname(UPLOAD_QUEUE), exist_ok=True)

# df = pd.read_csv(CSV_PATH, encoding='utf-8')
# for col in ['VideoStatus', 'VideoPath']:
#     if col not in df.columns:
#         df[col] = "pending"

# def generate_subtitle(txt):
#     return TextClip(
#         txt,
#         font='Arial-Bold',
#         fontsize=100,
#         color='white',
#         stroke_color='black',
#         stroke_width=3,
#         size=(1000, None),
#         method='caption',
#         align='center'
#     )

# def safe_filename(filename):
#     return "_".join(filename.split()).replace(':', '').replace('/', '').replace('\\', '')

# def queue_video_for_upload(index, video_path, title, description, tags, category_id):
#     if os.path.exists(UPLOAD_QUEUE):
#         queue_df = pd.read_csv(UPLOAD_QUEUE, encoding='utf-8')
#     else:
#         queue_df = pd.DataFrame(columns=["Index", "VideoPath", "Title", "Description", "Tags", "UploadStatus", "Timestamp", "CategoryId"])

#     for col in ["Index", "VideoPath", "Title", "Description", "Tags", "UploadStatus", "Timestamp", "CategoryId"]:
#         if col not in queue_df.columns:
#             queue_df[col] = ""

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     queue_df = pd.concat([queue_df, pd.DataFrame([{
#         "Index": index,
#         "VideoPath": video_path,
#         "Title": title,
#         "Description": description,
#         "Tags": tags,
#         "UploadStatus": "pending",
#         "Timestamp": timestamp,
#         "CategoryId": category_id
#     }])], ignore_index=True)

#     queue_df.to_csv(UPLOAD_QUEUE, index=False)

# def process_video(row):
#     video_id = row["ID"]
#     try:
#         audio_path = row["AudioPath"]
#         subtitle_path = row["SubtitlePath"]
#         title = str(row.get("Title", f"Video {video_id}"))
#         description = str(row.get("Description", title))
#         tags = str(row.get("Tags", "shorts,education"))

#         stock_videos = [f for f in os.listdir(STOCK_VIDEO_DIR) if f.endswith(('.mp4', '.mov'))]
#         if len(stock_videos) < 2:
#             raise Exception("Need at least 2 stock videos.")
#         random.shuffle(stock_videos)

#         audio = AudioFileClip(audio_path)
#         required_duration = audio.duration

#         segments = []
#         total_duration = 0
#         for video_file in stock_videos:
#             if total_duration >= required_duration:
#                 break
#             clip = VideoFileClip(os.path.join(STOCK_VIDEO_DIR, video_file)).resize(TARGET_RESOLUTION)
#             segment_duration = min(clip.duration, required_duration - total_duration)
#             segments.append(clip.subclip(0, segment_duration))
#             total_duration += segment_duration

#         if total_duration < required_duration:
#             raise Exception("Insufficient video content to match audio duration.")

#         full_video = concatenate_videoclips(segments).set_audio(audio)
#         subtitles = SubtitlesClip(subtitle_path, generate_subtitle)
#         final = CompositeVideoClip([full_video, subtitles.set_pos(("center", "center"))])
#         final = final.set_duration(required_duration)

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"video_{video_id}_{timestamp}.mp4"
#         output_path = os.path.join(VIDEO_OUTPUT_DIR, safe_filename(filename))

#         final.write_videofile(
#             output_path,
#             codec='libx264',
#             audio_codec='aac',
#             preset='ultrafast',
#             threads=4,
#             fps=30
#         )

#         return video_id, output_path, title, description, tags, "completed", None

#     except Exception as e:
#         return video_id, None, None, None, None, f"error: {e}", e

# def main():
#     pending_rows = df[df["VideoStatus"].astype(str).str.lower() != "completed"]

#     results = []
#     with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
#         futures = {executor.submit(process_video, row): i for i, row in pending_rows.iterrows()}
#         for future in as_completed(futures):
#             i = futures[future]
#             try:
#                 video_id, output_path, title, description, tags, status, error = future.result()
#                 df.at[i, "VideoStatus"] = status
#                 if output_path:
#                     df.at[i, "VideoPath"] = output_path
#                     queue_video_for_upload(video_id, output_path, title, description, tags, CATEGORY_ID)
#             except Exception as e:
#                 df.at[i, "VideoStatus"] = f"error: {e}"

#     df.to_csv(CSV_PATH, index=False)

# if __name__ == "__main__":
#     main()


import os
import random
import pandas as pd
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.tools.subtitles import SubtitlesClip
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import traceback

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Generate videos and queue them for upload.")
parser.add_argument("--csv", default="pipelines/input.csv", help="Path to input CSV")
parser.add_argument("--upload_queue", default="pipelines/upload_queue.csv", help="Path to upload queue CSV")
parser.add_argument("--category_id", default="27", help="YouTube video category ID")
args = parser.parse_args()

CSV_PATH = args.csv
UPLOAD_QUEUE = args.upload_queue
CATEGORY_ID = args.category_id

VIDEO_OUTPUT_DIR = "output/final_videos"
STOCK_VIDEO_DIR = "video/backgrounds"
AUDIO_DIR = "audio/narrations"
SUBTITLE_DIR = "subtitles/srt"
TARGET_RESOLUTION = (1080, 1920)

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(UPLOAD_QUEUE), exist_ok=True)

print(f"📥 Loading input CSV: {CSV_PATH}")
try:
    with open(CSV_PATH, 'r', encoding='utf-8', errors='replace') as f:
        df = pd.read_csv(f)
except Exception as e:
    print(f"❌ Failed to read CSV: {e}")
    raise

for col in ['VideoStatus', 'VideoPath']:
    if col not in df.columns:
        df[col] = "pending"

def generate_subtitle(txt):
    return TextClip(
        txt,
        font='Arial-Bold',
        fontsize=100,
        color='white',
        stroke_color='black',
        stroke_width=3,
        size=(1000, None),
        method='caption',
        align='center'
    )

def safe_filename(filename):
    return "_".join(filename.split()).replace(':', '').replace('/', '').replace('\\', '')

def convert_srt_time(t):
    # Expected format: "HH:MM:SS,mmm"
    try:
        hh, mm, ss_ms = t.split(':')
        ss, ms = ss_ms.split(',')
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000
    except Exception as e:
        print(f"❌ Invalid SRT timestamp: {t} ({e})")
        return 0


def load_srt_utf8(path):
    subs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        i = 0
        while i < len(lines):
            if lines[i].strip().isdigit():
                time_line = lines[i + 1].strip()
                if " --> " not in time_line:
                    i += 1
                    continue
                start, end = time_line.split(" --> ")
                start = convert_srt_time(start)
                end = convert_srt_time(end)
                i += 2
                text = []
                while i < len(lines) and lines[i].strip():
                    text.append(lines[i])
                    i += 1
                subs.append(((start, end), "\n".join(text)))
            i += 1
    except Exception as e:
        print(f"❌ Error reading subtitle file {path}: {e}")
        traceback.print_exc()
    return subs

def queue_video_for_upload(index, video_path, title, description, tags, category_id):
    try:
        if os.path.exists(UPLOAD_QUEUE):
            with open(UPLOAD_QUEUE, 'r', encoding='utf-8', errors='replace') as f:
                queue_df = pd.read_csv(f)
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

        queue_df.to_csv(UPLOAD_QUEUE, index=False, encoding='utf-8')
        print(f"✅ Queued for upload: {video_path}")
    except Exception as e:
        print(f"❌ Failed to update upload queue: {e}")
        traceback.print_exc()

def process_video(row):
    video_id = row["ID"]
    print(f"\n🎬 Processing video ID: {video_id}")

    try:
        audio_path = row["AudioPath"]
        subtitle_path = row["SubtitlePath"]
        title = str(row.get("Title", f"Video {video_id}"))
        description = str(row.get("Description", title))
        tags = str(row.get("Tags", "shorts,education"))

        print(f"🎧 Loading audio: {audio_path}")
        audio = AudioFileClip(audio_path)
        required_duration = audio.duration

        print(f"📦 Loading stock videos from: {STOCK_VIDEO_DIR}")
        stock_videos = [f for f in os.listdir(STOCK_VIDEO_DIR) if f.endswith(('.mp4', '.mov'))]
        if len(stock_videos) < 2:
            raise Exception("Need at least 2 stock videos.")
        random.shuffle(stock_videos)

        print(f"🎞️ Composing video segments...")
        segments = []
        total_duration = 0
        for video_file in stock_videos:
            if total_duration >= required_duration:
                break
            clip = VideoFileClip(os.path.join(STOCK_VIDEO_DIR, video_file)).resize(TARGET_RESOLUTION)
            segment_duration = min(clip.duration, required_duration - total_duration)
            segments.append(clip.subclip(0, segment_duration))
            total_duration += segment_duration

        if total_duration < required_duration:
            raise Exception("Insufficient video content to match audio duration.")

        print(f"🧩 Concatenating and adding audio")
        full_video = concatenate_videoclips(segments).set_audio(audio)

        print(f"📝 Adding subtitles from: {subtitle_path}")
        subtitle_data = load_srt_utf8(subtitle_path)
        subtitles = SubtitlesClip(subtitle_data, generate_subtitle)

        print(f"🎛️ Compositing final video")
        final = CompositeVideoClip([full_video, subtitles.set_pos(("center", "center"))])
        final = final.set_duration(required_duration)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{video_id}_{timestamp}.mp4"
        output_path = os.path.join(VIDEO_OUTPUT_DIR, safe_filename(filename))

        print(f"💾 Writing final video to: {output_path}")
        final.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',
            threads=4,
            fps=30
        )

        return video_id, output_path, title, description, tags, "completed", None

    except Exception as e:
        print(f"❌ Error while processing ID {video_id}: {e}")
        traceback.print_exc()
        return video_id, None, None, None, None, f"error: {e}", e

def main():
    pending_rows = df[df["VideoStatus"].astype(str).str.lower() != "completed"]
    print(f"🧮 Found {len(pending_rows)} videos to process")

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = {executor.submit(process_video, row): i for i, row in pending_rows.iterrows()}
        for future in as_completed(futures):
            i = futures[future]
            try:
                video_id, output_path, title, description, tags, status, error = future.result()
                df.at[i, "VideoStatus"] = status
                if output_path:
                    df.at[i, "VideoPath"] = output_path
                    queue_video_for_upload(video_id, output_path, title, description, tags, CATEGORY_ID)
            except Exception as e:
                df.at[i, "VideoStatus"] = f"error: {e}"
                print(f"❌ Failed in thread: {e}")
                traceback.print_exc()

    df.to_csv(CSV_PATH, index=False, encoding='utf-8')
    print("✅ All video statuses updated in CSV.")

if __name__ == "__main__":
    main()
