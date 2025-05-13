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

# df = pd.read_csv(CSV_PATH)
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
#         queue_df = pd.read_csv(UPLOAD_QUEUE)
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
#     for i, row in pending_rows.iterrows():
#         video_id, output_path, title, description, tags, status, error = process_video(row)
#         df.at[i, "VideoStatus"] = status
#         if output_path:
#             df.at[i, "VideoPath"] = output_path
#             queue_video_for_upload(video_id, output_path, title, description, tags, CATEGORY_ID)
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

df = pd.read_csv(CSV_PATH)
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

def process_video(row):
    video_id = row["ID"]
    try:
        audio_path = row["AudioPath"]
        subtitle_path = row["SubtitlePath"]
        title = str(row.get("Title", f"Video {video_id}"))
        description = str(row.get("Description", title))
        tags = str(row.get("Tags", "shorts,education"))

        stock_videos = [f for f in os.listdir(STOCK_VIDEO_DIR) if f.endswith(('.mp4', '.mov'))]
        if len(stock_videos) < 2:
            raise Exception("Need at least 2 stock videos.")
        random.shuffle(stock_videos)

        audio = AudioFileClip(audio_path)
        required_duration = audio.duration

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

        full_video = concatenate_videoclips(segments).set_audio(audio)
        subtitles = SubtitlesClip(subtitle_path, generate_subtitle)
        final = CompositeVideoClip([full_video, subtitles.set_pos(("center", "center"))])
        final = final.set_duration(required_duration)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{video_id}_{timestamp}.mp4"
        output_path = os.path.join(VIDEO_OUTPUT_DIR, safe_filename(filename))

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
        return video_id, None, None, None, None, f"error: {e}", e

def main():
    pending_rows = df[df["VideoStatus"].astype(str).str.lower() != "completed"]

    results = []
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

    df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    main()
