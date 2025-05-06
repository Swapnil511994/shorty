# import os
# import pandas as pd
# from moviepy.editor import (
#     VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
# )
# from moviepy.video.tools.subtitles import SubtitlesClip
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import threading

# # Thread-safe print
# print_lock = threading.Lock()
# def safe_print(*args, **kwargs):
#     with print_lock:
#         print(*args, **kwargs)

# # Paths
# CSV_PATH = "data/input.csv"
# VIDEO_OUTPUT_DIR = "output/final_videos"
# STOCK_VIDEO_DIR = "video/backgrounds"
# AUDIO_DIR = "audio/narrations"
# SUBTITLE_DIR = "subtitles/srt"

# # Create output directory if not exists
# os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# # Load CSV
# df = pd.read_csv(CSV_PATH)

# # Styled Subtitle Generator
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

# # Process a single video
# def process_video(row):
#     video_id = row["ID"]
#     try:
#         audio_path = row["AudioPath"]
#         subtitle_path = row["SubtitlePath"]

#         safe_print(f"\n🎬 Processing ID {video_id}...")
#         safe_print(f"📝 Subtitle file: {subtitle_path}")

#         # Pick stock video
#         stock_videos = [f for f in os.listdir(STOCK_VIDEO_DIR) if f.endswith(('.mp4', '.mov'))]
#         if not stock_videos:
#             raise Exception("No stock videos found.")
#         selected_video = stock_videos[video_id % len(stock_videos)]
#         stock_video_path = os.path.join(STOCK_VIDEO_DIR, selected_video)

#         # Load assets
#         audio = AudioFileClip(audio_path)
#         base_clip = VideoFileClip(stock_video_path)
#         required_duration = audio.duration

#         # Loop video
#         loop_count = int(required_duration // base_clip.duration) + 1
#         looped_video = concatenate_videoclips([base_clip] * loop_count).subclip(0, required_duration)
#         looped_video = looped_video.set_audio(audio)

#         # Add subtitles
#         subtitles = SubtitlesClip(subtitle_path, generate_subtitle)
#         final = CompositeVideoClip([looped_video, subtitles.set_pos(("center", "center"))])
#         final = final.set_duration(required_duration)

#         # Export
#         output_path = os.path.join(VIDEO_OUTPUT_DIR, f"video_{video_id}.mp4")
#         final.write_videofile(
#             output_path,
#             codec='libx264',
#             audio_codec='aac',
#             preset='ultrafast',
#             threads=4,
#             fps=30
#         )

#         safe_print(f"✅ Video created for ID {video_id}: {output_path}")
#         return (video_id, output_path, "completed", None)

#     except Exception as e:
#         safe_print(f"❌ Error processing ID {video_id}: {e}")
#         return (video_id, None, f"error: {e}", e)

# # Main processing function
# def process_all_videos():
#     # Filter only pending videos
#     pending_rows = df[df["VideoStatus"].str.lower() != "completed"].to_dict('records')
    
#     # Determine optimal thread count (leave 2 cores free)
#     max_workers = min(8, (os.cpu_count() or 4) - 2)
#     safe_print(f"\n🚀 Processing {len(pending_rows)} videos with {max_workers} threads...")

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {executor.submit(process_video, row): row["ID"] for row in pending_rows}
        
#         for future in as_completed(futures):
#             video_id, output_path, status, error = future.result()
            
#             # Update DataFrame
#             idx = df[df["ID"] == video_id].index[0]
#             if output_path:
#                 df.at[idx, "VideoPath"] = output_path
#             df.at[idx, "VideoStatus"] = status

#     # Save CSV once after all processing
#     df.to_csv(CSV_PATH, index=False)
#     safe_print("\n📦 All videos processed and CSV updated.")

# if __name__ == "__main__":
#     process_all_videos()



import os
import random
import pandas as pd
from moviepy.editor import (
    VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
)
from moviepy.video.tools.subtitles import SubtitlesClip
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Thread-safe print
print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# Paths
CSV_PATH = "data/input.csv"
VIDEO_OUTPUT_DIR = "output/final_videos"
STOCK_VIDEO_DIR = "video/backgrounds"
AUDIO_DIR = "audio/narrations"
SUBTITLE_DIR = "subtitles/srt"

# Fixed output resolution for YouTube Shorts
TARGET_RESOLUTION = (1080, 1920)

# Create output directory if not exists
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH)

# Styled Subtitle Generator
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

# Process a single video
def process_video(row):
    video_id = row["ID"]
    try:
        audio_path = row["AudioPath"]
        subtitle_path = row["SubtitlePath"]

        safe_print(f"\n🎬 Processing ID {video_id}...")
        safe_print(f"📝 Subtitle file: {subtitle_path}")

        # Pick random stock videos
        stock_videos = [f for f in os.listdir(STOCK_VIDEO_DIR) if f.endswith(('.mp4', '.mov'))]
        if len(stock_videos) < 2:
            raise Exception("Need at least 2 unique stock videos to avoid repetition.")
        random.shuffle(stock_videos)

        # Load audio
        audio = AudioFileClip(audio_path)
        required_duration = audio.duration

        # Pick and assemble video clips to match duration
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
            raise Exception("Not enough video duration available to cover the audio.")

        full_video = concatenate_videoclips(segments).set_audio(audio)

        # Add subtitles
        subtitles = SubtitlesClip(subtitle_path, generate_subtitle)
        final = CompositeVideoClip([full_video, subtitles.set_pos(("center", "center"))])
        final = final.set_duration(required_duration)

        # Export
        output_path = os.path.join(VIDEO_OUTPUT_DIR, f"video_{video_id}.mp4")
        final.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',
            threads=4,
            fps=30
        )

        safe_print(f"✅ Video created for ID {video_id}: {output_path}")
        return (video_id, output_path, "completed", None)

    except Exception as e:
        safe_print(f"❌ Error processing ID {video_id}: {e}")
        return (video_id, None, f"error: {e}", e)

# Main processing function
def process_all_videos():
    # Filter only pending videos
    pending_rows = df[df["VideoStatus"].str.lower() != "completed"].to_dict('records')

    # Determine optimal thread count (leave 2 cores free)
    max_workers = min(8, (os.cpu_count() or 4) - 2)
    safe_print(f"\n🚀 Processing {len(pending_rows)} videos with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, row): row["ID"] for row in pending_rows}

        for future in as_completed(futures):
            video_id, output_path, status, error = future.result()
            idx = df[df["ID"] == video_id].index[0]
            if output_path:
                df.at[idx, "VideoPath"] = output_path
            df.at[idx, "VideoStatus"] = status

    df.to_csv(CSV_PATH, index=False)
    safe_print("\n📦 All videos processed and CSV updated.")

if __name__ == "__main__":
    process_all_videos()
