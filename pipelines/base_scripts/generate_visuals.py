# === HiDream-Full Pipeline with Prompt Optimization ===
import os
import subprocess
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import CLIPTokenizerFast, LlamaForCausalLM, PreTrainedTokenizerFast
from diffusers import HiDreamImagePipeline
from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, TextClip
)
from moviepy.video.tools.subtitles import SubtitlesClip
import warnings
warnings.filterwarnings("ignore")
import triton
print(triton.__version__)


# === Paths ===
CSV_PATH = "pipelines/datum.csv"
IMAGE_OUTPUT_DIR = "images/generated"
VIDEO_OUTPUT_DIR = "output/final_videos"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# === HiDream + LLaMA3 Setup ===
HIDREAM_MODEL = "HiDream-ai/HiDream-I1-Full"
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    LLAMA_MODEL,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

pipe = HiDreamImagePipeline.from_pretrained(
    HIDREAM_MODEL,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    torch_dtype=torch.bfloat16
).to("cuda")

# === Prompt Tokenizer ===
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
def count_tokens(text): return len(tokenizer(text)["input_ids"])

# === Enhance Prompt ===
def enhance_prompt(raw_prompt):
    parts = raw_prompt.split("|")
    if len(parts) < 5:
        return raw_prompt
    subject, action, style, lighting, details = [x.strip() for x in parts[:5]]
    enhanced = f"{subject}, {action}, {style}, {lighting}, {details}, cinematic, 8K, ultra detailed, volumetric light"
    return enhanced

# === Generate Prompts ===
def generate_prompts(story_text):
    system_prompt = """Create 3 HiDream prompts from this news story using:
    Format: [Subject|Action|Style|Lighting|Details]
    Style should be cinematic or photo-realistic.
    Use | instead of commas."""
    result = subprocess.run(
        ['ollama', 'run', 'llama3', system_prompt + "\nSTORY:\n" + story_text],
        capture_output=True, text=True, encoding='utf-8'
    )
    raw_lines = [line.strip() for line in result.stdout.split('\n') if "|" in line]
    prompts = []
    for p in raw_lines[:3]:
        enhanced = enhance_prompt(p)
        if count_tokens(enhanced) <= 115:
            prompts.append(enhanced)
    return prompts or [
        "Diplomatic standoff, cinematic lighting, high detail, UN background, volumetric fog",
        "Military escalation map, realistic infographic style, glowing red zones, data overlays",
        "High-tension negotiation table, cinematic mood, deep shadows, anxious faces"
    ]

# === Generate Image ===
def generate_images(prompts, story_id):
    images = []
    for i, prompt in enumerate(prompts):
        try:
            print(f"\n🎨 Generating: {prompt}")
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=5.0,
                num_inference_steps=50,
                generator=torch.Generator("cuda").manual_seed(0)
            ).images[0]
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            fname = f"story_{story_id}_scene_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = os.path.join(IMAGE_OUTPUT_DIR, fname)
            image.save(path)
            images.append(path)
        except Exception as e:
            print(f"❌ Failed: {e}")
    return images

# === Video Creation ===
def create_video(images, audio_path, subtitle_path, output_dir, story_id):
    audio = AudioFileClip(audio_path)
    duration = audio.duration / len(images)
    clips = [ImageClip(p).set_duration(duration).resize(height=1080) for p in images]
    video = concatenate_videoclips(clips).set_audio(audio)
    subtitles = SubtitlesClip(subtitle_path, lambda txt: TextClip(
        txt, fontsize=70, font='Arial-Bold', color='white',
        stroke_color='black', stroke_width=2, size=(900, None), method='caption'))
    final = CompositeVideoClip([video, subtitles.set_position(('center', 'bottom'))])
    out_path = os.path.join(output_dir, f"story_{story_id}.mp4")
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", threads=4, preset='ultrafast', ffmpeg_params=['-crf', '22'])
    return out_path

# === Process Stories ===
def process_stories():
    df = pd.read_csv(CSV_PATH)
    for idx, row in df.iterrows():
        if str(row.get("VideoStatus", "")).lower() == "completed":
            continue
        story_id = row["ID"]
        try:
            with open(row["StoryPath"], 'r', encoding='utf-8') as f:
                story_text = f.read()
            prompts = generate_prompts(story_text)
            images = generate_images(prompts, story_id)
            if not images:
                raise Exception("No images generated")
            video_path = create_video(images, row["AudioPath"], row["SubtitlePath"], VIDEO_OUTPUT_DIR, story_id)
            df.at[idx, "VideoPath"] = video_path
            df.at[idx, "VideoStatus"] = "completed"
            print(f"✅ Done: {video_path}")
        except Exception as e:
            df.at[idx, "VideoStatus"] = f"error: {e}"
            print(f"❌ Error in story {story_id}: {e}")
    df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    process_stories()
