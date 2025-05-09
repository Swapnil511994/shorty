# === SDXL/HiDream with Variant-Flexible Fallback ===
import os
import subprocess
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import CLIPTokenizerFast
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    LCMScheduler,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline
)
from peft import LoraConfig, inject_adapter_in_model
import warnings
warnings.filterwarnings("ignore")

# === Configuration ===
CSV_PATH = "pipelines/datum.csv"
IMAGE_OUTPUT_DIR = "images/generated"
VIDEO_OUTPUT_DIR = "output/final_videos"
STYLES_CONFIG = "pipelines/styles.json"
CONTROLNET_TYPE = "canny"

os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

MODEL_BASES = [
    "HiDream-ai/HiDream-I1-Full",
    "HiDream-ai/HiDream-I1-Dev",
    "stabilityai/stable-diffusion-xl-base-1.0"
]
MODEL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
LCM_LORA = "latent-consistency/lcm-lora-sdxl"

with open(STYLES_CONFIG, 'r') as f:
    STYLE_MAP = json.load(f)

tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
def count_tokens(text): return len(tokenizer(text)["input_ids"])

def enhance_prompt(p):
    parts = p.split("|")
    if len(parts) < 5: return p
    subject, action, style, lighting, details = [x.strip() for x in parts[:5]]
    artist_line = STYLE_MAP.get(style.lower(), {}).get("style_prompt", "")
    visual = f"{subject}, {action}, {lighting}, {details}, {artist_line}, 8K, ultra detailed"
    return visual, style.lower()

def generate_prompts(story_text):
    prompt_cmd = """Create 3 SDXL prompts from this news story using:
    Format: [Subject|Action|Style|Lighting|Details]"""
    result = subprocess.run(['ollama', 'run', 'llama3', prompt_cmd + "\nSTORY:\n" + story_text], capture_output=True, text=True, encoding='utf-8')
    raw_prompts = [line.strip() for line in result.stdout.split('\n') if "|" in line]
    final = []
    for p in raw_prompts[:3]:
        prompt, style = enhance_prompt(p)
        if count_tokens(prompt) <= 77:
            final.append((prompt, style))
    if not final:
        final = [
            ("Tense cyberpunk meeting between diplomats, neon AR fog, red holo screens", "cyberpunk"),
            ("Geopolitical map glowing with red zones, infographic style with data overlays", "infographic"),
            ("Cinematic world leaders clashing over war timeline, 35mm film lighting", "cinematic")
        ]
    return final

def init_pipeline(style):
    print(f"\n🔁 Loading pipeline for style: {style}")
    controlnet_required = style in STYLE_MAP and STYLE_MAP[style].get("use_controlnet", False)
    pipe = None
    for model_base in MODEL_BASES:
        try:
            variant = "fp16" if "stabilityai" in model_base else None
            if controlnet_required:
                controlnet = ControlNetModel.from_pretrained(
                    f"diffusers/controlnet-sdxl-{CONTROLNET_TYPE}", torch_dtype=torch.float16
                )
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    model_base, controlnet=controlnet, torch_dtype=torch.float16, variant=variant
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_base, torch_dtype=torch.float16, variant=variant
                )
            pipe.to("cuda")
            print(f"✅ Loaded base model: {model_base}")
            break
        except Exception as e:
            print(f"⚠️ Failed to load model {model_base}: {e}")
            continue
    if not pipe:
        raise EnvironmentError("No working base model could be loaded.")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    try:
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["to_k", "to_q", "to_v", "to_out.0"], lora_dropout=0.0, bias="none")
        inject_adapter_in_model(lora_config, pipe.unet)
        pipe.load_lora_weights(LCM_LORA)
    except Exception as e:
        print(f"⚠️ Skipping LCM-LoRA: {e}")
    lora_info = STYLE_MAP.get(style, {}).get("lora")
    if lora_info:
        try:
            pipe.load_lora_weights(lora_info["repo"], weight_name=lora_info["filename"], adapter_name=style)
            pipe.set_adapters([style], adapter_weights=[lora_info.get("strength", 0.5)])
        except Exception as e:
            print(f"⚠️ Skipping style LoRA '{style}': {e}")
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    return pipe

def enhance(image):
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return Image.fromarray(np.array(image))

def generate_images(pipe, prompts, story_id):
    images = []
    for i, (prompt, style) in enumerate(prompts):
        print(f"\n🎨 Prompt {i+1}: {prompt}")
        try:
            result = pipe(prompt=prompt, negative_prompt="blurry, low detail, malformed", num_inference_steps=8, guidance_scale=2.5, height=1024, width=1024)
            image = enhance(result.images[0])
            fname = f"story_{story_id}_scene_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = os.path.join(IMAGE_OUTPUT_DIR, fname)
            image.save(path)
            images.append(path)
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    return images

def create_video(images, audio_path, subtitle_path, output_dir, story_id):
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
    from moviepy.video.tools.subtitles import SubtitlesClip
    audio = AudioFileClip(audio_path)
    duration = audio.duration / len(images)
    clips = [ImageClip(p).set_duration(duration).resize(height=1080) for p in images]
    video = concatenate_videoclips(clips).set_audio(audio)
    subtitles = SubtitlesClip(subtitle_path, lambda txt: TextClip(txt, fontsize=70, font='Arial-Bold', color='white', stroke_color='black', stroke_width=2, size=(900, None), method='caption'))
    final = CompositeVideoClip([video, subtitles.set_position(('center', 'bottom'))])
    out_path = os.path.join(output_dir, f"story_{story_id}.mp4")
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", threads=4, preset='ultrafast', ffmpeg_params=['-crf', '22'])
    return out_path

def process_stories():
    df = pd.read_csv(CSV_PATH)
    for idx, row in df.iterrows():
        if str(row.get("VideoStatus", "")).lower() == "completed":
            continue
        story_id = row["ID"]
        try:
            with open(row["StoryPath"], 'r') as f:
                story_text = f.read()
            prompts = generate_prompts(story_text)
            if not prompts:
                raise ValueError("No valid prompts")
            style = prompts[0][1]
            pipe = init_pipeline(style)
            images = generate_images(pipe, prompts, story_id)
            if not images:
                raise Exception("No images generated")
            video_path = create_video(images, row["AudioPath"], row["SubtitlePath"], VIDEO_OUTPUT_DIR, story_id)
            df.at[idx, "VideoPath"] = video_path
            df.at[idx, "VideoStatus"] = "completed"
            print(f"✅ Video created: {video_path}")
        except Exception as e:
            df.at[idx, "VideoStatus"] = f"error: {e}"
            print(f"❌ Failed: {e}")
    df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    process_stories()
