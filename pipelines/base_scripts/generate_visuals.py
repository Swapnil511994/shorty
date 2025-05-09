import os
import subprocess
import pandas as pd
from datetime import datetime
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from peft import LoraConfig, inject_adapter_in_model
import warnings
import argparse

# Suppress warnings
warnings.filterwarnings("ignore")

# ===== Configuration =====
CSV_PATH = "pipelines/datum.csv"  # Same as your pipeline
IMAGE_OUTPUT_DIR = "images/generated"
VIDEO_OUTPUT_DIR = "output/final_videos"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description="Generate story visuals.")
parser.add_argument("--csv", type=str, help="Path to input CSV", default=os.getenv("CSV_PATH", "pipelines/input.csv"))
args = parser.parse_args()
CSV_PATH = args.csv

# SDXL Models
MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
LCM_LORA = "latent-consistency/lcm-lora-sdxl"

# ===== Initialize SDXL Pipeline =====
def init_sdxl_pipeline():
    print("🚀 Initializing SDXL pipeline...")
    
    # Hardware optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_BASE,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

    # LCM-LoRA injection
    try:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        inject_adapter_in_model(lora_config, pipe.unet)
        pipe.load_lora_weights(LCM_LORA)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        print("⚡ LCM-LoRA successfully injected!")
    except Exception as e:
        print(f"⚠️ LoRA injection failed: {str(e)}")
    
    # Performance optimizations
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    
    return pipe

# ===== Generate Prompts with Ollama =====
def generate_image_prompts(story_text):
    """Use Ollama to convert story into visual scene prompts"""
    system_prompt = """You are an expert AI prompt engineer for Stable Diffusion. 
    Convert this news story into 5 visual scenes for a YouTube Shorts video with:
    1. Official/realistic style for institutions
    2. Abstract metaphors for economic concepts
    3. Relatable human elements for consumer impact
    Return exactly 5 prompts, one per line, in this format:
    "Scene: [description], [art style], [lighting], [detail level]"
    """
    
    result = subprocess.run(
        ['ollama', 'run', 'llama3', system_prompt + "\n\nSTORY:\n" + story_text],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    if result.returncode != 0:
        raise Exception(f"Ollama error: {result.stderr}")
    
    # Extract prompts from output
    prompts = [line for line in result.stdout.split('\n') if line.strip() and "Scene:" in line]
    return prompts[:5]  # Return max 5 scenes

# ===== Generate Images =====
def generate_images(pipe, prompts, story_id):
    """Generate images for each prompt"""
    images = []
    for i, prompt in enumerate(prompts):
        try:
            print(f"🎨 Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
            image = pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, bad anatomy",
                num_inference_steps=8,
                guidance_scale=2.0,
                height=1920,
                width=1080  # Vertical format for Shorts
            ).images[0]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(IMAGE_OUTPUT_DIR, f"story_{story_id}_scene_{i}_{timestamp}.png")
            image.save(img_path)
            images.append(img_path)
        except Exception as e:
            print(f"⚠️ Failed to generate image: {str(e)}")
    
    return images

# ===== Create Video from Images =====
def create_video(images, audio_path, subtitle_path, output_dir, story_id):
    """Stitch images into video synced with audio and subtitles"""
    from moviepy.editor import (
        ImageClip,
        AudioFileClip,
        concatenate_videoclips,
        CompositeVideoClip,
        TextClip
    )
    from moviepy.video.tools.subtitles import SubtitlesClip
    
    # Load audio to get duration
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration
    scene_duration = total_duration / len(images)
    
    # Create image clips
    clips = []
    for img_path in images:
        img = ImageClip(img_path).set_duration(scene_duration)
        clips.append(img)
    
    # Combine with audio
    video = concatenate_videoclips(clips).set_audio(audio)
    
    # Add subtitles
    def generator(txt):
        return TextClip(
            txt,
            font='Arial-Bold',
            fontsize=60,
            color='white',
            stroke_color='black',
            stroke_width=2,
            size=(900, None),
            method='caption',
            align='center'
        )
    
    subtitles = SubtitlesClip(subtitle_path, generator)
    final = CompositeVideoClip([video, subtitles.set_position(('center', 'bottom'))])
    
    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"story_{story_id}_{timestamp}.mp4")
    
    final.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset='ultrafast'
    )
    
    return output_path

# ===== Main Pipeline Integration =====
def process_stories():
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    
    # Initialize SDXL
    pipe = init_sdxl_pipeline()
    
    for idx, row in df.iterrows():
        if str(row.get("VideoStatus", "")).lower() == "completed":
            continue
            
        story_id = row["ID"]
        story_path = row.get("StoryPath")
        audio_path = row.get("AudioPath")
        subtitle_path = row.get("SubtitlePath")
        
        if not all([story_path, audio_path, subtitle_path]):
            print(f"⚠️ Skipping incomplete row {story_id}")
            continue
            
        try:
            # Read story
            with open(story_path, 'r', encoding='utf-8') as f:
                story_text = f.read()
            
            # Generate image prompts
            print(f"\n📝 Generating prompts for story {story_id}...")
            prompts = generate_image_prompts(story_text)
            print("Generated Prompts:")
            for p in prompts:
                print(f"- {p[:80]}...")
            
            # Generate images
            image_paths = generate_images(pipe, prompts, story_id)
            if not image_paths:
                raise Exception("No images generated")
            
            # Create video
            print("\n🎬 Creating video...")
            video_path = create_video(
                image_paths,
                audio_path,
                subtitle_path,
                VIDEO_OUTPUT_DIR,
                story_id
            )
            
            # Update CSV
            df.at[idx, "VideoPath"] = video_path
            df.at[idx, "VideoStatus"] = "completed"
            print(f"✅ Successfully created video: {video_path}")
            
        except Exception as e:
            df.at[idx, "VideoStatus"] = f"error: {str(e)}"
            print(f"❌ Failed to process story {story_id}: {str(e)}")
    
    # Save CSV
    df.to_csv(CSV_PATH, index=False)
    print("\n🎉 Pipeline completed!")

if __name__ == "__main__":
    process_stories()