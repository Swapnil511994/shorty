# import os
# import subprocess
# import pandas as pd
# from datetime import datetime
# from PIL import Image, ImageFilter
# import torch
# import numpy as np
# from diffusers import (
#     StableDiffusionXLPipeline,
#     StableDiffusionXLImg2ImgPipeline,
#     LCMScheduler
# )
# from transformers import CLIPTokenizerFast
# from peft import LoraConfig, inject_adapter_in_model
# import warnings

# warnings.filterwarnings("ignore")

# CSV_PATH = "pipelines/datum.csv"
# IMAGE_OUTPUT_DIR = "images/generated"
# VIDEO_OUTPUT_DIR = "output/final_videos"
# os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
# os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
# MODEL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
# LCM_LORA = "latent-consistency/lcm-lora-sdxl"

# tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
# def count_tokens(text): return len(tokenizer(text)["input_ids"])

# def generate_image_prompts(story_text):
#     system_prompt = """Create 3 SDXL prompts from this news story using:
#     Format: [Subject|Action|Style|Lighting|Details]
#     Rules:
#     - Max 70 tokens per prompt
#     - Style: cinematic/cyberpunk/infographic
#     - Use | not commas
#     - Key details only
#     Example: Diplomat meeting|Tense negotiation|Cyberpunk|Neon blue|AR glasses"""

#     result = subprocess.run(
#         ['ollama', 'run', 'llama3', system_prompt + "\nSTORY:\n" + story_text],
#         capture_output=True, text=True, encoding='utf-8'
#     )

#     raw_prompts = [line.strip() for line in result.stdout.split('\n') if "|" in line]
#     final_prompts = []

#     for p in raw_prompts[:3]:
#         if count_tokens(p) > 77:
#             parts = p.split("|")
#             shortened = "|".join(parts[:5])
#             final_prompts.append(shortened)
#         else:
#             final_prompts.append(p)

#     if len(final_prompts) < 3:
#         final_prompts.extend([
#             "Diplomats|Negotiating|Cyberpunk|Neon lights|AR interfaces",
#             "UN chamber|Holographic maps|Sci-fi|Blue lighting|8K",
#             "Officials|Debating|Cinematic|Dramatic|Close-up"
#         ])

#     print("Generated Prompts:")
#     for i, p in enumerate(final_prompts[:3]):
#         print(f"{i+1}. {p} ({count_tokens(p)} tokens)")

#     return final_prompts[:3]

# def init_sdxl_pipelines():
#     print("🚀 Initializing SDXL Turbo engine with refiner...")

#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

#     base = StableDiffusionXLPipeline.from_pretrained(
#         MODEL_BASE,
#         torch_dtype=torch.float16,
#         variant="fp16",
#         use_safetensors=True
#     ).to("cuda")

#     refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#         MODEL_REFINER,
#         torch_dtype=torch.float16,
#         variant="fp16"
#     ).to("cuda")

#     # Inject LCM-LoRA
#     try:
#         lora_config = LoraConfig(
#             r=16, lora_alpha=32,
#             target_modules=["to_k", "to_q", "to_v", "to_out.0"],
#             lora_dropout=0.0, bias="none"
#         )
#         inject_adapter_in_model(lora_config, base.unet)
#         base.load_lora_weights(LCM_LORA)
#         base.scheduler = LCMScheduler.from_config(base.scheduler.config)
#         print("⚡ LCM-LoRA successfully injected!")
#     except Exception as e:
#         print(f"⚠️ LCM injection failed: {e}")

#     try:
#         base.unet = torch.compile(base.unet)
#     except Exception:
#         print("⚠️ torch.compile failed or unavailable")

#     base.enable_vae_slicing()
#     base.enable_model_cpu_offload()
#     refiner.enable_vae_tiling()

#     return base, refiner

# def enhance(image):
#     img = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
#     return Image.fromarray(np.array(img))

# def generate_images(base, refiner, prompts, story_id):
#     images = []
#     for i, prompt in enumerate(prompts):
#         try:
#             print(f"\n🎨 Generating image {i+1}/3")
#             print(f"Prompt: {prompt}")

#             main_prompt = "|".join(prompt.split("|")[:3])
#             style_prompt = "|".join(prompt.split("|")[3:])

#             latents = base(
#                 prompt=main_prompt,
#                 prompt_2=style_prompt,
#                 negative_prompt="blurry,bad anatomy",
#                 negative_prompt_2="deformed,ugly",
#                 num_inference_steps=8,
#                 guidance_scale=2.0,
#                 height=1024,
#                 width=1024,
#                 output_type="latent"
#             ).images

#             image = refiner(
#                 prompt=main_prompt,
#                 image=latents,
#                 strength=0.3,
#                 num_inference_steps=4
#             ).images[0]

#             image = enhance(image)

#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             img_path = os.path.join(IMAGE_OUTPUT_DIR, f"story_{story_id}_scene_{i}_{timestamp}.png")
#             image.save(img_path)
#             images.append(img_path)
#         except Exception as e:
#             print(f"⚠️ Image generation failed: {str(e)}")

#     return images

# def create_video(images, audio_path, subtitle_path, output_dir, story_id):
#     from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
#     from moviepy.video.tools.subtitles import SubtitlesClip

#     try:
#         audio = AudioFileClip(audio_path)
#         duration = audio.duration / len(images)

#         clips = [ImageClip(p).set_duration(duration).resize(height=1080) for p in images]
#         video = concatenate_videoclips(clips).set_audio(audio)

#         subtitles = SubtitlesClip(subtitle_path,
#             lambda txt: TextClip(txt, fontsize=70, font='Arial-Bold', color='white',
#                                  stroke_color='black', stroke_width=2, size=(900, None), method='caption'))

#         final = CompositeVideoClip([video, subtitles.set_position(('center', 'bottom'))])
#         output_path = os.path.join(output_dir, f"story_{story_id}.mp4")
#         final.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac",
#                               threads=4, preset='ultrafast', ffmpeg_params=['-crf', '22'])
#         return output_path
#     except Exception as e:
#         raise Exception(f"Video creation failed: {str(e)}")

# def process_stories():
#     df = pd.read_csv(CSV_PATH)
#     base, refiner = init_sdxl_pipelines()

#     for idx, row in df.iterrows():
#         if str(row.get("VideoStatus", "")).lower() == "completed":
#             continue

#         story_id = row["ID"]
#         try:
#             with open(row["StoryPath"], 'r') as f:
#                 story_text = f.read()

#             prompts = generate_image_prompts(story_text)
#             images = generate_images(base, refiner, prompts, story_id)

#             if not images:
#                 raise ValueError("No images generated")

#             video_path = create_video(
#                 images, row["AudioPath"], row["SubtitlePath"],
#                 VIDEO_OUTPUT_DIR, story_id
#             )

#             df.at[idx, "VideoPath"] = video_path
#             df.at[idx, "VideoStatus"] = "completed"
#             print(f"✅ Success: {video_path}")

#         except Exception as e:
#             df.at[idx, "VideoStatus"] = f"error: {str(e)}"
#             print(f"❌ Failed story {story_id}: {str(e)}")

#     df.to_csv(CSV_PATH, index=False)

# if __name__ == "__main__":
#     process_stories()



import os
import subprocess
import pandas as pd
from datetime import datetime
from PIL import Image, ImageFilter
import torch
import numpy as np
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    LCMScheduler
)
from transformers import CLIPTokenizerFast
from peft import LoraConfig, inject_adapter_in_model
import warnings

warnings.filterwarnings("ignore")

# ===== Paths =====
CSV_PATH = "pipelines/datum.csv"
IMAGE_OUTPUT_DIR = "images/generated"
VIDEO_OUTPUT_DIR = "output/final_videos"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# ===== Models =====
MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
LCM_LORA = "latent-consistency/lcm-lora-sdxl"

tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
def count_tokens(text): return len(tokenizer(text)["input_ids"])

# ===== Prompt Optimizer =====
def optimize_prompt(p):
    parts = p.split("|")
    if len(parts) < 5:
        return p  # Skip if poorly formatted

    subject, action, style, lighting, details = parts[:5]

    # Visual grounding
    subject_map = {
        "Terrorist escalation": "military general or terrorist leader at command table",
        "Regional tensions": "map of conflict zones with red and orange overlays",
        "Global politics": "world leaders in dark-lit press conference or split frame",
        "Diplomats": "Indian and Pakistani ambassadors debating at futuristic UN table",
        "Officials": "silhouetted military figures pointing at digital screens"
    }

    style_tags = {
        "Cyberpunk": "neon lights, AR glasses, holograms, glowing city",
        "Infographic": "isometric layout, labeled graphs, blue/orange zones, data overlays",
        "Cinematic": "shallow depth of field, moody lighting, dramatic framing"
    }

    subject_phrase = subject_map.get(subject.strip(), subject.strip())
    style_phrase = style_tags.get(style.strip(), style.strip())

    full_prompt = f"{subject_phrase}, {action}, {style_phrase}, {lighting}, {details}, 8K, photorealistic, UHD"
    return full_prompt

# ===== Generate Optimized Prompts =====
def generate_image_prompts(story_text):
    system_prompt = """Create 3 SDXL prompts from this news story using:
    Format: [Subject|Action|Style|Lighting|Details]
    Rules:
    - Max 70 tokens per prompt
    - Style: cinematic/cyberpunk/infographic
    - Use | not commas
    - Key details only
    Example: Diplomat meeting|Tense negotiation|Cyberpunk|Neon blue|AR glasses"""

    result = subprocess.run(
        ['ollama', 'run', 'llama3', system_prompt + "\nSTORY:\n" + story_text],
        capture_output=True, text=True, encoding='utf-8'
    )

    raw_prompts = [line.strip() for line in result.stdout.split('\n') if "|" in line]
    final_prompts = []

    for p in raw_prompts[:3]:
        full = optimize_prompt(p)
        if count_tokens(full) <= 77:
            final_prompts.append(full)

    if len(final_prompts) < 3:
        fallback = [
            "Indian diplomat and Pakistani diplomat arguing, red neon chamber, AR glasses, cyberpunk lighting",
            "Futuristic map showing terror networks across Asia, isometric, glowing lines, infographic style",
            "Split-screen cinematic battle of world leaders, dark tone, global escalation, photoreal"
        ]
        final_prompts.extend(fallback[:3 - len(final_prompts)])

    print("Optimized Prompts:")
    for i, p in enumerate(final_prompts):
        print(f"{i+1}. {p} ({count_tokens(p)} tokens)")

    return final_prompts

# ===== SDXL Turbo + Refiner =====
def init_sdxl_pipelines():
    print("🚀 Initializing SDXL Turbo pipeline")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    base = StableDiffusionXLPipeline.from_pretrained(
        MODEL_BASE, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True
    ).to("cuda")

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_REFINER, torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    try:
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0, bias="none"
        )
        inject_adapter_in_model(lora_config, base.unet)
        base.load_lora_weights(LCM_LORA)
        base.scheduler = LCMScheduler.from_config(base.scheduler.config)
        print("⚡ LoRA loaded with LCM scheduler")
    except Exception as e:
        print(f"⚠️ LCM Injection failed: {e}")

    try:
        base.unet = torch.compile(base.unet)
    except:
        print("⚠️ torch.compile failed")

    base.enable_vae_slicing()
    base.enable_model_cpu_offload()
    refiner.enable_vae_tiling()

    return base, refiner

# ===== Image Enhancer =====
def enhance(image):
    img = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return Image.fromarray(np.array(img))

# ===== Generate Images =====
def generate_images(base, refiner, prompts, story_id):
    images = []
    for i, prompt in enumerate(prompts):
        try:
            print(f"\n🎨 Generating image {i+1}")
            print(f"Prompt: {prompt}")

            latents = base(
                prompt=prompt,
                negative_prompt="blurry, distorted, poorly drawn, low detail",
                num_inference_steps=8,
                guidance_scale=2.5,
                height=1024,
                width=1024,
                output_type="latent"
            ).images

            image = refiner(
                prompt=prompt,
                image=latents,
                strength=0.3,
                num_inference_steps=4
            ).images[0]

            image = enhance(image)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(IMAGE_OUTPUT_DIR, f"story_{story_id}_scene_{i}_{ts}.png")
            image.save(path)
            images.append(path)
        except Exception as e:
            print(f"❌ Image {i} failed: {e}")
    return images

# ===== Create Video =====
def create_video(images, audio_path, subtitle_path, output_dir, story_id):
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
    from moviepy.video.tools.subtitles import SubtitlesClip

    audio = AudioFileClip(audio_path)
    duration = audio.duration / len(images)
    clips = [ImageClip(p).set_duration(duration).resize(height=1080) for p in images]
    video = concatenate_videoclips(clips).set_audio(audio)

    subtitles = SubtitlesClip(subtitle_path,
        lambda txt: TextClip(txt, fontsize=70, font='Arial-Bold', color='white',
                             stroke_color='black', stroke_width=2, size=(900, None), method='caption'))

    final = CompositeVideoClip([video, subtitles.set_position(('center', 'bottom'))])
    out_path = os.path.join(output_dir, f"story_{story_id}.mp4")

    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac",
                          threads=4, preset='ultrafast', ffmpeg_params=['-crf', '22'])

    return out_path

# ===== Main Pipeline =====
def process_stories():
    df = pd.read_csv(CSV_PATH)
    base, refiner = init_sdxl_pipelines()

    for idx, row in df.iterrows():
        if str(row.get("VideoStatus", "")).lower() == "completed":
            continue

        story_id = row["ID"]
        try:
            with open(row["StoryPath"], 'r') as f:
                story_text = f.read()

            prompts = generate_image_prompts(story_text)
            images = generate_images(base, refiner, prompts, story_id)

            if not images:
                raise ValueError("No images generated")

            video_path = create_video(
                images, row["AudioPath"], row["SubtitlePath"],
                VIDEO_OUTPUT_DIR, story_id
            )

            df.at[idx, "VideoPath"] = video_path
            df.at[idx, "VideoStatus"] = "completed"
            print(f"✅ Done: {video_path}")
        except Exception as e:
            df.at[idx, "VideoStatus"] = f"error: {str(e)}"
            print(f"❌ Failed story {story_id}: {e}")

    df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    process_stories()
