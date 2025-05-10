import os
import json
import torch
import pandas as pd
from PIL import Image, ImageFilter
from datetime import datetime
from diffusers import (
    HiDreamImagePipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    LCMScheduler
)
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from peft import LoraConfig, inject_adapter_in_model

# === Config ===
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
LCM_LORA = "latent-consistency/lcm-lora-sdxl"

with open(STYLES_CONFIG, 'r') as f:
    STYLE_MAP = json.load(f)


def load_llama_encoder():
    print("🔑 Loading Meta-Llama-3.1-8B-Instruct...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    encoder = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16
    )
    return tokenizer, encoder


def init_pipeline(model_base, style):
    is_hidream = model_base.startswith("HiDream-ai/")
    controlnet_required = style in STYLE_MAP and STYLE_MAP[style].get("use_controlnet", False)

    try:
        if is_hidream:
            tokenizer_4, text_encoder_4 = load_llama_encoder()
            pipe = HiDreamImagePipeline.from_pretrained(
                model_base,
                tokenizer_4=tokenizer_4,
                text_encoder_4=text_encoder_4,
                torch_dtype=torch.bfloat16
            ).to("cuda")
        else:
            variant = "fp16"
            if controlnet_required:
                controlnet = ControlNetModel.from_pretrained(
                    f"diffusers/controlnet-sdxl-{CONTROLNET_TYPE}", torch_dtype=torch.float16
                )
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    model_base, controlnet=controlnet,
                    torch_dtype=torch.float16,
                    variant=variant
                ).to("cuda")
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_base,
                    torch_dtype=torch.float16,
                    variant=variant
                ).to("cuda")
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            try:
                lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["to_k", "to_q", "to_v", "to_out.0"], lora_dropout=0.0, bias="none")
                inject_adapter_in_model(lora_config, pipe.unet)
                pipe.load_lora_weights(LCM_LORA)
            except Exception as e:
                print(f"⚠️ Skipping LCM-LoRA: {e}")

        # Optional: Style-specific LoRA
        lora_info = STYLE_MAP.get(style, {}).get("lora")
        if lora_info:
            try:
                pipe.load_lora_weights(lora_info["repo"], weight_name=lora_info["filename"], adapter_name=style)
                pipe.set_adapters([style], adapter_weights=[lora_info.get("strength", 0.5)])
            except Exception as e:
                print(f"⚠️ Style LoRA failed: {e}")

        return pipe

    except Exception as e:
        print(f"❌ Pipeline load failed for {model_base}: {e}")
        return None


def enhance(image):
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return Image.fromarray(torch.tensor(image).cpu().numpy())


def generate_images(pipe, prompts, story_id, is_hidream):
    images = []
    for i, prompt in enumerate(prompts):
        try:
            kwargs = dict(
                height=1024,
                width=1024,
                generator=torch.Generator("cuda").manual_seed(0)
            )
            if is_hidream:
                kwargs.update(dict(
                    guidance_scale=5.0 if "Full" in pipe.config._name_or_path else 0.0,
                    num_inference_steps=50 if "Full" in pipe.config._name_or_path else 28
                ))
                result = pipe(prompt, **kwargs)
            else:
                result = pipe(prompt=prompt, negative_prompt="blurry, malformed", num_inference_steps=15, guidance_scale=6.0, **kwargs)
            image = enhance(result.images[0])
            fname = f"story_{story_id}_scene_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = os.path.join(IMAGE_OUTPUT_DIR, fname)
            image.save(path)
            images.append(path)
        except Exception as e:
            print(f"❌ Generation error: {e}")
    return images


def process_stories():
    df = pd.read_csv(CSV_PATH)
    for idx, row in df.iterrows():
        if str(row.get("VideoStatus", "")).lower() == "completed":
            continue
        try:
            story_id = row["ID"]
            style = row.get("Style", "realism").lower()
            prompts = [row["Prompt1"], row["Prompt2"], row["Prompt3"]]

            pipe = None
            for model_base in MODEL_BASES:
                pipe = init_pipeline(model_base, style)
                if pipe:
                    break

            if not pipe:
                raise RuntimeError("No pipeline could be initialized")

            is_hidream = isinstance(pipe, HiDreamImagePipeline)
            images = generate_images(pipe, prompts, story_id, is_hidream)

            if not images:
                raise Exception("No images generated")

            df.at[idx, "VideoStatus"] = "completed"
            df.at[idx, "ImagePaths"] = "|".join(images)
            print(f"✅ Completed for ID {story_id}")

        except Exception as e:
            df.at[idx, "VideoStatus"] = f"error: {str(e)}"
            print(f"❌ Failed story {row['ID']}: {str(e)}")

    df.to_csv(CSV_PATH, index=False)


if __name__ == "__main__":
    process_stories()
