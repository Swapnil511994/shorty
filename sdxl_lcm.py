import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline,
    LCMScheduler
)
from peft import LoraConfig, inject_adapter_in_model
import warnings
from PIL import Image, ImageFilter
import numpy as np

# Disable all warnings
warnings.filterwarnings("ignore")

# Constants
MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
LCM_LORA = "latent-consistency/lcm-lora-sdxl"

class SDXLTurboGenerator:
    def __init__(self):
        self.device = "cuda"
        self._init_pipelines()
        
    def _init_pipelines(self):
        """Initialize with error handling and optimizations"""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("🚀 Loading SDXL Turbo Engine...")
        
        # Base pipeline
        self.base = StableDiffusionXLPipeline.from_pretrained(
            MODEL_BASE,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)
        
        # Refiner pipeline
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODEL_REFINER,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)
        
        # LCM-LoRA injection
        self._inject_lora()
        
        # Optimizations
        self._optimize_pipelines()
        
    def _inject_lora(self):
        """Safe LoRA injection with fallback"""
        try:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout=0.0,
                bias="none",
            )
            inject_adapter_in_model(lora_config, self.base.unet)
            self.base.load_lora_weights(LCM_LORA)
            self.base.scheduler = LCMScheduler.from_config(self.base.scheduler.config)
            print("⚡ LCM-LoRA successfully injected!")
        except Exception as e:
            print(f"⚠️ LoRA injection failed: {str(e)}")
            print("Running in standard mode (slower but stable)")
            
    def _optimize_pipelines(self):
        """Apply performance optimizations"""
        try:
            self.base.unet = torch.compile(self.base.unet)
        except:
            print("⚠️ torch.compile not available")
            
        self.base.enable_vae_slicing()
        self.base.enable_model_cpu_offload()
        self.refiner.enable_vae_tiling()
        
    def enhance(self, image):
        """Advanced post-processing"""
        img = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        img = Image.fromarray(np.array(img))  # Convert to array and back for better sharpness
        return img
        
    def generate(
        self,
        prompt,
        negative_prompt="blurry, low quality, bad anatomy",
        steps=8,
        high_res=True,
        apply_refiner=True,
        save_path="output.png"
    ):
        """
        Generate ultra-quality images with smart fallbacks
        
        Args:
            prompt: Text prompt
            steps: 4-20 (fewer = faster, more = better quality)
            high_res: 1024x1024 if True, else 768x768
            apply_refiner: Use refiner for extra detail
        """
        try:
            print(f"\n✨ Generating: '{prompt[:60]}...'")
            
            # Base generation
            latents = self.base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=2.0,
                height=1024 if high_res else 768,
                width=1024 if high_res else 768,
                output_type="latent" if apply_refiner else "pil",
            ).images
            
            # Refinement if enabled
            if apply_refiner:
                image = self.refiner(
                    prompt=prompt,
                    image=latents,
                    strength=0.3,
                    num_inference_steps=max(4, int(steps * 0.5)),
                ).images[0]
            else:
                image = latents[0]
            
            # Post-processing
            image = self.enhance(image)
            image.save(save_path)
            print(f"✅ Saved masterpiece to {save_path}")
            return image
            
        except torch.cuda.OutOfMemoryError:
            print("⚠️ VRAM overload! Retrying with reduced settings...")
            return self.generate(
                prompt,
                steps=max(4, steps//2),
                high_res=False,
                apply_refiner=False
            )

# Example Usage
if __name__ == "__main__":
    try:
        generator = SDXLTurboGenerator()
        
        # Single generation
        generator.generate(
            prompt="""Tense meeting between Indian and Pakistani diplomats, cyberpunk style:
            - **Characters**: Indian diplomat in smart turban with AR glasses, Pakistani diplomat with robotic arm
            - **Setting**: High-tech UN chamber with holographic maps and glowing negotiation table
            - **Style**: Neon cyberpunk by Simon Stålenhag and Jakub Rozalski
            - **Lighting**: Dark with neon accents (blue vs. green lighting factions)
            - **Details**: 8K UHD, intricate gadgetry, volumetric fog, cybernetic implants
            - **Mood**: High tension with visible stress lines on faces""",
            save_path="batch_output_10.png"
        )
        
        # Batch processing example
        # prompts = [
        #     "Futuristic cityscape with flying cars, neon lights, rain reflections",
        #     "Portrait of a steampunk inventor with intricate goggles",
        #     "Mystical forest with glowing mushrooms and fairies"
        # ]
        
        # for i, prompt in enumerate(prompts):
        #     generator.generate(
        #         prompt=prompt,
        #         save_path=f"batch_output_{i}.png"
        #     )
            
    except Exception as e:
        print(f"💥 Critical error: {str(e)}")