# import torch
# from diffusers import (
#     StableDiffusionXLPipeline,
#     StableDiffusionXLImg2ImgPipeline,
#     LCMScheduler
# )
# from peft import LoraConfig, inject_adapter_in_model
# from PIL import Image, ImageFilter
# import warnings

# # Suppress all non-critical warnings
# warnings.filterwarnings("ignore")

# def initialize_pipelines():
#     """Initialize optimized SDXL pipelines with fallbacks"""
#     # Hardware optimizations
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
    
#     print("🚀 Loading SDXL pipelines (this may take a few minutes)...")
    
#     try:
#         # Base pipeline with error handling
#         base = StableDiffusionXLPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-base-1.0",
#             torch_dtype=torch.float16,  # Using float16 for wider compatibility
#             variant="fp16",
#             use_safetensors=True,
#         ).to("cuda")
        
#         # Refiner pipeline
#         refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-refiner-1.0",
#             torch_dtype=torch.float16,
#             variant="fp16",
#         ).to("cuda")
        
#         # LCM-LoRA injection with proper error handling
#         print("⚡ Injecting LCM-LoRA adapter...")
#         lora_config = LoraConfig(
#             r=16,
#             lora_alpha=32,
#             target_modules=["to_k", "to_q", "to_v", "to_out.0"],
#             lora_dropout=0.0,
#             bias="none",
#         )
#         inject_adapter_in_model(lora_config, base.unet)
        
#         # Load LCM weights with explicit local_files_only check
#         try:
#             base.load_lora_weights(
#                 "latent-consistency/lcm-lora-sdxl",
#                 weight_name="pytorch_lora_weights.safetensors"
#             )
#         except Exception as e:
#             print(f"⚠️ Couldn't load LCM-LoRA: {str(e)}")
#             print("Using base SDXL without LCM acceleration")
        
#         base.scheduler = LCMScheduler.from_config(base.scheduler.config)
        
#         # Performance optimizations
#         try:
#             base.unet = torch.compile(base.unet)  # Will fail on Windows sometimes
#         except:
#             print("⚠️ torch.compile not available, continuing without it")
        
#         base.enable_vae_slicing()
#         base.enable_model_cpu_offload()
#         refiner.enable_vae_tiling()
        
#         return base, refiner
        
#     except Exception as e:
#         print(f"❌ Pipeline initialization failed: {str(e)}")
#         raise

# def enhance_image(image):
#     """Post-processing with error handling"""
#     try:
#         img = image.filter(ImageFilter.SHARPEN)
#         img = img.filter(ImageFilter.DETAIL)
#         return img
#     except:
#         return image  # Return original if processing fails

# def generate_image(base, refiner, prompt, output_file, steps=12):
#     """Robust generation workflow"""
#     print(f"🎨 Generating: '{prompt[:50]}...'")
    
#     try:
#         # First pass (base model)
#         latents = base(
#             prompt=prompt,
#             negative_prompt="blurry, low quality, bad anatomy, deformed",
#             num_inference_steps=steps,
#             guidance_scale=2.0,  # Lower than default for LCM
#             height=1024,
#             width=1024,
#             output_type="latent",
#         ).images
        
#         # Second pass (refiner)
#         image = refiner(
#             prompt=prompt,
#             image=latents,
#             strength=0.3,
#             num_inference_steps=int(steps*0.5),
#         ).images[0]
        
#         # Post-processing
#         image = enhance_image(image)
#         image.save(output_file)
#         print(f"✅ Successfully saved to {output_file}")
#         return image
        
#     except torch.cuda.OutOfMemoryError:
#         print("⚠️ Out of VRAM! Trying smaller resolution...")
#         return generate_image(base, refiner, prompt, output_file, steps=8)  # Retry with fewer steps
        
#     except Exception as e:
#         print(f"❌ Generation failed: {str(e)}")
#         return None

# if __name__ == "__main__":
#     try:
#         # Initialize with error handling
#         base_pipe, refiner_pipe = initialize_pipelines()
        
#         # Example generation with fallback prompts
#         generate_image(
#             base_pipe,
#             refiner_pipe,
#             prompt="A futuristic cyberpunk cityscape at night, neon lights reflecting on wet streets, 8k cinematic",
#             output_file="cyberpunk_city.png",
#             steps=14
#         )
        
#     except Exception as e:
#         print(f"❌ Critical failure: {str(e)}")
#         print("\n🔧 Troubleshooting Guide:")
#         print("1. VRAM Check: Run 'nvidia-smi' to verify GPU memory")
#         print("2. Driver Update: https://www.nvidia.com/Download/index.aspx")
#         print("3. Package Versions:")
#         print("   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118")
#         print("   pip install diffusers==0.25.0 transformers==4.35.0 peft==0.7.1")
#         print("4. Reduce resolution to 768x768 if crashes persist")


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
            prompt="A cyberpunk samurai standing on a neon-lit Tokyo skyscraper, 8k cinematic lighting",
            save_path="batch_output_10.png"
        )
        
        # Batch processing example
        prompts = [
            "Futuristic cityscape with flying cars, neon lights, rain reflections",
            "Portrait of a steampunk inventor with intricate goggles",
            "Mystical forest with glowing mushrooms and fairies"
        ]
        
        for i, prompt in enumerate(prompts):
            generator.generate(
                prompt=prompt,
                save_path=f"batch_output_{i}.png"
            )
            
    except Exception as e:
        print(f"💥 Critical error: {str(e)}")