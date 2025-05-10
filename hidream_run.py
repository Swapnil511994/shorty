import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from diffusers import HiDreamImagePipeline
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


# Clear PyTorch memory BEFORE anything else
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print("🔑 Loading tokenizer and encoder (on CPU)...")
tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
text_encoder = LlamaForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    output_hidden_states=True,
    output_attentions=True,
    device_map="cpu"
)

print("🚀 Loading HiDream-I1-Fast pipeline...")
pipe = HiDreamImagePipeline.from_pretrained(
    "HiDream-ai/HiDream-I1-Fast",
    tokenizer_4=tokenizer,
    text_encoder_4=text_encoder,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True  # ✅ THIS is essential on low VRAM setups
).to("cuda")

prompt = "Crisis meeting at the UN headquarters with dynamic lighting, sharp photorealistic characters, intense expressions, 35mm lens cinematic detail"

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=0.0,  # For HiDream, guidance is typically 0
    num_inference_steps=16,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]

image.save("hidream_output.png")
print("✅ Image saved: hidream_output.png")
