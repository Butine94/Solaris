#!/usr/bin/env python3
"""Fast image generation script"""

import torch
import yaml
import os
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def generate_fast():
    """Generate images quickly"""
    print("🚀 Fast Generation Mode")
    
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        config['model'],
        dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # Optimize
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_attention_slicing()
    
    # Generate
    prompts = [
        "wide ocean reflecting stars, cinematic",
        "space station floating above, dramatic lighting", 
        "tall black silhouette, atmospheric"
    ]
    
    os.makedirs('outputs', exist_ok=True)
    start = time.time()
    
    for i, prompt in enumerate(prompts):
        print(f"⚡ Shot {i+1}/3...")
        img = pipe(
            prompt,
            num_inference_steps=config.get('steps', 8),
            height=config.get('size', [384, 512])[0],
            width=config.get('size', [384, 512])[1]
        ).images[0]
        
        img.save(f'outputs/shot_{i+1}.png')
    
    print(f"✅ Done in {time.time()-start:.1f}s")

if __name__ == "__main__":
    generate_fast()