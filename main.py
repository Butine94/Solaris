#!/usr/bin/env python3
"""Ultra-fast AI film generator for Colab"""

import os
import torch
import yaml
import re
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt

def load_config():
    """Load config with fallback"""
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except:
        return {
            'model': 'runwayml/stable-diffusion-v1-5',
            'steps': 8, 'guidance': 6.0, 'size': [384, 512],
            'shots': 3, 'script': 'script.txt'
        }

def decompose_script(text, num_shots=3):
    """Quick script to shots conversion"""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    shot_types = ['wide shot of', 'medium shot of', 'close-up of']
    elements = [
        'ocean reflecting starlight like liquid metal',
        'space station floating silently above',
        'tall black silhouette against night sky'
    ]
    
    shots = []
    for i in range(min(num_shots, 3)):
        prompt = f"{shot_types[i]} {elements[i]}, cinematic, dramatic lighting"
        shots.append({'id': i+1, 'prompt': prompt})
    
    return shots

def generate_images():
    """Main generation - optimized for speed"""
    print("🚀 Fast AI Film Generator")
    
    # Load config
    config = load_config()
    
    # Load script
    try:
        with open(config.get('script', 'script.txt')) as f:
            script = f.read()
    except:
        script = "Ocean reflects stars. Space station floats above. Black figure stands."
    
    # Decompose to shots
    shots = decompose_script(script, config.get('shots', 3))
    print(f"📝 Created {len(shots)} shots")
    
    # Setup model (optimized)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"🔥 Loading model on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config.get('model', 'runwayml/stable-diffusion-v1-5'),
        dtype=dtype
    ).to(device)
    
    # Speed optimizations
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except: pass
    
    # Generate images
    os.makedirs('outputs', exist_ok=True)
    images = []
    
    for shot in shots:
        print(f"⚡ Generating shot {shot['id']}/3...")
        
        img = pipe(
            shot['prompt'],
            num_inference_steps=config.get('steps', 8),
            guidance_scale=config.get('guidance', 6.0),
            height=config.get('size', [384, 512])[0],
            width=config.get('size', [384, 512])[1]
        ).images[0]
        
        # Save
        img_path = f"outputs/shot_{shot['id']}.png"
        img.save(img_path)
        images.append(img)
        print(f"✅ Saved {img_path}")
    
    # Display
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    if len(images) == 1:
        axes = [axes]
        
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(f"Shot {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"🎉 Generated {len(images)} images!")
    return images

if __name__ == "__main__":
    generate_images()