#!/usr/bin/env python3
"""
Fix config file issues - create clean default.yaml
"""

import os
import yaml

# Remove any conflicting config files
files_to_remove = ['config.yaml', 'config/default.yaml']
for f in files_to_remove:
    if os.path.exists(f):
        os.remove(f)
        print(f"✓ Removed {f}")

# Create clean default.yaml
config = {
    'backend': 'auto',
    'diffusion': {
        'base_model': 'runwayml/stable-diffusion-v1-5',
        'guidance_scale': 7.5,
        'num_inference_steps': 20,
        'height': 512,
        'width': 768,
        'seed': 42,
        'dtype': 'fp16'
    },
    'generation': {
        'fps': 24,
        'output_dir': 'outputs',
        'frames_dir': 'outputs/frames',
        'output_video': 'outputs/cinematic_film.mp4',
        'num_scenes': 5,
        'shot_duration': 3.0
    },
    'style': {
        'base_prompt': 'cinematic, dramatic lighting, film grain, professional cinematography',
        'negative_prompt': 'blurry, low quality, distorted, amateur'
    },
    'data': {
        'scripts_file': 'data/sample_scripts.txt'
    }
}

# Write clean YAML file
with open('default.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print("✓ Created clean default.yaml")

# Create sample script if missing
os.makedirs('data', exist_ok=True)
if not os.path.exists('data/sample_scripts.txt'):
    with open('data/sample_scripts.txt', 'w') as f:
        f.write("A wide ocean reflects the stars like liquid metal. A space station floats above, silent and still. A tall black shape stands against the sky. The only sound is the hum of machines. One person stands small in the vast emptiness.")
    print("✓ Created sample script")

print("\n🎉 Config fixed! Now run: python main.py")