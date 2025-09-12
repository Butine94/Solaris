#!/usr/bin/env python3
"""
Setup script for AI Film Generator
Creates necessary files and directories
"""

import os
import yaml

def create_directories():
    """Create necessary directory structure"""
    dirs = [
        'utils',
        'models', 
        'scripts',
        'data',
        'outputs',
        'outputs/frames',
        'config'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        
    # Create __init__.py files for Python modules
    init_files = ['utils/__init__.py', 'models/__init__.py', 'scripts/__init__.py']
    for init_file in init_files:
        if not os.path.exists(init_file):
            open(init_file, 'w').close()
    
    print("✓ Directory structure created")

def create_config():
    """Create default config.yaml"""
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
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✓ config.yaml created")

def create_sample_script():
    """Create sample script"""
    sample_script = """A wide ocean reflects the stars like liquid metal. A space station floats above, silent and still. A tall black shape stands against the sky. The only sound is the hum of machines. One person stands small in the vast emptiness."""
    
    with open('data/sample_scripts.txt', 'w') as f:
        f.write(sample_script)
    
    print("✓ Sample script created")

def main():
    print("🎬 Setting up AI Film Generator...")
    print("=" * 40)
    
    create_directories()
    create_config()
    create_sample_script()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Run generation: python main.py")
    print("3. Create video: python main.py --video")

if __name__ == "__main__":
    main()