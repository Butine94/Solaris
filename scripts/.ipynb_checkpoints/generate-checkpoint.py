#!/usr/bin/env python3
"""
Fast cinematic image generation from script
"""

import sys
import os
import yaml
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.text import ScriptDecomposer, load_script, save_shots_metadata
from utils.video import create_output_dirs, get_image_paths
from models.diffusion import CinematicDiffusionModel

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_cinematic_images(config_path: str = "config.yaml"):
    """Main generation pipeline"""
    print("🎬 Starting Cinematic AI Film Generation...")
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Create output directories
    create_output_dirs(config)
    
    # Load and decompose script
    print("📝 Loading and decomposing script...")
    script_text = load_script(config['data']['scripts_file'])
    
    decomposer = ScriptDecomposer()
    shots = decomposer.decompose_to_shots(script_text, config['generation']['num_scenes'])
    
    print(f"✓ Decomposed into {len(shots)} cinematic shots:")
    for shot in shots:
        print(f"  - {shot['description']}")
    
    # Save shots metadata
    metadata_path = os.path.join(config['generation']['output_dir'], "shots_metadata.json")
    save_shots_metadata(shots, metadata_path)
    
    # Initialize diffusion model
    print("🚀 Loading AI diffusion model...")
    model = CinematicDiffusionModel(config)
    
    # Generate images for all shots
    print("🎨 Generating cinematic images...")
    updated_shots = model.generate_shots(shots, config['generation']['frames_dir'])
    
    # Performance summary
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(updated_shots)
    
    print(f"\n🎉 Generation Complete!")
    print(f"📊 Performance:")
    print(f"   - Total time: {total_time:.2f} seconds")
    print(f"   - Average per image: {avg_time:.2f} seconds")
    print(f"   - Images generated: {len(updated_shots)}")
    print(f"   - Output directory: {config['generation']['frames_dir']}")
    
    # Cleanup
    model.cleanup()
    
    return updated_shots

if __name__ == "__main__":
    # Handle command line arguments
    config_file = sys.argv[1] if len(sys.argv) > 1 else "default.yaml"
    
    try:
        shots = generate_cinematic_images(config_file)
        print("\n✨ Ready for video assembly! Run video generation next.")
        
    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        sys.exit(1)