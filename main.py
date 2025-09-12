#!/usr/bin/env python3
"""
AI Cinematic Film Generator - Main Entry Point
Generates 5 high-quality cinematic images from script quickly
"""

import os
import sys
import yaml
import time
import argparse
from pathlib import Path

from utils.text import ScriptDecomposer, load_script, save_shots_metadata
from utils.video import VideoAssembler, create_output_dirs
from models.diffusion import CinematicDiffusionModel

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file {config_path} not found!")
        sys.exit(1)

def create_sample_script():
    """Create sample script file if it doesn't exist"""
    script_path = "data/sample_scripts.txt"
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(script_path):
        sample_script = """A wide ocean reflects the stars like liquid metal. A space station floats above, silent and still. A tall black shape stands against the sky. The only sound is the hum of machines. One person stands small in the vast emptiness."""
        
        with open(script_path, 'w') as f:
            f.write(sample_script)
        print(f"✓ Created sample script at {script_path}")

def generate_images_pipeline(config: dict) -> list:
    """Fast image generation pipeline"""
    print("🎬 AI CINEMATIC FILM GENERATOR")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create output directories
    create_output_dirs(config)
    
    # Load and decompose script
    print("📝 Processing script...")
    script_text = load_script(config['data']['scripts_file'])
    
    decomposer = ScriptDecomposer()
    shots = decomposer.decompose_to_shots(script_text, config['generation']['num_scenes'])
    
    print(f"✓ Script decomposed into {len(shots)} cinematic shots:")
    for i, shot in enumerate(shots, 1):
        print(f"   {i}. {shot['type'].title()}: {shot['element']}")
    
    # Initialize and run diffusion model
    print("\n🚀 Loading AI diffusion model...")
    model = CinematicDiffusionModel(config)
    
    print("🎨 Generating high-quality images...")
    updated_shots = model.generate_shots(shots, config['generation']['frames_dir'])
    
    # Save metadata
    metadata_path = os.path.join(config['generation']['output_dir'], "shots_metadata.json")
    save_shots_metadata(updated_shots, metadata_path)
    
    # Performance summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 IMAGE GENERATION COMPLETE!")
    print(f"📊 Performance Summary:")
    print(f"   ⚡ Total time: {total_time:.2f} seconds")
    print(f"   🖼️  Images generated: {len(updated_shots)}")
    print(f"   ⏱️  Average per image: {total_time/len(updated_shots):.2f}s")
    print(f"   📁 Output directory: {config['generation']['frames_dir']}")
    
    # Cleanup
    model.cleanup()
    
    return updated_shots

def create_video_pipeline(config: dict, shots: list):
    """Create video from generated images"""
    print("\n🎥 Creating cinematic video...")
    
    assembler = VideoAssembler(fps=config['generation']['fps'])
    
    video_path = assembler.create_video(
        shots, 
        config['generation']['output_video'],
        config['generation']['shot_duration']
    )
    
    if video_path:
        print(f"✓ Video created: {video_path}")
    else:
        print("❌ Video creation failed")
    
    return video_path

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Generate cinematic film from script using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Generate images only
  python main.py --video           # Generate images and video  
  python main.py --config custom.yaml  # Use custom config
        """
    )
    
    parser.add_argument(
        '--config', 
        default='default.yaml',
        help='Path to configuration file (default: default.yaml, looks in config/ folder)'
    )
    
    parser.add_argument(
        '--video',
        action='store_true',
        help='Also create video from generated images'
    )
    
    parser.add_argument(
        '--images-only',
        action='store_true',
        help='Generate images only (default behavior)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create sample script if needed
        create_sample_script()
        
        # Generate images
        shots = generate_images_pipeline(config)
        
        # Optionally create video
        if args.video:
            video_path = create_video_pipeline(config, shots)
            print(f"\n🎬 Complete pipeline finished!")
            print(f"   🖼️  Images: {config['generation']['frames_dir']}")
            if video_path:
                print(f"   🎥 Video: {video_path}")
        else:
            print(f"\n✨ Image generation complete! Use --video flag to create video.")
            
        print(f"\n💡 Next steps:")
        print(f"   - Check images in: {config['generation']['frames_dir']}")
        print(f"   - Edit config.yaml to adjust settings")
        print(f"   - Run with --video to create final film")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Generation interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("💡 Try:")
        print("   - Check your GPU memory (need ~6GB)")
        print("   - Verify config.yaml exists")
        print("   - Install requirements: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()