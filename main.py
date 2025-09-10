#!/usr/bin/env python3
"""
Solaris - AI Film Generator
Main entry point for the application.

Usage:
    python -m solaris [mode] [options]
    
Modes:
    text    - Generate simple text-based storyboard frames
    ai      - Generate AI images using diffusion model  
    config  - Use configuration file settings
    
Examples:
    python -m solaris text
    python -m solaris ai
    python -m solaris config --config configs/default.yaml
"""

import argparse
import os
import sys
import yaml

# Add the project root to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.text import read_script
from utils.video import create_video_from_frames
from scripts.generate import generate_text_frames
from scripts.train import generate_ai_images


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return None


def run_text_mode(script_path="data/sample_scripts.txt", output_dir="outputs"):
    """Generate simple text-based frames."""
    print("Running in TEXT mode - generating text-based storyboard frames...")
    
    if not os.path.exists(script_path):
        print(f"Error: Script file not found: {script_path}")
        return False
    
    try:
        # Read script
        lines = read_script(script_path)
        print(f"Loaded {len(lines)} scenes from script")
        
        # Generate text frames
        frame_paths = generate_text_frames(lines, output_dir)
        
        # Create video
        video_path = os.path.join(output_dir, "text_storyboard.mp4")
        create_video_from_frames(frame_paths, video_path, fps=1)
        
        print(f"✓ Text-based storyboard video created: {video_path}")
        return True
        
    except Exception as e:
        print(f"Error in text mode: {e}")
        return False


def run_ai_mode(script_path="data/sample_scripts.txt", output_dir="outputs"):
    """Generate AI images using diffusion model."""
    print("Running in AI mode - generating AI-powered images...")
    
    if not os.path.exists(script_path):
        print(f"Error: Script file not found: {script_path}")
        return False
    
    try:
        # Check if diffusers is available
        try:
            from models.diffusion_model import DiffusionGenerator
        except ImportError as e:
            print(f"Error: Required packages not installed.")
            print("Please install: pip install diffusers transformers accelerate torch")
            return False
        
        # Read script
        lines = read_script(script_path)
        print(f"Loaded {len(lines)} prompts for AI generation")
        
        # Generate AI images
        success = generate_ai_images(lines, output_dir)
        
        if success:
            print("✓ AI-generated storyboard completed")
        else:
            print("✗ AI generation failed")
        
        return success
        
    except Exception as e:
        print(f"Error in AI mode: {e}")
        return False


def run_config_mode(config_path):
    """Run using configuration file."""
    print(f"Running with config file: {config_path}")
    
    config = load_config(config_path)
    if not config:
        return False
    
    try:
        # Extract settings from config
        script_path = config.get('data', {}).get('scripts_file', 'data/sample_scripts.txt')
        output_dir = config.get('generation', {}).get('output_dir', 'outputs')
        backend = config.get('backend', 'text')
        
        print(f"Script: {script_path}")
        print(f"Output: {output_dir}")
        print(f"Backend: {backend}")
        
        if backend == 'auto' or backend == 'diffusion':
            return run_ai_mode(script_path, output_dir)
        else:
            return run_text_mode(script_path, output_dir)
            
    except Exception as e:
        print(f"Error in config mode: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Solaris - AI Film Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m solaris text                    # Generate text-based frames
    python -m solaris ai                      # Generate AI images
    python -m solaris config                  # Use default config
    python -m solaris config --config custom.yaml
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['text', 'ai', 'config'],
        default='text',
        nargs='?',
        help='Generation mode (default: text)'
    )
    
    parser.add_argument(
        '--config',
        default='configs/default.yaml',
        help='Configuration file path (default: configs/default.yaml)'
    )
    
    parser.add_argument(
        '--script',
        default='data/sample_scripts.txt',
        help='Script file path (default: data/sample_scripts.txt)'
    )
    
    parser.add_argument(
        '--output',
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("SOLARIS - AI Film Generator")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run based on mode
    success = False
    
    if args.mode == 'text':
        success = run_text_mode(args.script, args.output)
    elif args.mode == 'ai':
        success = run_ai_mode(args.script, args.output)
    elif args.mode == 'config':
        success = run_config_mode(args.config)
    
    if success:
        print("\n✓ Generation completed successfully!")
        print(f"Check the '{args.output}' directory for results.")
    else:
        print("\n✗ Generation failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()