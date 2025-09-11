#!/usr/bin/env python3
"""
Film AI - Main Application Entry Point
Generate cinematic videos from text scripts using AI diffusion models
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import logging

# Import our modules
from utils.text import ScriptProcessor, create_narration_text
from utils.video import create_cinematic_video, ensure_directory_exists
from models.diffusion import create_diffusion_generator
from scripts.generate import FilmGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('film_ai.log')
    ]
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data files if they don't exist"""
    
    # Create sample script
    sample_script = """A wide ocean reflects the stars like liquid metal.
A space station floats above, silent and still.
A tall black shape stands against the sky.
The only sound is the hum of machines.
One person stands small in the vast emptiness."""
    
    os.makedirs('data', exist_ok=True)
    
    script_path = 'data/sample_scripts.txt'
    if not os.path.exists(script_path):
        with open(script_path, 'w') as f:
            f.write(sample_script)
        logger.info(f"Created sample script: {script_path}")
    
    # Create config if it doesn't exist
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        sample_config = {
            'backend': 'auto',
            'diffusion': {
                'base_model': 'runwayml/stable-diffusion-v1-5',
                'guidance_scale': 7.5,
                'num_inference_steps': 30,
                'height': 512,
                'width': 512,
                'seed': 42,
                'dtype': 'fp16'
            },
            'generation': {
                'fps': 24,
                'output_dir': 'outputs',
                'frames_dir': 'outputs/frames',
                'output_video': 'outputs/sample_output.mp4',
                'num_scenes': 5
            },
            'data': {
                'scripts_file': 'data/sample_scripts.txt'
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)
        logger.info(f"Created sample config: {config_path}")

class FilmAI:
    """Main application class for Film AI"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.generator = None
        
        # Create sample data if needed
        create_sample_data()
        
        # Initialize generator
        self.generator = FilmGenerator(config_path)
        
    def generate_quick_film(self, text: str = None) -> str:
        """Generate a quick film from text"""
        logger.info("🎬 Starting quick film generation...")
        
        if text is None:
            text = """A wide ocean reflects the stars like liquid metal.
A space station floats above, silent and still.
A tall black shape stands against the sky.
The only sound is the hum of machines.
One person stands small in the vast emptiness."""
        
        return self.generator.generate_film(script_text=text)
    
    def generate_from_file(self, script_path: str) -> str:
        """Generate film from script file"""
        logger.info(f"🎬 Generating film from: {script_path}")
        return self.generator.generate_film(script_file=script_path)
    
    def interactive_mode(self):
        """Interactive mode for film generation"""
        print("\n🎬 Welcome to Film AI - Interactive Mode")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Generate film from default script")
            print("2. Generate film from custom text")
            print("3. Generate film from file")
            print("4. Generate variations for a shot")
            print("5. Regenerate specific shot")
            print("6. Show current config")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-6): ").strip()
            
            try:
                if choice == '0':
                    print("👋 Goodbye!")
                    break
                    
                elif choice == '1':
                    print("\n🎥 Generating film from default script...")
                    video_path = self.generate_quick_film()
                    print(f"✅ Film generated: {video_path}")
                    
                elif choice == '2':
                    print("\n📝 Enter your script (press Enter twice to finish):")
                    lines = []
                    while True:
                        line = input()
                        if line == '' and lines:
                            break
                        lines.append(line)
                    
                    script_text = '\n'.join(lines)
                    if script_text.strip():
                        video_path = self.generate_quick_film(script_text)
                        print(f"✅ Film generated: {video_path}")
                    else:
                        print("❌ No script provided")
                        
                elif choice == '3':
                    script_path = input("Enter script file path: ").strip()
                    if os.path.exists(script_path):
                        video_path = self.generate_from_file(script_path)
                        print(f"✅ Film generated: {video_path}")
                    else:
                        print("❌ File not found")
                        
                elif choice == '4':
                    try:
                        shot_id = int(input("Enter shot ID to generate variations for: "))
                        num_variations = int(input("Number of variations (default 3): ") or "3")
                        variations = self.generator.generate_variations(shot_id, num_variations)
                        print(f"✅ Generated {len(variations)} variations for shot {shot_id}")
                        for i, path in enumerate(variations, 1):
                            print(f"  Variation {i}: {path}")
                    except (ValueError, FileNotFoundError) as e:
                        print(f"❌ Error: {e}")
                        
                elif choice == '5':
                    try:
                        shot_id = int(input("Enter shot ID to regenerate: "))
                        new_prompt = input("Enter new prompt (optional): ").strip() or None
                        new_image = self.generator.regenerate_shot(shot_id, new_prompt)
                        print(f"✅ Regenerated shot {shot_id}: {new_image}")
                    except (ValueError, FileNotFoundError) as e:
                        print(f"❌ Error: {e}")
                        
                elif choice == '6':
                    print("\n📋 Current Configuration:")
                    print("-" * 30)
                    config = self.generator.config
                    print(f"Model: {config['diffusion']['base_model']}")
                    print(f"Resolution: {config['diffusion']['width']}x{config['diffusion']['height']}")
                    print(f"Inference Steps: {config['diffusion']['num_inference_steps']}")
                    print(f"Guidance Scale: {config['diffusion']['guidance_scale']}")
                    print(f"Num Scenes: {config['generation']['num_scenes']}")
                    print(f"Output Dir: {config['generation']['output_dir']}")
                    
                else:
                    print("❌ Invalid choice")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Interactive mode error: {e}")

def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(
        description='Film AI - Generate cinematic videos from text scripts',
        epilog='Examples:\n'
               '  python main.py                           # Interactive mode\n'
               '  python main.py --quick                   # Quick generation with default script\n'
               '  python main.py --file script.txt        # Generate from file\n'
               '  python main.py --text "Your script"     # Generate from text\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', '-c', default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick generation with default script')
    parser.add_argument('--file', '-f', help='Generate from script file')
    parser.add_argument('--text', '-t', help='Generate from text input')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Start interactive mode (default if no other options)')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize Film AI
        film_ai = FilmAI(args.config)
        
        # Override output if specified
        if args.output:
            film_ai.generator.config['generation']['output_video'] = args.output
        
        # Determine mode
        if args.quick:
            print("🚀 Quick film generation...")
            video_path = film_ai.generate_quick_film()
            print(f"\n🎉 Success! Your film is ready:")
            print(f"📽️  Video: {video_path}")
            
        elif args.file:
            print(f"📁 Generating film from file: {args.file}")
            video_path = film_ai.generate_from_file(args.file)
            print(f"\n🎉 Success! Your film is ready:")
            print(f"📽️  Video: {video_path}")
            
        elif args.text:
            print("✍️  Generating film from custom text...")
            video_path = film_ai.generate_quick_film(args.text)
            print(f"\n🎉 Success! Your film is ready:")
            print(f"📽️  Video: {video_path}")
            
        else:
            # Default to interactive mode
            film_ai.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

def show_banner():
    """Show application banner"""
    banner = """
    ╔═══════════════════════════════════════╗
    ║              🎬 FILM AI 🎬             ║
    ║                                       ║
    ║     Cinematic Video Generation        ║
    ║        from Text Scripts              ║
    ║                                       ║
    ║  Transform words into visual stories  ║
    ╚═══════════════════════════════════════╝
    """
    print(banner)

if __name__ == '__main__':
    show_banner()
    main()