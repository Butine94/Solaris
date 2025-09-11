#!/usr/bin/env python3
"""
Film AI Generation Script
Generates cinematic videos from text scripts using AI diffusion models
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add parent directory to path to import our modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from utils.text import ScriptProcessor, create_narration_text
    from utils.video import create_cinematic_video, ensure_directory_exists
    from models.diffusion import create_diffusion_generator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from the project root directory using: python -m scripts.generate")
    print("Or run: python main.py for the main interface")
    sys.exit(1)

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FilmGenerator:
    """Main class for generating cinematic films from scripts"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.script_processor = ScriptProcessor()
        self.diffusion_generator = None
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def generate_film(self, script_text: str = None, script_file: str = None) -> str:
        """
        Generate complete film from script
        
        Args:
            script_text: Raw script text (optional)
            script_file: Path to script file (optional, overrides script_text)
            
        Returns:
            Path to generated video file
        """
        try:
            # Step 1: Load and process script
            logger.info("Step 1: Processing script...")
            if script_file:
                script_text = self.script_processor.load_script(script_file)
            elif script_text is None:
                script_file = self.config['data']['scripts_file']
                script_text = self.script_processor.load_script(script_file)
            
            logger.info(f"Script loaded: {len(script_text)} characters")
            
            # Step 2: Decompose into shots
            logger.info("Step 2: Decomposing script into shots...")
            num_scenes = self.config['generation']['num_scenes']
            shots = self.script_processor.decompose_script(script_text, num_scenes)
            
            logger.info(f"Generated {len(shots)} shots:")
            for i, shot in enumerate(shots, 1):
                logger.info(f"  Shot {i}: {shot.description[:50]}...")
            
            # Export shots for inspection
            shots_yaml_path = os.path.join(self.config['generation']['output_dir'], 'shots.yaml')
            ensure_directory_exists(os.path.dirname(shots_yaml_path))
            self.script_processor.export_shots_to_yaml(shots, shots_yaml_path)
            logger.info(f"Shots exported to: {shots_yaml_path}")
            
            # Step 3: Generate images using diffusion model
            logger.info("Step 3: Generating images with AI diffusion...")
            self.diffusion_generator = create_diffusion_generator(config_dict=self.config['diffusion'])
            
            prompts = [shot.prompt for shot in shots]
            frames_dir = self.config['generation']['frames_dir']
            ensure_directory_exists(frames_dir)
            
            image_paths = self.diffusion_generator.generate_shot_images(
                prompts=prompts,
                output_dir=frames_dir,
                maintain_style_consistency=True
            )
            
            # Create preview grid
            grid_path = os.path.join(self.config['generation']['output_dir'], 'preview_grid.png')
            self.diffusion_generator.create_image_grid(image_paths, grid_path)
            logger.info(f"Preview grid saved: {grid_path}")
            
            # Step 4: Generate narration (placeholder - would integrate with TTS)
            logger.info("Step 4: Preparing narration...")
            narration_text = create_narration_text(shots)
            narration_path = os.path.join(self.config['generation']['output_dir'], 'narration.txt')
            
            with open(narration_path, 'w') as f:
                f.write(narration_text)
            logger.info(f"Narration saved: {narration_path}")
            logger.info(f"Narration text: {narration_text}")
            
            # Step 5: Assemble video
            logger.info("Step 5: Assembling final video...")
            durations = [shot.duration for shot in shots]
            output_video_path = self.config['generation']['output_video']
            ensure_directory_exists(os.path.dirname(output_video_path))
            
            # For now, create video without audio (TTS integration would go here)
            final_video_path = create_cinematic_video(
                image_paths=image_paths,
                shot_durations=durations,
                audio_path=None,  # TODO: Add TTS-generated audio
                output_path=output_video_path,
                fps=self.config['generation']['fps'],
                resolution=(self.config['diffusion']['width'], self.config['diffusion']['height']),
                add_effects=True
            )
            
            logger.info(f"✅ Film generation complete!")
            logger.info(f"📽️  Final video: {final_video_path}")
            logger.info(f"🖼️  Frames directory: {frames_dir}")
            logger.info(f"📝 Shots config: {shots_yaml_path}")
            logger.info(f"🎭 Preview grid: {grid_path}")
            
            return final_video_path
            
        except Exception as e:
            logger.error(f"Error during film generation: {e}")
            raise
        finally:
            # Cleanup
            if self.diffusion_generator:
                self.diffusion_generator.cleanup()
    
    def generate_variations(self, shot_id: int, num_variations: int = 3) -> list:
        """Generate variations for a specific shot"""
        logger.info(f"Generating {num_variations} variations for shot {shot_id}...")
        
        # Load shots
        shots_yaml_path = os.path.join(self.config['generation']['output_dir'], 'shots.yaml')
        if not os.path.exists(shots_yaml_path):
            raise FileNotFoundError("No shots.yaml found. Run generate_film first.")
        
        shots = self.script_processor.load_shots_from_yaml(shots_yaml_path)
        
        if shot_id < 1 or shot_id > len(shots):
            raise ValueError(f"Shot ID must be between 1 and {len(shots)}")
        
        shot = shots[shot_id - 1]
        
        # Generate variations
        if not self.diffusion_generator:
            self.diffusion_generator = create_diffusion_generator(config_dict=self.config['diffusion'])
        
        variations_dir = os.path.join(self.config['generation']['output_dir'], f'shot_{shot_id}_variations')
        variation_paths = self.diffusion_generator.batch_generate_variations(
            prompt=shot.prompt,
            num_variations=num_variations,
            output_dir=variations_dir
        )
        
        logger.info(f"Generated {len(variation_paths)} variations in: {variations_dir}")
        return variation_paths
    
    def regenerate_shot(self, shot_id: int, new_prompt: str = None) -> str:
        """Regenerate a specific shot with optional new prompt"""
        logger.info(f"Regenerating shot {shot_id}...")
        
        # Load shots
        shots_yaml_path = os.path.join(self.config['generation']['output_dir'], 'shots.yaml')
        shots = self.script_processor.load_shots_from_yaml(shots_yaml_path)
        
        if shot_id < 1 or shot_id > len(shots):
            raise ValueError(f"Shot ID must be between 1 and {len(shots)}")
        
        shot = shots[shot_id - 1]
        
        # Use new prompt if provided
        if new_prompt:
            shot.prompt = new_prompt
        
        # Generate new image
        if not self.diffusion_generator:
            self.diffusion_generator = create_diffusion_generator(config_dict=self.config['diffusion'])
        
        frames_dir = self.config['generation']['frames_dir']
        output_path = os.path.join(frames_dir, f"shot_{shot_id:02d}_regenerated.png")
        
        image_path = self.diffusion_generator._generate_single_image(
            prompt=self.diffusion_generator._enhance_cinematic_prompt(shot.prompt),
            negative_prompt=self.diffusion_generator._get_negative_prompt(),
            output_path=output_path,
            seed=self.diffusion_generator.base_seed + shot_id
        )
        
        logger.info(f"Shot {shot_id} regenerated: {image_path}")
        return image_path

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Generate cinematic films from text scripts')
    parser.add_argument('--config', '-c', default='config.yaml', help='Config file path')
    parser.add_argument('--script', '-s', help='Script file path (overrides config)')
    parser.add_argument('--text', '-t', help='Direct script text input')
    parser.add_argument('--variations', type=int, help='Generate variations for shot ID')
    parser.add_argument('--regenerate', type=int, help='Regenerate specific shot ID')
    parser.add_argument('--new-prompt', help='New prompt for regeneration')
    parser.add_argument('--output', '-o', help='Output video path (overrides config)')
    
    args = parser.parse_args()
    
    # Initialize generator
    film_generator = FilmGenerator(args.config)
    
    # Override output path if specified
    if args.output:
        film_generator.config['generation']['output_video'] = args.output
    
    try:
        if args.variations:
            # Generate variations for specific shot
            variations = film_generator.generate_variations(args.variations)
            print(f"Generated {len(variations)} variations for shot {args.variations}")
            
        elif args.regenerate:
            # Regenerate specific shot
            new_image = film_generator.regenerate_shot(args.regenerate, args.new_prompt)
            print(f"Regenerated shot {args.regenerate}: {new_image}")
            
        else:
            # Generate full film
            video_path = film_generator.generate_film(
                script_text=args.text,
                script_file=args.script
            )
            print(f"\n🎬 Film generation complete!")
            print(f"📽️  Video: {video_path}")
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()