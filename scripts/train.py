import os
import sys

# Add the project root to Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from utils.text import read_script
from utils.video import create_video_from_frames


def generate_ai_images(prompts, output_dir="outputs"):
    """
    Generate AI images and return success status.
    
    Args:
        prompts (list): List of text prompts for image generation
        output_dir (str): Directory to save generated images
        
    Returns:
        bool: True if generation successful, False otherwise
    """
    try:
        # Import here to avoid dependency issues if diffusers not installed
        from models.diffusion_model import DiffusionGenerator
        
        print(f"Generating {len(prompts)} AI images...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the generator
        generator = DiffusionGenerator()
        
        # Generate images
        image_paths = generator.generate_images(prompts, output_dir=output_dir)
        
        if not image_paths:
            print("No images were generated")
            return False
        
        print(f"Generated {len(image_paths)} images")
        
        # Create video from generated images
        video_path = os.path.join(output_dir, "ai_generated_film.mp4")
        create_video_from_frames(image_paths, video_path, fps=1)
        
        print(f"AI-generated video saved to {video_path}")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required packages: pip install diffusers transformers accelerate torch")
        return False
    except Exception as e:
        print(f"AI generation error: {e}")
        return False


def main():
    """Main function for standalone execution."""
    script_file = "data/sample_scripts.txt"
    output_dir = "outputs"
    
    # Check if script file exists
    if not os.path.exists(script_file):
        print(f"Error: Script file not found: {script_file}")
        print("Please ensure the script file exists at the specified path.")
        return False
    
    try:
        # Read prompts from script
        prompts = read_script(script_file)
        if not prompts:
            print(f"No prompts found in {script_file}")
            return False
        
        print(f"Loaded {len(prompts)} prompts from script")
        
        # Generate AI images
        success = generate_ai_images(prompts, output_dir)
        
        if success:
            print("✓ AI generation completed successfully!")
        else:
            print("✗ AI generation failed!")
            
        return success
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)