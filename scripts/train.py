import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text import read_script
from utils.video import create_video_from_frames
from models.diffusion_model import DiffusionGenerator

def main():
    script_file = "data/sample_scripts.txt"
    if not os.path.exists(script_file):
        raise FileNotFoundError(f"Script file not found: {script_file}")
    prompts = read_script(script_file)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating images from script...")
    generator = DiffusionGenerator()
    image_paths = generator.generate_images(prompts, output_dir=output_dir)

    video_path = os.path.join(output_dir, "generated_film.mp4")
    print("Creating video from frames...")
    create_video_from_frames(image_paths, output_path=video_path, fps=1)

    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    main()