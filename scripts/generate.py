from utils.text import read_script
from models.diffusion_model import DiffusionGenerator


def main():
    script_lines = read_script("data/sample_scripts.txt")

    generator = DiffusionGenerator()
    generator.generate_images(script_lines, output_dir="outputs")

if __name__ == "__main__":
    main()