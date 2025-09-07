import os
import sys
from pathlib import Path
import textwrap

# Ensure utils can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.text import read_script
from utils.video import create_video_from_frames
from PIL import Image, ImageDraw, ImageFont


def main():
    # Paths
    script_path = "data/sample_scripts.txt"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Read the script
    script_lines = read_script(script_path)
    print(f"Read {len(script_lines)} script lines.")

    # Generate image frames from script lines
    frame_paths = []
    for i, line in enumerate(script_lines):
        frame_path = os.path.join(output_dir, f"frame_{i:03}.png")

        # Create a blank image (1280x720 black background)
        img = Image.new('RGB', (1280, 720), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Use a visible font
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except OSError:
            font = ImageFont.load_default()

        # Wrap text so it fits nicely on the image
        wrapped_text = textwrap.fill(line, width=60)

        # Draw text
        draw.text((50, 50), wrapped_text, font=font, fill=(255, 255, 255))

        # Save frame
        img.save(frame_path)
        frame_paths.append(frame_path)
        print(f"Created frame: {frame_path}")

    # Assemble frames into video
    video_path = os.path.join(output_dir, "film_ai_output.mp4")
    create_video_from_frames(frame_paths, video_path)
    print("Video generation complete!")


if __name__ == "__main__":
    main()
