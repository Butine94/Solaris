import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text import read_script
from utils.video import create_video_from_frames

def main():
    script_path = "data/sample_scripts.txt"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    script_lines = read_script(script_path)

    frame_paths = []
    for i, line in enumerate(script_lines):

        frame_path = os.path.join(output_dir, f"frame_{i:03}.png")
        with open(frame_path, "w") as f:
            f.write(line)
        frame_paths.append(frame_path)

    video_path = os.path.join(output_dir, "film_ai_output.mp4")
    create_video_from_frames(frame_paths, video_path)

if __name__ == "__main__":
    main()
