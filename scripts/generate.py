import os
import sys
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Ensure repo root is in Python path
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from utils.text import read_script
from utils.video import create_video_from_frames

# -----------------------------
# Configuration
# -----------------------------
SCRIPT_PATH = "data/sample_scripts.txt"
OUTPUT_DIR = "outputs"
VIDEO_PATH = os.path.join(OUTPUT_DIR, "film_ai_output.mp4")
FRAME_SIZE = (640, 480)
FPS = 1  # 1 frame per second
BACKGROUND_COLOR = (30, 30, 30)  # dark gray
TEXT_COLOR = (255, 255, 255)  # white

# -----------------------------
# Ensure output directory exists
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Read script lines
# -----------------------------
lines = read_script(SCRIPT_PATH)

# -----------------------------
# Generate frames
# -----------------------------
frame_paths = []
font = ImageFont.load_default()  # Ensures text renders

for idx, line in enumerate(lines):
    img = Image.new("RGB", FRAME_SIZE, color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Center the text
    bbox = draw.textbbox((0, 0), line, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (FRAME_SIZE[0] - text_w) / 2
    text_y = (FRAME_SIZE[1] - text_h) / 2
    draw.text((text_x, text_y), line, fill=TEXT_COLOR, font=font)

    # Save frame
    frame_path = os.path.join(OUTPUT_DIR, f"frame_{idx:03d}.png")
    img.save(frame_path)
    frame_paths.append(frame_path)

# -----------------------------
# Create video from frames
# -----------------------------
create_video_from_frames(frame_paths, VIDEO_PATH)

print(f"Video saved to {VIDEO_PATH}")
