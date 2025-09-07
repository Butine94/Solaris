import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text import read_script
from utils.video import create_video_from_frames
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Configuration
# -----------------------------
SCRIPT_PATH = "data/sample_scripts.txt"
OUTPUT_DIR = "outputs"
VIDEO_PATH = os.path.join(OUTPUT_DIR, "film_ai_output.mp4")
FRAME_SIZE = (640, 480)
FPS = 1  # frames per second
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
# Load font
# -----------------------------
try:
    font = ImageFont.truetype("arial.ttf", size=32)
except OSError:
    font = ImageFont.load_default()  # fallback if Arial is unavailable

# -----------------------------
# Generate frames
# -----------------------------
frame_paths = []

for idx, line in enumerate(lines):
    # Create new image for each frame
    img = Image.new("RGB", FRAME_SIZE, color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Center the text
    bbox = draw.textbbox((0, 0), line, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (FRAME_SIZE[0] - text_w) / 2
    text_y = (FRAME_SIZE[1] - text_h) / 2
    draw.text((text_x, text_y), line, fill=TEXT_COLOR, font=font)

    # Save frame image
    frame_path = os.path.join(OUTPUT_DIR, f"frame_{idx:03d}.png")
    img.save(frame_path)
    frame_paths.append(frame_path)

# -----------------------------
# Create video from frames
# -----------------------------
create_video_from_frames(frame_paths, VIDEO_PATH)

print(f"Video saved to {VIDEO_PATH}")



