#!/usr/bin/env python3
"""
Script generation utilities for Solaris AI Film Generator.
Contains functions for generating text-based frames.
"""

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
FONT_SIZE = 40


def generate_text_frames(lines, output_dir="outputs"):
    """
    Generate text-based frames and return paths.
    
    Args:
        lines (list): List of text lines to render as frames
        output_dir (str): Directory to save frames to
        
    Returns:
        list: List of paths to generated frame images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frame_paths = []
    
    # Try to load a better font with fallbacks
    font = None
    font_paths = [
        "arial.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "C:\\Windows\\Fonts\\arial.ttf"  # Windows
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, FONT_SIZE)
            print(f"Using {font_path} for text rendering")
            break
        except IOError:
            continue
    
    if font is None:
        font = ImageFont.load_default()
        print("Using default font for text rendering")
    
    for idx, line in enumerate(lines):
        img = Image.new("RGB", FRAME_SIZE, color=BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        
        # Calculate text size and center it
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (FRAME_SIZE[0] - text_w) // 2
        text_y = (FRAME_SIZE[1] - text_h) // 2
        
        draw.text((text_x, text_y), line, fill=TEXT_COLOR, font=font)
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
        img.save(frame_path)
        frame_paths.append(frame_path)
    
    print(f"{len(frame_paths)} frames generated in {output_dir}")
    return frame_paths


def main():
    """Main function for standalone execution."""
    # -----------------------------
    # Ensure output directory exists
    # -----------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -----------------------------
    # Read script lines
    # -----------------------------
    lines = read_script(SCRIPT_PATH)
    if not lines:
        raise ValueError(f"No lines found in {SCRIPT_PATH}")
    
    # -----------------------------
    # Generate frames
    # -----------------------------
    frame_paths = generate_text_frames(lines, OUTPUT_DIR)
    
    # -----------------------------
    # Create video from frames
    # -----------------------------
    create_video_from_frames(frame_paths, VIDEO_PATH, fps=FPS)
    print(f"Video saved to {VIDEO_PATH}")


if __name__ == "__main__":
    main()