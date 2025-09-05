import os
from moviepy.editor import ImageSequenceClip

def frames_to_video(frames_dir, output_path, fps=24):

    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frames = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".png", ".jpg"))
    ])

    if not frames:
        raise ValueError("No image frames found in the folder.")

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264")
    print(f"Video saved to {output_path}")