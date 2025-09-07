import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def create_video_from_frames(frame_paths, output_path, fps=24):

    if not frame_paths:
        raise ValueError("No frames provided to create video.")
    
    clip = ImageSequenceClip(frame_paths, fps=fps)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clip.write_videofile(output_path, codec="libx264")
    print(f"Video saved to {output_path}")