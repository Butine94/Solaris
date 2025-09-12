import os
from typing import List, Dict
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
import numpy as np

class VideoAssembler:
    """Assembles images into cinematic video"""
    
    def __init__(self, fps: int = 24):
        self.fps = fps
    
    def create_video(self, shots: List[Dict], output_path: str, shot_duration: float = 3.0):
        """Create video from shot images"""
        clips = []
        
        for shot in shots:
            image_path = shot.get('image_path')
            if image_path and os.path.exists(image_path):
                # Create image clip with duration
                clip = ImageClip(image_path).set_duration(shot_duration)
                
                # Add cinematic effects
                clip = self._add_cinematic_effects(clip, shot['type'])
                clips.append(clip)
        
        if clips:
            # Concatenate all clips
            final_video = concatenate_videoclips(clips, method="compose")
            
            # Write video file
            final_video.write_videofile(
                output_path,
                fps=self.fps,
                codec='libx264',
                audio_codec='aac' if self._has_audio(shots) else None,
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            return output_path
        return None
    
    def _add_cinematic_effects(self, clip, shot_type: str):
        """Add cinematic effects based on shot type"""
        # Subtle zoom effect for establishing shots
        if shot_type == 'establishing':
            clip = clip.resize(lambda t: 1 + 0.02 * t)  # Slow zoom
        
        # Fade effects
        clip = clip.fadein(0.5).fadeout(0.5)
        
        return clip
    
    def _has_audio(self, shots: List[Dict]) -> bool:
        """Check if any shots have audio"""
        return any(shot.get('audio_path') for shot in shots)

def create_output_dirs(config: Dict):
    """Create necessary output directories"""
    os.makedirs(config['generation']['output_dir'], exist_ok=True)
    os.makedirs(config['generation']['frames_dir'], exist_ok=True)
    os.makedirs('data', exist_ok=True)
    print(f"✓ Created output directories: {config['generation']['output_dir']}, {config['generation']['frames_dir']}")

def get_image_paths(frames_dir: str, num_shots: int) -> List[str]:
    """Get expected image paths for shots"""
    return [os.path.join(frames_dir, f"shot_{i+1}.png") for i in range(num_shots)]

def validate_images(image_paths: List[str]) -> List[str]:
    """Validate that image files exist"""
    valid_paths = []
    for path in image_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"Warning: Image not found at {path}")
    return valid_paths