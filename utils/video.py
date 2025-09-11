import os
import numpy as np
from PIL import Image
from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, 
    VideoFileClip, concatenate_videoclips
)
from moviepy.audio.fx import audio_fadein, audio_fadeout
from typing import List, Tuple, Optional
import librosa
import soundfile as sf

class VideoAssembler:
    """Assembles images and audio into cinematic video"""
    
    def __init__(self, fps: int = 24, resolution: Tuple[int, int] = (512, 512)):
        self.fps = fps
        self.resolution = resolution
    
    def create_video_from_shots(
        self, 
        image_paths: List[str], 
        durations: List[float],
        audio_path: Optional[str] = None,
        output_path: str = "output.mp4",
        crossfade_duration: float = 0.5
    ) -> str:
        """Create video from image shots with optional audio"""
        
        if len(image_paths) != len(durations):
            raise ValueError("Number of images must match number of durations")
        
        # Create video clips from images
        clips = []
        for i, (img_path, duration) in enumerate(zip(image_paths, durations)):
            clip = self._create_image_clip(img_path, duration)
            
            # Add subtle zoom/pan effect for cinematic feel
            if i % 2 == 0:  # Alternate between zoom in and zoom out
                clip = self._add_zoom_effect(clip, zoom_factor=1.1)
            else:
                clip = self._add_pan_effect(clip, direction='right')
            
            clips.append(clip)
        
        # Add crossfade transitions
        if crossfade_duration > 0:
            clips = self._add_crossfades(clips, crossfade_duration)
        
        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Add audio if provided
        if audio_path and os.path.exists(audio_path):
            audio_clip = AudioFileClip(audio_path)
            
            # Match audio duration to video duration
            if audio_clip.duration != final_video.duration:
                if audio_clip.duration > final_video.duration:
                    audio_clip = audio_clip.subclip(0, final_video.duration)
                else:
                    # Loop audio if it's shorter than video
                    loops_needed = int(np.ceil(final_video.duration / audio_clip.duration))
                    audio_clips = [audio_clip] * loops_needed
                    audio_clip = concatenate_videoclips(audio_clips).subclip(0, final_video.duration)
            
            # Add fade in/out to audio
            audio_clip = audio_fadein(audio_clip, 0.5)
            audio_clip = audio_fadeout(audio_clip, 0.5)
            
            final_video = final_video.set_audio(audio_clip)
        
        # Export video
        final_video.write_videofile(
            output_path,
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        # Clean up
        final_video.close()
        if audio_path:
            audio_clip.close()
        
        return output_path
    
    def _create_image_clip(self, image_path: str, duration: float) -> ImageClip:
        """Create a video clip from an image"""
        # Load and resize image
        image = Image.open(image_path)
        image = image.resize(self.resolution, Image.Resampling.LANCZOS)
        
        # Create clip
        clip = ImageClip(np.array(image), duration=duration)
        clip = clip.set_fps(self.fps)
        
        return clip
    
    def _add_zoom_effect(self, clip: ImageClip, zoom_factor: float = 1.1) -> ImageClip:
        """Add subtle zoom effect to clip"""
        def make_frame(t):
            # Calculate zoom progress (0 to 1 over clip duration)
            progress = t / clip.duration
            current_zoom = 1.0 + (zoom_factor - 1.0) * progress
            
            # Get original frame
            frame = clip.get_frame(t)
            h, w = frame.shape[:2]
            
            # Calculate crop dimensions
            crop_h = int(h / current_zoom)
            crop_w = int(w / current_zoom)
            
            # Calculate crop position (center)
            y1 = (h - crop_h) // 2
            x1 = (w - crop_w) // 2
            y2 = y1 + crop_h
            x2 = x1 + crop_w
            
            # Crop and resize back to original size
            cropped = frame[y1:y2, x1:x2]
            resized = np.array(Image.fromarray(cropped).resize((w, h), Image.Resampling.LANCZOS))
            
            return resized
        
        return clip.fl(make_frame)
    
    def _add_pan_effect(self, clip: ImageClip, direction: str = 'right') -> ImageClip:
        """Add subtle pan effect to clip"""
        def make_frame(t):
            progress = t / clip.duration
            frame = clip.get_frame(t)
            h, w = frame.shape[:2]
            
            # Create slightly larger frame for panning
            pan_amount = int(w * 0.1)  # 10% pan
            
            if direction == 'right':
                x_offset = int(pan_amount * progress)
            else:  # left
                x_offset = int(pan_amount * (1 - progress))
            
            # Create larger canvas and place frame with offset
            canvas = np.zeros((h, w + pan_amount, 3), dtype=frame.dtype)
            canvas[:, x_offset:x_offset+w] = frame
            
            # Crop back to original size
            final_frame = canvas[:, :w]
            
            return final_frame
        
        return clip.fl(make_frame)
    
    def _add_crossfades(self, clips: List[ImageClip], fade_duration: float) -> List[ImageClip]:
        """Add crossfade transitions between clips"""
        if len(clips) <= 1:
            return clips
        
        faded_clips = [clips[0]]
        
        for i in range(1, len(clips)):
            # Add fade out to previous clip
            prev_clip = faded_clips[-1]
            if prev_clip.duration > fade_duration:
                faded_clips[-1] = prev_clip.fadeout(fade_duration)
            
            # Add fade in to current clip
            current_clip = clips[i]
            if current_clip.duration > fade_duration:
                current_clip = current_clip.fadein(fade_duration)
            
            faded_clips.append(current_clip)
        
        return faded_clips

class AudioProcessor:
    """Processes audio for video synchronization"""
    
    @staticmethod
    def split_audio_by_durations(audio_path: str, durations: List[float], output_dir: str) -> List[str]:
        """Split audio file into segments based on shot durations"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        output_paths = []
        current_time = 0.0
        
        for i, duration in enumerate(durations):
            start_sample = int(current_time * sr)
            end_sample = int((current_time + duration) * sr)
            
            # Extract segment
            segment = audio[start_sample:end_sample]
            
            # Save segment
            output_path = os.path.join(output_dir, f"segment_{i+1:02d}.wav")
            sf.write(output_path, segment, sr)
            output_paths.append(output_path)
            
            current_time += duration
        
        return output_paths
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """Get duration of audio file in seconds"""
        audio, sr = librosa.load(audio_path, sr=None)
        return len(audio) / sr
    
    @staticmethod
    def normalize_audio(audio_path: str, output_path: str = None) -> str:
        """Normalize audio levels"""
        if output_path is None:
            output_path = audio_path
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Normalize
        audio_normalized = librosa.util.normalize(audio)
        
        # Save
        sf.write(output_path, audio_normalized, sr)
        
        return output_path

def create_cinematic_video(
    image_paths: List[str],
    shot_durations: List[float],
    audio_path: Optional[str] = None,
    output_path: str = "cinematic_output.mp4",
    fps: int = 24,
    resolution: Tuple[int, int] = (512, 512),
    add_effects: bool = True
) -> str:
    """
    High-level function to create cinematic video from images and audio
    
    Args:
        image_paths: List of paths to generated images
        shot_durations: Duration for each shot in seconds
        audio_path: Optional path to narration audio file
        output_path: Output video file path
        fps: Frames per second
        resolution: Video resolution (width, height)
        add_effects: Whether to add cinematic effects (zoom/pan)
    
    Returns:
        Path to created video file
    """
    assembler = VideoAssembler(fps=fps, resolution=resolution)
    
    # Create video
    crossfade = 0.3 if add_effects else 0.0
    
    video_path = assembler.create_video_from_shots(
        image_paths=image_paths,
        durations=shot_durations,
        audio_path=audio_path,
        output_path=output_path,
        crossfade_duration=crossfade
    )
    
    return video_path

def ensure_directory_exists(path: str):
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)

def clean_temp_files(temp_dir: str):
    """Clean up temporary files"""
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")