import os
from moviepy.editor import ImageSequenceClip


def create_video_from_frames(frame_paths, output_path, fps=1, duration_per_frame=None):
    """
    Create video from frame images with better encoding settings.
    
    Args:
        frame_paths (list): List of paths to frame images
        output_path (str): Output video file path
        fps (float): Frames per second (default: 1)
        duration_per_frame (float): How long each frame shows (overrides fps)
    """
    if not frame_paths:
        raise ValueError("No frame paths provided")
    
    # Verify all frame files exist
    missing_frames = [path for path in frame_paths if not os.path.exists(path)]
    if missing_frames:
        raise FileNotFoundError(f"Missing frame files: {missing_frames}")
    
    print(f"Creating video from {len(frame_paths)} frames...")
    print(f"Output: {output_path}")
    
    try:
        # Create video clip from images
        if duration_per_frame:
            # Each image shows for specified duration
            durations = [duration_per_frame] * len(frame_paths)
            clip = ImageSequenceClip(frame_paths, durations=durations)
        else:
            # Use FPS
            clip = ImageSequenceClip(frame_paths, fps=fps)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write video with BETTER ENCODING SETTINGS
        clip.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',           # Explicit codec
            audio=False,              # No audio
            verbose=True,             # Show progress
            logger='bar',             # Progress bar
            temp_audiofile=None,      # No temp audio
            remove_temp=True,         # Clean up
            # CRITICAL ENCODING PARAMETERS:
            ffmpeg_params=[
                '-pix_fmt', 'yuv420p',    # Proper pixel format
                '-crf', '18',             # High quality (lower = better)
                '-preset', 'medium',      # Encoding speed vs quality
                '-movflags', '+faststart' # Web optimization
            ]
        )
        
        # Verify output file was created and has reasonable size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ Video created: {output_path}")
            print(f"✓ File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            
            # Warn if file seems too small
            if file_size < 10000:  # Less than 10KB
                print("⚠️  WARNING: Video file seems very small - there might be an encoding issue")
            
            return True
        else:
            print("✗ Video file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Video creation failed: {e}")
        return False
    finally:
        # Clean up clip
        if 'clip' in locals():
            clip.close()


def create_video_with_longer_frames(frame_paths, output_path, seconds_per_frame=2):
    """
    Create video where each frame is shown for multiple seconds.
    
    Args:
        frame_paths (list): List of paths to frame images
        output_path (str): Output video file path  
        seconds_per_frame (int): How many seconds to show each frame
    """
    return create_video_from_frames(
        frame_paths, 
        output_path, 
        fps=1,  # 1 fps for smooth playback
        duration_per_frame=seconds_per_frame
    )


def create_debug_video(frame_paths, output_path="debug_video_test.mp4"):
    """
    Create a test video with verbose output for debugging.
    """
    print("=" * 50)
    print("DEBUG VIDEO CREATION")
    print("=" * 50)
    
    print(f"Frame paths: {len(frame_paths)}")
    for i, path in enumerate(frame_paths):
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  {i+1}: {path} - {'✓' if exists else '✗'} ({size} bytes)")
    
    # Try creating video with different settings
    settings_to_try = [
        {"fps": 1, "name": "Standard (1 FPS)"},
        {"duration_per_frame": 3, "name": "3 seconds per frame"},
    ]
    
    for i, settings in enumerate(settings_to_try):
        test_path = output_path.replace('.mp4', f'_test_{i+1}.mp4')
        print(f"\nTesting: {settings['name']}")
        
        try:
            success = create_video_from_frames(frame_paths, test_path, **{k:v for k,v in settings.items() if k != 'name'})
            if success:
                file_size = os.path.getsize(test_path)
                print(f"✓ Success: {test_path} ({file_size:,} bytes)")
            else:
                print(f"✗ Failed: {test_path}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("=" * 50)