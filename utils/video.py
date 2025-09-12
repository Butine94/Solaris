"""Fast video assembly utilities"""

import os
from PIL import Image

def create_dirs(config):
    """Create output directories"""
    dirs = [
        config.get('output_dir', 'outputs'),
        config.get('frames_dir', 'outputs/frames'),
        'data'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ Created directories: {', '.join(dirs)}")

def make_video_fast(image_paths, output_path, duration=3.0):
    """Quick video creation (optional - requires moviepy)"""
    try:
        from moviepy.editor import ImageClip, concatenate_videoclips
        
        clips = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                clip = ImageClip(img_path).set_duration(duration)
                clips.append(clip)
        
        if clips:
            video = concatenate_videoclips(clips, method="compose")
            video.write_videofile(output_path, fps=24, verbose=False, logger=None)
            print(f"🎬 Video saved: {output_path}")
            return output_path
        
    except ImportError:
        print("⚠️ MoviePy not installed. Skipping video creation.")
        print("   Install with: pip install moviepy")
    except Exception as e:
        print(f"⚠️ Video creation failed: {e}")
    
    return None

def display_images(image_paths):
    """Quick image display"""
    try:
        import matplotlib.pyplot as plt
        
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_paths:
            print("❌ No images found to display")
            return
        
        num_imgs = len(valid_paths)
        fig, axes = plt.subplots(1, num_imgs, figsize=(4*num_imgs, 4))
        
        if num_imgs == 1:
            axes = [axes]
        
        for i, img_path in enumerate(valid_paths):
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Shot {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        print(f"🖼️ Displayed {num_imgs} images")
        
    except ImportError:
        print("⚠️ Matplotlib not available for display")

def get_image_paths(frames_dir, num_shots=3):
    """Get expected image paths"""
    return [os.path.join(frames_dir, f"shot_{i+1}.png") for i in range(num_shots)]

def validate_images(image_paths):
    """Check which images exist"""
    valid = []
    for path in image_paths:
        if os.path.exists(path):
            valid.append(path)
        else:
            print(f"⚠️ Missing: {path}")
    
    print(f"✅ Found {len(valid)}/{len(image_paths)} images")
    return valid

# Quick test
if __name__ == "__main__":
    config = {'output_dir': 'test_outputs', 'frames_dir': 'test_outputs/frames'}
    create_dirs(config)
    
    # Test paths
    paths = get_image_paths('test_outputs/frames', 3)
    print("Expected paths:", paths)