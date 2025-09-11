#!/usr/bin/env python3
"""
Debug script to diagnose black image issues in Solaris.
This will help identify where the problem occurs in the image generation pipeline.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Configuration
FRAME_SIZE = (640, 480)
BACKGROUND_COLOR = (30, 30, 30)  # dark gray
TEXT_COLOR = (255, 255, 255)  # white
FONT_SIZE = 40


def test_basic_image_creation():
    """Test 1: Can we create a basic colored image?"""
    print("=" * 50)
    print("TEST 1: Basic Image Creation")
    print("=" * 50)
    
    try:
        # Create a simple red image
        img = Image.new("RGB", FRAME_SIZE, color=(255, 0, 0))  # Red
        img.save("debug_test_red.png")
        print("✓ Red image created successfully")
        
        # Create image with our background color
        img2 = Image.new("RGB", FRAME_SIZE, color=BACKGROUND_COLOR)
        img2.save("debug_test_background.png")
        print("✓ Background color image created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Basic image creation failed: {e}")
        return False


def test_text_rendering():
    """Test 2: Can we render text on images?"""
    print("\n" + "=" * 50)
    print("TEST 2: Text Rendering")
    print("=" * 50)
    
    try:
        # Test with default font
        img = Image.new("RGB", FRAME_SIZE, color=BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        
        test_text = "Hello World!"
        font = ImageFont.load_default()
        
        # Draw text at a fixed position first
        draw.text((50, 50), test_text, fill=TEXT_COLOR, font=font)
        img.save("debug_test_simple_text.png")
        print("✓ Simple text rendering successful")
        
        # Test text centering
        bbox = draw.textbbox((0, 0), test_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (FRAME_SIZE[0] - text_w) // 2
        text_y = (FRAME_SIZE[1] - text_h) // 2
        
        print(f"Text dimensions: {text_w} x {text_h}")
        print(f"Text position: ({text_x}, {text_y})")
        
        img2 = Image.new("RGB", FRAME_SIZE, color=BACKGROUND_COLOR)
        draw2 = ImageDraw.Draw(img2)
        draw2.text((text_x, text_y), test_text, fill=TEXT_COLOR, font=font)
        img2.save("debug_test_centered_text.png")
        print("✓ Centered text rendering successful")
        
        return True
    except Exception as e:
        print(f"✗ Text rendering failed: {e}")
        return False


def test_font_loading():
    """Test 3: Font loading capabilities"""
    print("\n" + "=" * 50)
    print("TEST 3: Font Loading")
    print("=" * 50)
    
    font_paths = [
        "arial.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "C:\\Windows\\Fonts\\arial.ttf"  # Windows
    ]
    
    fonts_tested = []
    working_font = None
    
    # Test default font
    try:
        default_font = ImageFont.load_default()
        fonts_tested.append(("Default", True, "Built-in default font"))
        working_font = default_font
        print("✓ Default font loaded")
    except Exception as e:
        fonts_tested.append(("Default", False, str(e)))
        print(f"✗ Default font failed: {e}")
    
    # Test TrueType fonts
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, FONT_SIZE)
            fonts_tested.append((font_path, True, "Successfully loaded"))
            working_font = font
            print(f"✓ Font loaded: {font_path}")
        except Exception as e:
            fonts_tested.append((font_path, False, str(e)))
            print(f"✗ Font failed: {font_path} - {e}")
    
    # Test with working font
    if working_font:
        try:
            img = Image.new("RGB", FRAME_SIZE, color=BACKGROUND_COLOR)
            draw = ImageDraw.Draw(img)
            draw.text((50, 50), "Font Test!", fill=TEXT_COLOR, font=working_font)
            img.save("debug_test_font.png")
            print("✓ Font rendering test successful")
        except Exception as e:
            print(f"✗ Font rendering test failed: {e}")
    
    return fonts_tested


def test_script_reading():
    """Test 4: Script file reading"""
    print("\n" + "=" * 50)
    print("TEST 4: Script Reading")
    print("=" * 50)
    
    try:
        from utils.text import read_script
        
        # Try to read the sample script
        script_path = "data/sample_scripts.txt"
        if os.path.exists(script_path):
            lines = read_script(script_path)
            print(f"✓ Script file found: {script_path}")
            print(f"✓ Lines read: {len(lines)}")
            print("First 3 lines:")
            for i, line in enumerate(lines[:3]):
                print(f"  {i+1}: '{line}'")
            return lines
        else:
            print(f"✗ Script file not found: {script_path}")
            # Create a test script
            test_lines = ["Scene 1: A dark room", "Scene 2: A bright day", "Scene 3: The end"]
            return test_lines
    except Exception as e:
        print(f"✗ Script reading failed: {e}")
        return ["Test Scene 1", "Test Scene 2", "Test Scene 3"]


def test_full_pipeline(lines):
    """Test 5: Full image generation pipeline"""
    print("\n" + "=" * 50)
    print("TEST 5: Full Pipeline")
    print("=" * 50)
    
    try:
        output_dir = "debug_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load font
        font = ImageFont.load_default()
        
        frame_paths = []
        for idx, line in enumerate(lines[:3]):  # Only test first 3
            print(f"Generating frame {idx+1}: '{line}'")
            
            img = Image.new("RGB", FRAME_SIZE, color=BACKGROUND_COLOR)
            draw = ImageDraw.Draw(img)
            
            # Center the text
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = (FRAME_SIZE[0] - text_w) // 2
            text_y = (FRAME_SIZE[1] - text_h) // 2
            
            draw.text((text_x, text_y), line, fill=TEXT_COLOR, font=font)
            
            frame_path = os.path.join(output_dir, f"debug_frame_{idx:03d}.png")
            img.save(frame_path)
            frame_paths.append(frame_path)
            
            print(f"  ✓ Frame saved: {frame_path}")
        
        print(f"✓ Generated {len(frame_paths)} frames")
        return frame_paths
        
    except Exception as e:
        print(f"✗ Full pipeline failed: {e}")
        return []


def analyze_generated_images(frame_paths):
    """Test 6: Analyze the generated images"""
    print("\n" + "=" * 50)
    print("TEST 6: Image Analysis")
    print("=" * 50)
    
    for frame_path in frame_paths:
        try:
            img = Image.open(frame_path)
            img_array = np.array(img)
            
            print(f"Analyzing: {frame_path}")
            print(f"  Image size: {img.size}")
            print(f"  Image mode: {img.mode}")
            print(f"  Array shape: {img_array.shape}")
            print(f"  Min pixel value: {img_array.min()}")
            print(f"  Max pixel value: {img_array.max()}")
            print(f"  Mean pixel value: {img_array.mean():.2f}")
            
            # Check if image is all black
            if img_array.max() == 0:
                print("  ⚠️  WARNING: Image appears to be completely black!")
            elif img_array.min() == img_array.max():
                print(f"  ⚠️  WARNING: Image has uniform color: {img_array.min()}")
            else:
                print("  ✓ Image has varying pixel values")
            
            # Display image if in notebook environment
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 6))
                plt.imshow(img_array)
                plt.title(f"Generated Frame: {os.path.basename(frame_path)}")
                plt.axis('off')
                plt.show()
                print("  ✓ Image displayed above")
            except:
                print("  ℹ️  Image display not available (not in notebook environment)")
                
        except Exception as e:
            print(f"  ✗ Analysis failed: {e}")


def test_video_creation(frame_paths):
    """Test 7: Video creation from frames"""
    print("\n" + "=" * 50)
    print("TEST 7: Video Creation")
    print("=" * 50)
    
    if not frame_paths:
        print("✗ No frames to create video from")
        return False
    
    try:
        from utils.video import create_video_from_frames
        
        video_path = "debug_output/debug_video.mp4"
        create_video_from_frames(frame_paths, video_path, fps=1)
        
        if os.path.exists(video_path):
            print(f"✓ Video created: {video_path}")
            print(f"  File size: {os.path.getsize(video_path)} bytes")
            return True
        else:
            print("✗ Video file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Video creation failed: {e}")
        return False


def main():
    """Run all diagnostic tests"""
    print("SOLARIS IMAGE GENERATION DIAGNOSTICS")
    print("=" * 50)
    
    # Run tests
    test_basic_image_creation()
    test_text_rendering()
    fonts_info = test_font_loading()
    lines = test_script_reading()
    frame_paths = test_full_pipeline(lines)
    analyze_generated_images(frame_paths)
    test_video_creation(frame_paths)
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 50)
    print("\nGenerated files for inspection:")
    print("- debug_test_red.png (should be red)")
    print("- debug_test_background.png (should be dark gray)")
    print("- debug_test_simple_text.png (text at fixed position)")
    print("- debug_test_centered_text.png (centered text)")
    print("- debug_test_font.png (font test)")
    print("- debug_output/ folder (pipeline test results)")
    
    print("\nFont Summary:")
    for font_name, success, message in fonts_info:
        status = "✓" if success else "✗"
        print(f"  {status} {font_name}: {message}")


if __name__ == "__main__":
    main()