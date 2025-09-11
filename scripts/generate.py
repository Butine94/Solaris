def generate_text_frames(lines, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []
    
    for idx, line in enumerate(lines):
        # Use the SAME values that worked in your test
        img = Image.new("RGB", (640, 480), (50, 50, 50))  # Hardcoded gray
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()  # Simple default font
        
        # Simple fixed position (not calculated)
        draw.text((50, 200), line, fill=(255, 255, 255), font=font)  # Hardcoded white
        
        frame_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
        img.save(frame_path)
        frame_paths.append(frame_path)
        
    return frame_paths