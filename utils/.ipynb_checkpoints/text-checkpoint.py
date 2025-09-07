import os

def load_script_lines(script_path):

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script file not found: {script_path}")

    with open(script_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def print_script_summary(lines):
    
    print(f"Total scenes: {len(lines)}")
    for i, line in enumerate(lines[:5], start=1):
        print(f"Scene {i}: {line}")