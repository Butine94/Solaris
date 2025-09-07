import os
import sys

# -----------------------------
# Ensure repo root is in Python path
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

def read_script(script_path):
    """
    Reads a text file and returns non-empty lines as a list.
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script file not found: {script_path}")

    with open(script_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def print_script_summary(lines):
    """
    Prints a brief summary of the script.
    """
    print(f"Total scenes: {len(lines)}")
    for i, line in enumerate(lines[:5], start=1):
        print(f"Scene {i}: {line}")
