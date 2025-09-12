"""Fast text processing for cinematic shots"""

import re

def script_to_shots(text, num_shots=3):
    """Quick conversion of script to visual shots"""
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    # Pre-defined shot types for speed
    shot_types = ['establishing', 'medium', 'close']
    
    # Quick visual extraction
    visuals = extract_visuals(sentences)
    
    # Build shots
    shots = []
    for i in range(min(num_shots, len(visuals))):
        shot = {
            'id': i + 1,
            'type': shot_types[i % len(shot_types)],
            'element': visuals[i],
            'prompt': f"{shot_types[i % len(shot_types)]} shot of {visuals[i]}, cinematic, dramatic lighting"
        }
        shots.append(shot)
    
    return shots

def extract_visuals(sentences):
    """Fast visual element extraction"""
    # Key mappings for speed
    keywords = {
        'ocean': 'vast ocean reflecting starlight like liquid metal',
        'space': 'space station floating silently above', 
        'figure': 'tall black silhouette against night sky',
        'machine': 'glowing machinery in darkness',
        'person': 'lone figure in cosmic emptiness'
    }
    
    visuals = []
    text = ' '.join(sentences).lower()
    
    # Extract based on keywords
    for keyword, visual in keywords.items():
        if keyword in text and len(visuals) < 5:
            visuals.append(visual)
    
    # Fill remaining with defaults
    defaults = [
        'starlit horizon with cosmic beauty',
        'mysterious shadows and light',
        'futuristic technological structures'
    ]
    
    while len(visuals) < 3:
        visuals.extend(defaults)
    
    return visuals[:3]

def load_script(filepath):
    """Load script with fallback"""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "A wide ocean reflects stars like liquid metal. A space station floats above. A tall black shape stands against the sky."

# Quick test
if __name__ == "__main__":
    sample = "Ocean reflects stars. Space station floats. Black figure stands."
    shots = script_to_shots(sample)
    for shot in shots:
        print(f"Shot {shot['id']}: {shot['prompt']}")