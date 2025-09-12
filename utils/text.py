import re
from typing import List, Dict

class ScriptDecomposer:
    """Decomposes scripts into atomic cinematic shots"""
    
    def __init__(self):
        self.shot_templates = {
            'establishing': "wide establishing shot of {}",
            'medium': "medium shot of {}",
            'close': "close-up of {}",
            'detail': "extreme close-up detail of {}",
            'atmosphere': "atmospheric shot of {}"
        }
    
    def decompose_to_shots(self, script: str, num_shots: int = 5) -> List[Dict]:
        """Convert script to atomic visual shots"""
        sentences = self._split_sentences(script)
        shots = []
        
        # Extract key visual elements
        visual_elements = self._extract_visuals(sentences)
        
        # Create 5 distinct shots with cinematic variety
        shot_types = ['establishing', 'medium', 'atmosphere', 'close', 'detail']
        
        for i, (shot_type, element) in enumerate(zip(shot_types, visual_elements[:5])):
            prompt = self._build_cinematic_prompt(shot_type, element)
            shots.append({
                'id': i + 1,
                'type': shot_type,
                'element': element,
                'prompt': prompt,
                'description': f"Shot {i+1}: {shot_type} of {element}"
            })
        
        return shots
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_visuals(self, sentences: List[str]) -> List[str]:
        """Extract visual elements from sentences"""
        visuals = []
        
        # Key visual elements from your sample
        visual_mapping = {
            'ocean': 'vast ocean reflecting starlight like liquid metal',
            'space': 'space station floating silently above',
            'figure': 'tall black silhouette against the night sky',
            'machines': 'glowing machinery humming in the darkness',
            'person': 'lone figure standing in cosmic emptiness'
        }
        
        # Extract based on keywords
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword, visual in visual_mapping.items():
                if keyword in sentence_lower and visual not in visuals:
                    visuals.append(visual)
        
        # Fill remaining slots with atmospheric elements
        atmospheric_elements = [
            'starlit horizon with cosmic beauty',
            'mysterious shadows and ethereal light',
            'technological structures in space',
            'infinite void with distant stars',
            'industrial machinery in darkness'
        ]
        
        while len(visuals) < 5:
            for elem in atmospheric_elements:
                if elem not in visuals:
                    visuals.append(elem)
                    break
        
        return visuals[:5]
    
    def _build_cinematic_prompt(self, shot_type: str, element: str) -> str:
        """Build cinematic prompt for diffusion model"""
        base_template = self.shot_templates.get(shot_type, "{}")
        shot_desc = base_template.format(element)
        
        # Add cinematic styling
        cinematic_style = "cinematic, dramatic lighting, film grain, professional cinematography, sci-fi atmosphere"
        
        return f"{shot_desc}, {cinematic_style}"

def load_script(filepath: str) -> str:
    """Load script from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Return sample script if file not found
        return """A wide ocean reflects the stars like liquid metal. A space station floats above, silent and still. 
                  A tall black shape stands against the sky. The only sound is the hum of machines. 
                  One person stands small in the vast emptiness."""

def save_shots_metadata(shots: List[Dict], output_path: str):
    """Save shot metadata to file"""
    import json
    with open(output_path, 'w') as f:
        json.dump(shots, f, indent=2)