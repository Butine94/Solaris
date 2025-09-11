import re
import yaml
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Shot:
    """Represents a single cinematic shot"""
    id: int
    description: str
    prompt: str
    duration: float = 3.0
    scene_type: str = "medium"  # wide, medium, close, extreme_close
    lighting: str = "natural"   # natural, dramatic, soft, neon, etc.

class ScriptProcessor:
    """Processes raw script text into structured cinematic shots"""
    
    def __init__(self):
        self.shot_keywords = {
            'wide': ['ocean', 'vast', 'space', 'sky', 'horizon', 'landscape'],
            'medium': ['person', 'figure', 'character', 'standing'],
            'close': ['face', 'eyes', 'hand', 'detail'],
            'extreme_close': ['reflection', 'tear', 'breath', 'whisper']
        }
        
        self.lighting_keywords = {
            'dramatic': ['dark', 'shadow', 'silhouette', 'black'],
            'soft': ['gentle', 'warm', 'peaceful'],
            'neon': ['metal', 'machine', 'station', 'glow'],
            'natural': ['stars', 'sky', 'light']
        }
    
    def load_script(self, file_path: str) -> str:
        """Load script from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Script file not found: {file_path}")
    
    def decompose_script(self, script_text: str, num_scenes: int = 5) -> List[Shot]:
        """Decompose script into atomic shots"""
        sentences = self._split_into_sentences(script_text)
        shots = []
        
        for i, sentence in enumerate(sentences[:num_scenes]):
            shot = self._create_shot_from_sentence(i + 1, sentence)
            shots.append(shot)
        
        return shots
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling multiple formats"""
        # Clean and split text
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _create_shot_from_sentence(self, shot_id: int, sentence: str) -> Shot:
        """Create a Shot object from a sentence"""
        scene_type = self._detect_scene_type(sentence)
        lighting = self._detect_lighting(sentence)
        prompt = self._enhance_prompt(sentence, scene_type, lighting)
        
        return Shot(
            id=shot_id,
            description=sentence,
            prompt=prompt,
            duration=self._calculate_duration(sentence),
            scene_type=scene_type,
            lighting=lighting
        )
    
    def _detect_scene_type(self, sentence: str) -> str:
        """Detect shot type based on content"""
        sentence_lower = sentence.lower()
        
        for shot_type, keywords in self.shot_keywords.items():
            if any(keyword in sentence_lower for keyword in keywords):
                return shot_type
        
        return "medium"  # default
    
    def _detect_lighting(self, sentence: str) -> str:
        """Detect lighting mood from sentence"""
        sentence_lower = sentence.lower()
        
        for lighting_type, keywords in self.lighting_keywords.items():
            if any(keyword in sentence_lower for keyword in keywords):
                return lighting_type
        
        return "natural"  # default
    
    def _enhance_prompt(self, sentence: str, scene_type: str, lighting: str) -> str:
        """Enhance sentence into a detailed diffusion prompt"""
        base_prompt = sentence
        
        # Add cinematic style
        style_additions = [
            "cinematic composition",
            "professional cinematography",
            "film grain",
            "35mm lens"
        ]
        
        # Add scene-specific enhancements
        scene_enhancements = {
            'wide': "establishing shot, wide angle, epic scale",
            'medium': "medium shot, balanced composition",
            'close': "close-up, detailed, intimate",
            'extreme_close': "extreme close-up, macro, highly detailed"
        }
        
        # Add lighting enhancements
        lighting_enhancements = {
            'dramatic': "dramatic lighting, high contrast, deep shadows",
            'soft': "soft lighting, gentle shadows, warm tones",
            'neon': "neon lighting, cyberpunk aesthetic, metallic reflections",
            'natural': "natural lighting, balanced exposure"
        }
        
        enhanced_prompt = f"{base_prompt}, {scene_enhancements[scene_type]}, {lighting_enhancements[lighting]}, {', '.join(style_additions)}"
        
        return enhanced_prompt
    
    def _calculate_duration(self, sentence: str) -> float:
        """Calculate shot duration based on sentence complexity"""
        word_count = len(sentence.split())
        base_duration = 2.5
        
        # Longer sentences get more time
        duration = base_duration + (word_count * 0.1)
        
        # Cap between 2-6 seconds
        return max(2.0, min(6.0, duration))
    
    def export_shots_to_yaml(self, shots: List[Shot], output_path: str):
        """Export shots to YAML for inspection/editing"""
        shots_data = []
        for shot in shots:
            shots_data.append({
                'id': shot.id,
                'description': shot.description,
                'prompt': shot.prompt,
                'duration': shot.duration,
                'scene_type': shot.scene_type,
                'lighting': shot.lighting
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump({'shots': shots_data}, f, default_flow_style=False, indent=2)
    
    def load_shots_from_yaml(self, yaml_path: str) -> List[Shot]:
        """Load shots from YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        shots = []
        for shot_data in data['shots']:
            shot = Shot(
                id=shot_data['id'],
                description=shot_data['description'],
                prompt=shot_data['prompt'],
                duration=shot_data.get('duration', 3.0),
                scene_type=shot_data.get('scene_type', 'medium'),
                lighting=shot_data.get('lighting', 'natural')
            )
            shots.append(shot)
        
        return shots

def create_narration_text(shots: List[Shot]) -> str:
    """Create narration text from shots for TTS"""
    narration_parts = []
    
    for shot in shots:
        # Convert visual description to narration
        narration = shot.description.replace("reflects", "reflecting")
        narration = narration.replace("stands", "standing")
        narration = narration.replace("floats", "floating")
        
        narration_parts.append(narration)
    
    return " ".join(narration_parts)

def estimate_narration_duration(text: str, words_per_minute: int = 150) -> float:
    """Estimate narration duration in seconds"""
    word_count = len(text.split())
    duration_minutes = word_count / words_per_minute
    return duration_minutes * 60