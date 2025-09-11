import torch
import os
from typing import List, Optional, Dict, Any
from PIL import Image
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)
from diffusers.utils import make_image_grid
import random

class DiffusionImageGenerator:
    """
    Handles image generation using various diffusion models
    with focus on cinematic consistency and quality
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._get_device()
        self.pipeline = None
        self.model_loaded = False
        
        # Style consistency settings
        self.base_seed = config.get('seed', 42)
        self.style_seed = None
        
    def _get_device(self) -> str:
        """Determine best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps" 
        else:
            return "cpu"
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load the specified diffusion model"""
        if model_name is None:
            model_name = self.config.get('base_model', 'runwayml/stable-diffusion-v1-5')
        
        print(f"Loading model: {model_name} on {self.device}")
        
        try:
            # Determine model type and load accordingly
            if 'xl' in model_name.lower():
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.config.get('dtype') == 'fp16' else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.config.get('dtype') == 'fp16' else None
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.config.get('dtype') == 'fp16' else torch.float32,
                    use_safetensors=True
                )
            
            # Set up scheduler for better quality
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
            
            # Enable VAE slicing for memory efficiency
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            
            self.model_loaded = True
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_shot_images(
        self, 
        prompts: List[str], 
        output_dir: str,
        maintain_style_consistency: bool = True
    ) -> List[str]:
        """
        Generate images for a sequence of shots with style consistency
        
        Args:
            prompts: List of enhanced prompts for each shot
            output_dir: Directory to save generated images
            maintain_style_consistency: Whether to maintain visual consistency
            
        Returns:
            List of paths to generated images
        """
        if not self.model_loaded:
            self.load_model()
        
        os.makedirs(output_dir, exist_ok=True)
        generated_paths = []
        
        # Set style seed for consistency across shots
        if maintain_style_consistency and self.style_seed is None:
            self.style_seed = self.base_seed
        
        print(f"Generating {len(prompts)} shots...")
        
        for i, prompt in enumerate(prompts):
            print(f"Generating shot {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Enhance prompt for cinematic quality
            enhanced_prompt = self._enhance_cinematic_prompt(prompt)
            negative_prompt = self._get_negative_prompt()
            
            # Generate image
            image_path = self._generate_single_image(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                output_path=os.path.join(output_dir, f"shot_{i+1:02d}.png"),
                seed=self.style_seed + i if maintain_style_consistency else None
            )
            
            generated_paths.append(image_path)
        
        print("All shots generated successfully!")
        return generated_paths
    
    def _generate_single_image(
        self, 
        prompt: str, 
        negative_prompt: str,
        output_path: str,
        seed: Optional[int] = None
    ) -> str:
        """Generate a single image with specified parameters"""
        
        # Set random seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        try:
            # Generate image
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=self.config.get('height', 512),
                    width=self.config.get('width', 512),
                    num_inference_steps=self.config.get('num_inference_steps', 30),
                    guidance_scale=self.config.get('guidance_scale', 7.5),
                    generator=generator,
                    num_images_per_prompt=1
                )
            
            # Save image
            image = result.images[0]
            image.save(output_path, quality=95, optimize=True)
            
            print(f"  → Saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating image: {e}")
            raise
    
    def _enhance_cinematic_prompt(self, base_prompt: str) -> str:
        """Add cinematic enhancement to base prompt"""
        cinematic_terms = [
            "cinematic lighting",
            "film grain", 
            "depth of field",
            "professional photography",
            "high quality",
            "detailed",
            "atmospheric",
            "moody"
        ]
        
        # Add cinematic enhancements
        enhanced = f"{base_prompt}, {', '.join(cinematic_terms[:4])}"
        
        # Add quality boosters
        quality_terms = ["masterpiece", "best quality", "ultra detailed", "8k resolution"]
        enhanced += f", {', '.join(quality_terms[:2])}"
        
        return enhanced
    
    def _get_negative_prompt(self) -> str:
        """Get negative prompt to avoid common artifacts"""
        negative_terms = [
            "blurry",
            "low quality", 
            "worst quality",
            "jpeg artifacts",
            "watermark",
            "signature",
            "text",
            "cropped",
            "malformed",
            "bad anatomy",
            "disfigured",
            "poorly drawn",
            "extra limbs",
            "missing limbs",
            "floating limbs",
            "disconnected limbs",
            "long neck",
            "long body",
            "duplicate",
            "mutation",
            "poorly drawn face",
            "poorly drawn hands",
            "poorly drawn feet"
        ]
        
        return ", ".join(negative_terms)
    
    def create_style_reference(self, reference_prompt: str, output_path: str) -> str:
        """Create a reference image to maintain style consistency"""
        if not self.model_loaded:
            self.load_model()
        
        enhanced_prompt = self._enhance_cinematic_prompt(reference_prompt)
        negative_prompt = self._get_negative_prompt()
        
        return self._generate_single_image(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            output_path=output_path,
            seed=self.base_seed
        )
    
    def batch_generate_variations(
        self, 
        prompt: str, 
        num_variations: int, 
        output_dir: str
    ) -> List[str]:
        """Generate multiple variations of the same prompt"""
        if not self.model_loaded:
            self.load_model()
        
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        
        enhanced_prompt = self._enhance_cinematic_prompt(prompt)
        negative_prompt = self._get_negative_prompt()
        
        for i in range(num_variations):
            seed = self.base_seed + i * 1000  # Spread out seeds
            output_path = os.path.join(output_dir, f"variation_{i+1:02d}.png")
            
            path = self._generate_single_image(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                output_path=output_path,
                seed=seed
            )
            paths.append(path)
        
        return paths
    
    def create_image_grid(self, image_paths: List[str], output_path: str) -> str:
        """Create a grid of images for preview"""
        images = [Image.open(path) for path in image_paths]
        
        # Calculate grid dimensions
        n_images = len(images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        
        # Create grid
        grid = make_image_grid(images, rows=rows, cols=cols)
        grid.save(output_path, quality=95)
        
        return output_path
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.model_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model cleanup completed")

class LoRAManager:
    """Manages LoRA adapters for style consistency"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.loaded_loras = {}
    
    def load_lora(self, lora_path: str, adapter_name: str, weight: float = 0.8):
        """Load a LoRA adapter"""
        try:
            self.pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            self.loaded_loras[adapter_name] = weight
            print(f"Loaded LoRA: {adapter_name} with weight {weight}")
        except Exception as e:
            print(f"Error loading LoRA {adapter_name}: {e}")
    
    def set_lora_weights(self, weights_dict: Dict[str, float]):
        """Set weights for loaded LoRA adapters"""
        for adapter_name, weight in weights_dict.items():
            if adapter_name in self.loaded_loras:
                self.pipeline.set_adapters([adapter_name], adapter_weights=[weight])
                self.loaded_loras[adapter_name] = weight

def create_diffusion_generator(config_path: str = None, config_dict: Dict = None) -> DiffusionImageGenerator:
    """
    Factory function to create a DiffusionImageGenerator
    
    Args:
        config_path: Path to YAML config file
        config_dict: Dictionary with configuration parameters
        
    Returns:
        Configured DiffusionImageGenerator instance
    """
    if config_dict is not None:
        config = config_dict
    elif config_path is not None:
        import yaml
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        config = full_config.get('diffusion', {})
    else:
        # Default configuration
        config = {
            'base_model': 'runwayml/stable-diffusion-v1-5',
            'guidance_scale': 7.5,
            'num_inference_steps': 30,
            'height': 512,
            'width': 512,
            'seed': 42,
            'dtype': 'fp16'
        }
    
    return DiffusionImageGenerator(config)