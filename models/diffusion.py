import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
from typing import List, Dict

class CinematicDiffusionModel:
    """Fast, high-quality image generation for cinematic shots"""
    
    def __init__(self, config: Dict):
        self.config = config['diffusion']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self._load_model()
    
    def _load_model(self):
        """Load and optimize the diffusion model"""
        print(f"🚀 Loading model on {self.device}...")
        
        # Fix dtype based on device
        if self.device == "cpu":
            dtype = torch.float32
            print("⚠️ Using CPU - switching to float32")
        else:
            dtype = torch.float16 if self.config['dtype'] == 'fp16' else torch.float32
        
        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config['base_model'],
            dtype=dtype,  # Use correct dtype
            use_safetensors=True
        )
        
        # Optimize for speed
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory efficient attention if available (only on GPU)
        if self.device == "cuda":
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            
            # Enable xformers for faster inference
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✅ XFormers enabled for faster generation")
            except:
                print("⚠️ XFormers not available")
        
        print("✅ Model loaded and optimized!")
    
    def generate_shots(self, shots: List[Dict], output_dir: str) -> List[Dict]:
        """Generate images for all shots quickly"""
        if not self.pipe:
            raise RuntimeError("Model not loaded")
        
        # Set consistent seed for reproducibility
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config['seed'])
        
        updated_shots = []
        
        for i, shot in enumerate(shots):
            print(f"Generating shot {i+1}/5: {shot['type']}")
            
            # Build complete prompt
            prompt = self._enhance_prompt(shot['prompt'])
            negative_prompt = "blurry, low quality, distorted, amateur, ugly, deformed"
            
            # Generate image
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=self.config['height'],
                    width=self.config['width'],
                    num_inference_steps=self.config['num_inference_steps'],
                    guidance_scale=self.config['guidance_scale'],
                    generator=generator
                )
            
            # Save image
            image = result.images[0]
            image_path = os.path.join(output_dir, f"shot_{i+1}.png")
            image.save(image_path)
            
            # Update shot with image path
            shot['image_path'] = image_path
            updated_shots.append(shot)
            
            print(f"✓ Shot {i+1} saved to {image_path}")
        
        return updated_shots
    
    def _enhance_prompt(self, base_prompt: str) -> str:
        """Enhance prompt for cinematic quality"""
        enhancements = [
            "highly detailed",
            "8k resolution",
            "professional photography",
            "perfect composition",
            "atmospheric lighting",
            "cinematic color grading"
        ]
        
        return f"{base_prompt}, {', '.join(enhancements)}"
    
    def generate_single_image(self, prompt: str, output_path: str) -> str:
        """Generate a single high-quality image"""
        if not self.pipe:
            raise RuntimeError("Model not loaded")
        
        enhanced_prompt = self._enhance_prompt(prompt)
        negative_prompt = "blurry, low quality, distorted, amateur"
        
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config['seed'])
        
        with torch.no_grad():
            result = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                height=self.config['height'],
                width=self.config['width'],
                num_inference_steps=self.config['num_inference_steps'],
                guidance_scale=self.config['guidance_scale'],
                generator=generator
            )
        
        image = result.images[0]
        image.save(output_path)
        return output_path
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.pipe:
            del self.pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None