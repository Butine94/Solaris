import torch
from diffusers import StableDiffusionPipeline
import os

class DiffusionGenerator:
    
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name).to(self.device)

    def generate_images(self, prompts, output_dir="outputs/frames", height=512, width=512, seed=42):
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
            generator = torch.manual_seed(seed + i)
            
            # Add error handling
            try:
                result = self.pipe(prompt, height=height, width=width, generator=generator)
                image = result.images[0]
                
                # Debug: check if image is actually generated
                print(f"Image mode: {image.mode}, size: {image.size}")
                
                filename = os.path.join(output_dir, f"frame_{i+1:03d}.png")
                image.save(filename)
                image_paths.append(filename)
            except Exception as e:
                print(f"Error generating image {i}: {e}")
                
        return image_paths