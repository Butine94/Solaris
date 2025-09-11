#!/usr/bin/env python3
"""
Training Script for Custom LoRA Adapters
Trains style-specific LoRA adapters for consistent film generation
"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import logging
from tqdm import tqdm

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from diffusers.loaders import AttnProcsLayers
    from diffusers.models.attention_processor import LoRAAttnProcessor
    from diffusers.optimization import get_scheduler
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install diffusers transformers accelerate")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StyleDataset(Dataset):
    """Dataset for training style-consistent LoRA adapters"""
    
    def __init__(self, image_paths: List[str], prompts: List[str], size: int = 512):
        self.image_paths = image_paths
        self.prompts = prompts
        self.size = size
        
        assert len(image_paths) == len(prompts), "Must have equal number of images and prompts"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        prompt = self.prompts[idx]
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return {
            'images': image,
            'prompts': prompt
        }

class LoRATrainer:
    """Trainer for LoRA adapters on diffusion models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._get_device()
        self.pipeline = None
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.noise_scheduler = None
        
    def _get_device(self):
        """Get the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def setup_model(self):
        """Setup the base diffusion model for training"""
        model_name = self.config.get('base_model', 'runwayml/stable-diffusion-v1-5')
        
        logger.info(f"Setting up model: {model_name}")
        
        # Load pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.get('dtype') == 'fp16' else torch.float32,
            use_safetensors=True
        )
        
        # Extract components
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.noise_scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        
        # Move to device
        self.unet.to(self.device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        
        # Set to train mode (only UNet will actually train)
        self.unet.train()
        self.vae.eval()
        self.text_encoder.eval()
        
        logger.info("Model setup complete")
    
    def setup_lora(self, rank: int = 4):
        """Setup LoRA layers in the UNet"""
        logger.info(f"Setting up LoRA with rank {rank}")
        
        # Create LoRA attention processors
        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
            )
        
        self.unet.set_attn_processor(lora_attn_procs)
        
        # Get trainable parameters
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
        
        logger.info(f"LoRA layers created with {sum(p.numel() for p in self.lora_layers.parameters())} parameters")
    
    def train(
        self,
        train_dataset: StyleDataset,
        output_dir: str,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        save_every: int = 25
    ):
        """Train the LoRA adapter"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup data loader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.lora_layers.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-08,
        )
        
        # Setup scheduler
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_epochs * len(train_dataloader),
        )
        
        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        global_step = 0
        for epoch in range(num_epochs):
            train_loss = 0.0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                images = batch['images'].to(self.device)
                prompts = batch['prompts']
                
                # Encode text
                text_inputs = self.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(self.device)
                
                with torch.no_grad():
                    text_embeddings = self.text_encoder(text_input_ids)[0]
                
                # Encode images to latent space
                with torch.no_grad():
                    latents = self.vae.encode(images).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predict noise
                model_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
                
                # Calculate loss
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Backward pass
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item()
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{train_loss / (step + 1):.4f}"
                })
            
            avg_train_loss = train_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_train_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
                self.save_lora_weights(checkpoint_dir)
                logger.info(f"Saved checkpoint: {checkpoint_dir}")
        
        logger.info("Training completed!")
    
    def save_lora_weights(self, output_dir: str):
        """Save LoRA weights"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LoRA weights
        self.pipeline.save_lora_weights(output_dir)
        
        # Save training config
        config_path = os.path.join(output_dir, "training_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def load_lora_weights(self, lora_path: str):
        """Load trained LoRA weights"""
        self.pipeline.load_lora_weights(lora_path)
        logger.info(f"Loaded LoRA weights from: {lora_path}")

def prepare_training_data(data_dir: str, style_name: str) -> tuple:
    """Prepare training data from a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    image_paths = []
    prompts = []
    
    # Look for images in the data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all images
    for file_path in data_path.rglob("*"):
        if file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))
            
            # Try to find corresponding text file for prompt
            txt_path = file_path.with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, 'r') as f:
                    prompt = f.read().strip()
            else:
                # Generate generic prompt with style
                prompt = f"{style_name} style, cinematic composition, professional photography"
            
            prompts.append(prompt)
    
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")
    
    logger.info(f"Found {len(image_paths)} images for training")
    return image_paths, prompts

def main():
    """Main training interface"""
    parser = argparse.ArgumentParser(description='Train LoRA adapters for film generation')
    parser.add_argument('--config', '-c', default='config.yaml', help='Config file path')
    parser.add_argument('--data-dir', required=True, help='Training data directory')
    parser.add_argument('--style-name', required=True, help='Style name for the LoRA')
    parser.add_argument('--output-dir', required=True, help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--rank', type=int, default=4, help='LoRA rank')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use diffusion config for training
    train_config = config.get('diffusion', {})
    
    try:
        # Prepare training data
        logger.info("Preparing training data...")
        image_paths, prompts = prepare_training_data(args.data_dir, args.style_name)
        
        # Create dataset
        dataset = StyleDataset(
            image_paths=image_paths,
            prompts=prompts,
            size=train_config.get('height', 512)
        )
        
        # Setup trainer
        trainer = LoRATrainer(train_config)
        trainer.setup_model()
        trainer.setup_lora(rank=args.rank)
        
        # Train
        trainer.train(
            train_dataset=dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
        logger.info(f"✅ Training completed! LoRA saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()