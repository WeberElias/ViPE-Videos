#!/usr/bin/env python3
# filepath: /home/webereli/bar/ViPE-Videos/train_dreambooth_script.py

import subprocess
import sys
import os
from pathlib import Path

def train_dreambooth(
    model_name="runwayml/stable-diffusion-v1-5",
    instance_dir="./instance_images",
    class_dir="./class_images", 
    output_dir="./dreambooth_output",
    instance_prompt="a photo of sks person",
    class_prompt="a photo of person",
    resolution=512,
    train_batch_size=1,
    learning_rate=1e-4,
    max_train_steps=800,
    lora_r=16,
    lora_alpha=27,
    lora_text_encoder_r=16,
    lora_text_encoder_alpha=17,
    num_class_images=200
):
    """
    Train a Dreambooth model with LoRA
    
    Args:
        model_name: Pretrained model name or path
        instance_dir: Directory containing instance images
        class_dir: Directory containing class images  
        output_dir: Directory to save trained model
        instance_prompt: Prompt for instance images
        class_prompt: Prompt for class images
        resolution: Training resolution
        train_batch_size: Batch size for training
        learning_rate: Learning rate
        max_train_steps: Maximum training steps
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_text_encoder_r: LoRA rank for text encoder
        lora_text_encoder_alpha: LoRA alpha for text encoder
        num_class_images: Number of class images to generate
    """
    
    # Validate directories exist
    if not os.path.exists(instance_dir):
        raise ValueError(f"Instance directory does not exist: {instance_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(class_dir).mkdir(parents=True, exist_ok=True)
    
    # Path to your existing dreambooth training script
    script_path = "dreambooth/examples/lora_dreambooth/train_dreambooth.py"
    
    # Build command using your existing script
    cmd = [
        "accelerate", "launch", script_path,  # Use the existing script
        f"--pretrained_model_name_or_path={model_name}",
        f"--instance_data_dir={instance_dir}",
        f"--class_data_dir={class_dir}",
        f"--output_dir={output_dir}",
        "--train_text_encoder",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        f"--instance_prompt={instance_prompt}",
        f"--class_prompt={class_prompt}",
        f"--resolution={resolution}",
        f"--train_batch_size={train_batch_size}",
        f"--num_class_images={num_class_images}",
        "--use_lora",
        f"--lora_r={lora_r}",
        f"--lora_alpha={lora_alpha}",
        f"--lora_text_encoder_r={lora_text_encoder_r}",
        f"--lora_text_encoder_alpha={lora_text_encoder_alpha}",
        f"--learning_rate={learning_rate}",
        f"--max_train_steps={max_train_steps}"
    ]
    
    print("Starting Dreambooth training with existing script...")
    print(f"Using script: {script_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        return True, output_dir
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print(f"Stderr: {e.stderr}")
        return False, None
    except FileNotFoundError:
        print("Error: 'accelerate' command not found. Please install accelerate:")
        print("pip install accelerate")
        return False, None

def train_character(character, base_model="runwayml/stable-diffusion-v1-5"):
    """
    Train a specific character using Dreambooth
    
    Args:
        character: Character object with training images path
        base_model: Base model to fine-tune
        
    Returns:
        tuple: (success, model_path)
    """
    if not character.training_images:
        print(f"No training images path set for character: {character.name}")
        return False, None
    
    # Create character-specific paths
    output_dir = f"./models/{character.name.lower().replace(' ', '_')}"
    class_dir = character.regularization_images or "./regularization_images"
    
    # Generate prompts
    instance_prompt = f"a photo of sks {character.description}"
    class_prompt = f"a photo of {character.description.split(',')[0]}"  # Use first part of description
    
    print(f"Training character: {character.name}")
    print(f"Instance prompt: {instance_prompt}")
    print(f"Class prompt: {class_prompt}")
    
    return train_dreambooth(
        model_name=base_model,
        instance_dir=character.training_images,
        class_dir=class_dir,
        output_dir=output_dir,
        instance_prompt=instance_prompt,
        class_prompt=class_prompt
    )

if __name__ == "__main__":
    # Example usage
    success, model_path = train_dreambooth(
        instance_dir="./training_images/alice",
        output_dir="./models/alice_model",
        instance_prompt="a photo of sks woman",
        class_prompt="a photo of woman"
    )
    
    if success:
        print(f"Training completed! Model saved to: {model_path}")
    else:
        print("Training failed!")