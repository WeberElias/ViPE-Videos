#!/usr/bin/env python3
# filepath: /home/webereli/bar/ViPE-Videos/helpers/train_dreambooth_script.py

import subprocess
import sys
import os
from pathlib import Path

def train_dreambooth(
    model_name,
    instance_dir,
    class_dir, 
    output_dir,
    instance_prompt,
    class_prompt,
    resolution,
    train_batch_size,
    learning_rate,
    max_train_steps,
    lora_r,
    lora_alpha,
    lora_text_encoder_r,
    lora_text_encoder_alpha,
    num_class_images
):
    """
    Train a Dreambooth model with LoRA
    
    All parameters are required and passed from train_character()
    """
    
    # Validate directories exist
    if not os.path.exists(instance_dir):
        raise ValueError(f"Instance directory does not exist: {instance_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(class_dir).mkdir(parents=True, exist_ok=True)
    
    # Path to your existing dreambooth training script
    script_path = "dreambooth/examples/lora_dreambooth/train_dreambooth.py"
    
    # Build command using your existing script with optimized parameters
    cmd = [
        "accelerate", "launch", script_path,
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
        f"--max_train_steps={max_train_steps}",
        "--mixed_precision=fp16",
        "--gradient_accumulation_steps=2",
        "--checkpointing_steps=200",
        "--enable_xformers_memory_efficient_attention",
        "--seed=42"
    ]
    
    print("Starting Dreambooth training...")
    print(f"Base model: {model_name}")
    print(f"Training steps: {max_train_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_r}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("LoRA adapters saved to:", output_dir)
        return True, output_dir
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        return False, None
    except FileNotFoundError:
        print("Error: 'accelerate' command not found. Please install accelerate:")
        print("pip install accelerate")
        return False, None

def train_character(character, saving_dir):
    """
    Train a character using DreamBooth LoRA with fixed parameters
    """
    print(f"Training character: {character.name}")
    
    # Fixed training parameters - change here to modify training settings
    num_class_images = 1000
    max_train_steps = 400
    learning_rate = 0.0002
    
    # Prepare character-specific paths and prompts
    folder_name = character.name.lower().replace(' ', '_').replace(',', '')
    output_dir = os.path.join(saving_dir, "models", folder_name)
    class_dir = os.path.join(saving_dir, "class_images")
    
    # Use the character's unique identifier for consistent training
    instance_prompt = f"a photo of {character.unique_identifier} {character.description.split(',')[0].strip()}"
    class_prompt = f"a photo of {character.description.split(',')[0].strip()}"
    
    print(f"Instance prompt: {instance_prompt}")
    print(f"Class prompt: {class_prompt}")
    print(f"Output directory: {output_dir}")
    print(f"Training with: {num_class_images} class images, {max_train_steps} steps, lr={learning_rate}")
    
    # Validate training data
    if not hasattr(character, 'training_images') or not os.path.exists(character.training_images):
        print(f"Error: No training images found for {character.name}")
        return False, None
    
    image_files = [f for f in os.listdir(character.training_images) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) < 3:
        print(f"Error: Need at least 3 training images for {character.name}, found {len(image_files)}")
        return False, None
    
    print(f"Found {len(image_files)} training images")
    
    # Call the training function with all required parameters
    try:
        success, model_path = train_dreambooth(
            model_name="runwayml/stable-diffusion-v1-5",
            instance_dir=character.training_images,
            class_dir=class_dir,
            output_dir=output_dir,
            instance_prompt=instance_prompt,
            class_prompt=class_prompt,
            resolution=512,
            train_batch_size=1,
            learning_rate=learning_rate,
            max_train_steps=max_train_steps,
            lora_r=16,
            lora_alpha=27,
            lora_text_encoder_r=16,
            lora_text_encoder_alpha=17,
            num_class_images=num_class_images
        )
        
        if success:
            print(f"Successfully trained character: {character.name}")
            return True, model_path
        else:
            print(f"Failed to train character: {character.name}")
            return False, None
            
    except Exception as e:
        print(f"Error training character {character.name}: {e}")
        return False, None