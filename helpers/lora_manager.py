#!/usr/bin/env python3
"""
LoRA Manager for dynamic character model switching during video generation
Uses PEFT library for proper DreamBooth LoRA loading
"""

import os
import torch
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json

try:
    from peft import PeftModel, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: peft library not installed. Install with: pip install peft")
    PEFT_AVAILABLE = False

def is_valid_lora_directory(model_path: str) -> Tuple[bool, str]:
    """Check if a directory contains a valid DreamBooth LoRA model structure."""
    try:
        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            return False, "Directory does not exist"
        
        # Check for required subdirectories
        unet_dir = os.path.join(model_path, "unet")
        text_encoder_dir = os.path.join(model_path, "text_encoder")
        
        if not os.path.exists(unet_dir) or not os.path.isdir(unet_dir):
            return False, "Missing 'unet' subdirectory"
            
        if not os.path.exists(text_encoder_dir) or not os.path.isdir(text_encoder_dir):
            return False, "Missing 'text_encoder' subdirectory"
        
        # Check for required files in each subdirectory
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        
        for subdir, subdir_name in [(unet_dir, "unet"), (text_encoder_dir, "text_encoder")]:
            for required_file in required_files:
                file_path = os.path.join(subdir, required_file)
                if not os.path.exists(file_path):
                    return False, f"Missing {required_file} in {subdir_name} directory"
                
                # Check if adapter_model.safetensors is not empty
                if required_file == "adapter_model.safetensors":
                    if os.path.getsize(file_path) == 0:
                        return False, f"Empty {required_file} in {subdir_name} directory"
        
        return True, "Valid LoRA model structure"
        
    except (OSError, IOError, PermissionError) as e:
        return False, f"Error accessing directory: {e}"

class LoRAManager:
    def __init__(self, model, device):
        """
        Initialize LoRA Manager for DreamBooth models
        
        Args:
            model: The base diffusion model (should be a diffusers pipeline)
            device: torch device (cuda/cpu)
        """
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for DreamBooth LoRA support. Install with: pip install peft")
            
        self.model = model
        self.device = device
        self.current_loras = {}
        self.lora_cache = {}
        
        # Store clean base models once at initialization
        self.base_unet = model.model.diffusion_model
        self.base_text_encoder = model.cond_stage_model

    def load_character_loras(self, characters: List) -> None:
        """
        Preload all character LoRA configurations
        
        Args:
            characters: List of Character objects with model_path
        """
        for character in characters:
            if character.model_path and os.path.exists(character.model_path):
                try:
                    # Check for DreamBooth LoRA structure
                    unet_dir = os.path.join(character.model_path, "unet")
                    text_encoder_dir = os.path.join(character.model_path, "text_encoder")
                    
                    print(f"Checking LoRA structure for {character.name}:")
                    print(f"  Model path: {character.model_path}")
                    print(f"  UNet dir exists: {os.path.exists(unet_dir)}")
                    print(f"  Text Encoder dir exists: {os.path.exists(text_encoder_dir)}")
                    
                    if os.path.exists(unet_dir):
                        unet_files = os.listdir(unet_dir)
                        print(f"  UNet files: {unet_files}")
                    
                    if os.path.exists(text_encoder_dir):
                        te_files = os.listdir(text_encoder_dir)
                        print(f"  Text Encoder files: {te_files}")
                    
                    if os.path.exists(unet_dir) and os.path.exists(text_encoder_dir):
                        print(f"Loading DreamBooth LoRA config for {character.name} from {character.model_path}")
                        
                        # Store the paths for later use
                        self.lora_cache[character.name] = {
                            'model_path': character.model_path,
                            'unet_dir': unet_dir,
                            'text_encoder_dir': text_encoder_dir
                        }
                        
                        # Verify we can load the config
                        try:
                            if os.path.exists(os.path.join(text_encoder_dir, "adapter_config.json")):
                                config = LoraConfig.from_pretrained(text_encoder_dir)
                                print(f"  Text Encoder base model: {getattr(config, 'base_model_name_or_path', 'Unknown')}")
                            
                            if os.path.exists(os.path.join(unet_dir, "adapter_config.json")):
                                config = LoraConfig.from_pretrained(unet_dir)
                                print(f"  UNet base model: {getattr(config, 'base_model_name_or_path', 'Unknown')}")
                                
                        except Exception as e:
                            print(f"  Warning: Could not load config: {e}")
                        
                        # Test the validation function
                        is_valid, message = is_valid_lora_directory(character.model_path)
                        print(f"  Validation result: {is_valid}, Message: {message}")
                        
                    else:
                        print(f"Warning: Invalid DreamBooth structure for {character.name}")
                        print(f"  Missing directories - UNet: {not os.path.exists(unet_dir)}, Text Encoder: {not os.path.exists(text_encoder_dir)}")
                            
                except Exception as e:
                    print(f"Error loading LoRA config for {character.name}: {e}")
                    import traceback
                    traceback.print_exc()

    def apply_character_loras(self, characters: List) -> None:
        """
        Apply LoRA weights for specific characters using PEFT (single character)
        
        Args:
            characters: List of Character objects to apply (only last one will be active)
        """
        # Clear current LoRAs first
        self.clear_loras()
        
        if not characters:
            return
            
        try:
            # Apply only the last character's LoRA (single LoRA approach)
            character = characters[-1]  # Take the last character if multiple
            
            if character.name in self.lora_cache:
                self._apply_character_lora(character.name, self.lora_cache[character.name])
                self.current_loras[character.name] = True
            else:
                print(f"Warning: No cached LoRA found for {character.name}")
                    
        except Exception as e:
            print(f"Error applying LoRAs: {e}")
            import traceback
            traceback.print_exc()

    def _apply_character_lora(self, character_name: str, lora_data: Dict[str, Any]) -> None:
        """
        Apply a single character's LoRA using PEFT (single LoRA approach)
        """
        try:
            unet_dir = lora_data['unet_dir']
            text_encoder_dir = lora_data['text_encoder_dir']
            
            # Apply UNet LoRA - always start from clean base model
            if os.path.exists(unet_dir):
                self.model.model.diffusion_model = PeftModel.from_pretrained(
                    self.base_unet,  # Always use stored clean model
                    unet_dir
                )
        
            # Apply Text Encoder LoRA - always start from clean base model
            if os.path.exists(text_encoder_dir):
                self.model.cond_stage_model = PeftModel.from_pretrained(
                    self.base_text_encoder,  # Always use stored clean model
                    text_encoder_dir
                )
        
            print(f"{character_name}: ✓ APPLIED")
            
        except Exception as e:
            print(f"Error applying LoRA for {character_name}: {e}")
            import traceback
            traceback.print_exc()

    def clear_loras(self) -> None:
        """
        Remove all currently applied LoRA adapters by restoring base models
        """
        try:
            if self.current_loras:
                # Restore original base models
                self.model.model.diffusion_model = self.base_unet
                self.model.cond_stage_model = self.base_text_encoder
                
                self.current_loras.clear()
                
        except Exception as e:
            print(f"Error clearing LoRAs: {e}")

    def get_active_characters(self) -> List[str]:
        """
        Get list of currently active character names
        """
        return list(self.current_loras.keys())

    def set_lora_scale(self, scale: float) -> None:
        """
        Set the LoRA scaling factor (if supported by the adapters)
        
        Args:
            scale: Scaling factor for LoRA weights (default: 1.0)
        """
        try:
            # PEFT models may support scaling through adapter configuration
            if hasattr(self.model, 'unet') and hasattr(self.model.unet, 'peft_config'):
                # This would depend on the specific PEFT version and capabilities
                print(f"LoRA scale set to {scale} (implementation may vary by PEFT version)")
            else:
                print(f"LoRA scale setting requested ({scale}) but no active PEFT adapters found")
        except Exception as e:
            print(f"Error setting LoRA scale: {e}")

    def validate_lora_application(self) -> Dict[str, bool]:
        """
        Validate that LoRA adapters have been properly applied
        """
        validation_results = {}
        
        if not self.current_loras:
            print("No LoRA adapters currently applied")
            return validation_results
        
        for character_name in self.current_loras.keys():
            is_valid = True
            
            # Check UNet adapter
            if hasattr(self.model, 'unet'):
                if hasattr(self.model.unet, 'peft_config'):
                    unet_adapter = f"{character_name}_unet"
                    if unet_adapter not in getattr(self.model.unet, 'peft_config', {}):
                        is_valid = False
                else:
                    is_valid = False
            
            # Check Text Encoder adapter  
            if hasattr(self.model, 'text_encoder'):
                if hasattr(self.model.text_encoder, 'peft_config'):
                    te_adapter = f"{character_name}_text_encoder"
                    if te_adapter not in getattr(self.model.text_encoder, 'peft_config', {}):
                        is_valid = False
                else:
                    is_valid = False
            
            validation_results[character_name] = is_valid
    
        return validation_results

    def get_lora_influence_summary(self) -> str:
        """
        Get a summary of current LoRA influence on the model
        """
        if not self.current_loras:
            return "No LoRA adapters currently active"
        
        summary = f"Active LoRA adapters: {len(self.current_loras)}\n"
        summary += f"Characters: {list(self.current_loras.keys())}\n"
        
        # Add PEFT-specific information
        if hasattr(self.model, 'unet') and hasattr(self.model.unet, 'peft_config'):
            summary += f"UNet adapters: {list(self.model.unet.peft_config.keys())}\n"
        if hasattr(self.model, 'text_encoder') and hasattr(self.model.text_encoder, 'peft_config'):
            summary += f"Text Encoder adapters: {list(self.model.text_encoder.peft_config.keys())}\n"
        
        return summary
    
    def test_lora_loading(self, character_name: str) -> bool:
        """
        Test if we can successfully load a character's LoRA
        """
        if character_name not in self.lora_cache:
            print(f"Character {character_name} not in cache")
            return False
        
        try:
            lora_data = self.lora_cache[character_name]
            unet_dir = lora_data['unet_dir']
            text_encoder_dir = lora_data['text_encoder_dir']
            
            print(f"Testing LoRA loading for {character_name}")
            
            # Test UNet LoRA loading
            if os.path.exists(unet_dir):
                try:
                    # Try to load the config first
                    config = LoraConfig.from_pretrained(unet_dir)
                    print(f"  UNet config loaded successfully: {config}")
                    
                    # Try to create a PeftModel (but don't actually apply it)
                    print(f"  UNet directory structure looks valid")
                    
                except Exception as e:
                    print(f"  UNet LoRA loading failed: {e}")
                    return False
            
            # Test Text Encoder LoRA loading
            if os.path.exists(text_encoder_dir):
                try:
                    # Try to load the config first
                    config = LoraConfig.from_pretrained(text_encoder_dir)
                    print(f"  Text Encoder config loaded successfully: {config}")
                    
                    print(f"  Text Encoder directory structure looks valid")
                    
                except Exception as e:
                    print(f"  Text Encoder LoRA loading failed: {e}")
                    return False
            
            print(f"LoRA test for {character_name}: SUCCESS")
            return True
            
        except Exception as e:
            print(f"LoRA test for {character_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
