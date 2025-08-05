#!/usr/bin/env python3
"""
LoRA Manager for dynamic character model switching during video generation
"""

import os
import torch
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json
from safetensors.torch import load_file

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
                    
                    # Try to read the file header to check if it's corrupted
                    try:
                        with open(file_path, 'rb') as f:
                            header = f.read(16)
                            if len(header) < 16:
                                return False, f"Corrupted {required_file} in {subdir_name} directory"
                    except (OSError, IOError):
                        return False, f"Cannot read {required_file} in {subdir_name} directory"
        
        return True, "Valid LoRA model structure"
        
    except (OSError, IOError, PermissionError) as e:
        return False, f"Error accessing directory: {e}"

class LoRAManager:
    def __init__(self, model, device):
        """
        Initialize LoRA Manager
        
        Args:
            model: The base diffusion model
            device: torch device (cuda/cpu)
        """
        self.model = model
        self.device = device
        self.current_loras = {}
        self.lora_cache = {}
        self.original_weights = {}  # Store original weights for restoration
        self.lora_scale = 1.0  # Default LoRA scaling factor
        
    def load_character_loras(self, characters: List) -> None:
        """
        Preload all character LoRA weights into cache
        
        Args:
            characters: List of Character objects with model_path
        """
        for character in characters:
            if character.model_path and os.path.exists(character.model_path):
                try:
                    # Check for DreamBooth LoRA structure: unet/ and text_encoder/ subdirectories
                    unet_dir = os.path.join(character.model_path, "unet")
                    text_encoder_dir = os.path.join(character.model_path, "text_encoder")
                    
                    if os.path.exists(unet_dir) and os.path.exists(text_encoder_dir):
                        # DreamBooth structure - load adapter files
                        unet_adapter = os.path.join(unet_dir, "adapter_model.safetensors")
                        text_encoder_adapter = os.path.join(text_encoder_dir, "adapter_model.safetensors")
                        
                        if os.path.exists(unet_adapter) and os.path.exists(text_encoder_adapter):
                            print(f"Loading DreamBooth LoRA for {character.name} from {character.model_path}")
                            
                            # Load the actual LoRA weights
                            unet_weights = load_file(unet_adapter)
                            text_encoder_weights = load_file(text_encoder_adapter)
                            
                            # Debug: Print available LoRA keys
                            print(f"  UNet LoRA keys ({len(unet_weights)}): {list(unet_weights.keys())[:5]}{'...' if len(unet_weights) > 5 else ''}")
                            print(f"  TextEncoder LoRA keys ({len(text_encoder_weights)}): {list(text_encoder_weights.keys())[:5]}{'...' if len(text_encoder_weights) > 5 else ''}")
                            
                            # Load adapter configs for proper scaling
                            unet_config_path = os.path.join(unet_dir, "adapter_config.json")
                            text_config_path = os.path.join(text_encoder_dir, "adapter_config.json")
                            
                            unet_config = {}
                            text_config = {}
                            
                            if os.path.exists(unet_config_path):
                                with open(unet_config_path, 'r') as f:
                                    unet_config = json.load(f)
                            
                            if os.path.exists(text_config_path):
                                with open(text_config_path, 'r') as f:
                                    text_config = json.load(f)
                            
                            # Store the weights and configs
                            self.lora_cache[character.name] = {
                                'unet_weights': unet_weights,
                                'text_encoder_weights': text_encoder_weights,
                                'unet_config': unet_config,
                                'text_config': text_config,
                                'model_path': character.model_path
                            }
                            
                        else:
                            print(f"Warning: Missing adapter_model.safetensors files for {character.name}")
                    else:
                        # Fallback: Look for direct safetensors files (old format)
                        lora_files = list(Path(character.model_path).glob("*.safetensors"))
                        if not lora_files:
                            lora_files = list(Path(character.model_path).glob("pytorch_lora_weights.safetensors"))
                        
                        if lora_files:
                            lora_path = str(lora_files[0])
                            print(f"Loading legacy LoRA for {character.name} from {lora_path}")
                            
                            # Try to load as safetensors or pickle
                            try:
                                if lora_path.endswith('.safetensors'):
                                    legacy_weights = load_file(lora_path)
                                else:
                                    legacy_weights = torch.load(lora_path, map_location=self.device)
                                
                                self.lora_cache[character.name] = {
                                    'legacy_weights': legacy_weights,
                                    'model_path': character.model_path
                                }
                            except Exception as e:
                                print(f"Error loading legacy LoRA weights: {e}")
                        else:
                            print(f"Warning: No LoRA files found for {character.name} in {character.model_path}")
                            
                except Exception as e:
                    print(f"Error loading LoRA for {character.name}: {e}")
    
    def apply_character_loras(self, characters: List) -> None:
        """
        Apply LoRA weights for specific characters
        
        Args:
            characters: List of Character objects to apply
        """
        # Clear current LoRAs first
        self.clear_loras()
        
        if not characters:
            return
            
        try:
            for character in characters:
                if character.name in self.lora_cache:
                    print(f"Applying LoRA for {character.name}")
                    self._inject_lora_weights(character.name, self.lora_cache[character.name])
                    self.current_loras[character.name] = True
                else:
                    print(f"Warning: No cached LoRA found for {character.name}")
                    
        except Exception as e:
            print(f"Error applying LoRAs: {e}")
    
    def clear_loras(self) -> None:
        """
        Remove all currently applied LoRA weights
        """
        try:
            if self.current_loras:
                print(f"Removing LoRAs for: {list(self.current_loras.keys())}")
                self._remove_lora_weights()
                self.current_loras.clear()
        except Exception as e:
            print(f"Error clearing LoRAs: {e}")
    
    def _inject_lora_weights(self, character_name: str, lora_data: Dict[str, Any]) -> None:
        """
        Inject LoRA weights into the model
        """
        try:
            # Handle DreamBooth structure
            if 'unet_weights' in lora_data and 'text_encoder_weights' in lora_data:
                print(f"Injecting DreamBooth LoRA weights for {character_name}")
                
                # Apply UNet LoRA weights
                self._apply_lora_to_module(
                    self.model.model.diffusion_model, 
                    lora_data['unet_weights'], 
                    f"{character_name}_unet"
                )
                
                # Apply Text Encoder LoRA weights (if model has text encoder)
                if hasattr(self.model, 'cond_stage_model'):
                    self._apply_lora_to_module(
                        self.model.cond_stage_model, 
                        lora_data['text_encoder_weights'], 
                        f"{character_name}_text_encoder"
                    )
                
            # Handle legacy format
            elif 'legacy_weights' in lora_data:
                print(f"Injecting legacy LoRA weights for {character_name}")
                self._apply_legacy_lora(lora_data['legacy_weights'], character_name)
                
        except Exception as e:
            print(f"Error injecting LoRA weights for {character_name}: {e}")
    
    def _apply_lora_to_module(self, module, lora_weights: Dict[str, torch.Tensor], prefix: str) -> None:
        """
        Apply LoRA weights to a specific module
        """
        # Group LoRA weights by base parameter name
        lora_pairs = {}
        
        # Parse all LoRA keys to find matching A/B pairs (DreamBooth convention)
        for key in lora_weights.keys():
            # Handle both naming conventions: lora_A/lora_B and lora_up/lora_down
            if 'lora_A.weight' in key:
                base_name = key.replace('.lora_A.weight', '')
                # Strip common prefixes from DreamBooth training
                base_name = base_name.replace('base_model.model.', '').replace('base_model.', '')
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {}
                lora_pairs[base_name]['A'] = key
            elif 'lora_B.weight' in key:
                base_name = key.replace('.lora_B.weight', '')
                # Strip common prefixes from DreamBooth training
                base_name = base_name.replace('base_model.model.', '').replace('base_model.', '')
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {}
                lora_pairs[base_name]['B'] = key
            elif 'lora_up.weight' in key:
                base_name = key.replace('.lora_up.weight', '')
                # Strip common prefixes from DreamBooth training
                base_name = base_name.replace('base_model.model.', '').replace('base_model.', '')
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {}
                lora_pairs[base_name]['up'] = key
            elif 'lora_down.weight' in key:
                base_name = key.replace('.lora_down.weight', '')
                # Strip common prefixes from DreamBooth training
                base_name = base_name.replace('base_model.model.', '').replace('base_model.', '')
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {}
                lora_pairs[base_name]['down'] = key
        
        print(f"Found {len(lora_pairs)} LoRA pairs")
        
        applied_count = 0
        
        # Apply LoRA weights to matching parameters - now that we're training on the correct model,
        # we should have exact parameter name matches
        for name, param in module.named_parameters():
            if 'weight' not in name:
                continue
                
            # Convert model parameter name to LoRA base name format
            param_base = name.replace('.weight', '')
            
            # Check if we can find this parameter in any of the LoRA pairs
            # Try direct match first, then with various prefix combinations
            matching_pair = None
            matching_base = None
            
            # Direct match (no prefix)
            if param_base in lora_pairs:
                pair = lora_pairs[param_base]
                if ('A' in pair and 'B' in pair) or ('up' in pair and 'down' in pair):
                    matching_pair = pair
                    matching_base = param_base
            
            # If no direct match, try finding any LoRA pair that ends with our parameter name
            if not matching_pair:
                for lora_base, pair in lora_pairs.items():
                    if lora_base.endswith(param_base) and (('A' in pair and 'B' in pair) or ('up' in pair and 'down' in pair)):
                        matching_pair = pair
                        matching_base = lora_base
                        break
            
            # If still no match, try the other way - see if our parameter name ends with any LoRA base
            if not matching_pair:
                for lora_base, pair in lora_pairs.items():
                    if param_base.endswith(lora_base) and (('A' in pair and 'B' in pair) or ('up' in pair and 'down' in pair)):
                        matching_pair = pair
                        matching_base = lora_base
                        break
            
            if matching_pair:
                try:
                    # Store original weight if not already stored
                    full_name = f"{prefix}.{name}"
                    if full_name not in self.original_weights:
                        self.original_weights[full_name] = param.data.clone()
                    
                    # Get LoRA weights (handle both A/B and up/down conventions)
                    if 'A' in matching_pair and 'B' in matching_pair:
                        lora_A = lora_weights[matching_pair['A']].to(param.device)
                        lora_B = lora_weights[matching_pair['B']].to(param.device)
                        # DreamBooth convention: B @ A (B is "up", A is "down")
                        lora_up = lora_B
                        lora_down = lora_A
                    else:
                        lora_up = lora_weights[matching_pair['up']].to(param.device)
                        lora_down = lora_weights[matching_pair['down']].to(param.device)
                    
                    # Calculate LoRA delta: up @ down * scale
                    if lora_up.dim() >= 2 and lora_down.dim() >= 2:
                        # Reshape for matrix multiplication if needed
                        if lora_up.dim() > 2:
                            lora_up = lora_up.view(lora_up.shape[0], -1)
                        if lora_down.dim() > 2:
                            lora_down = lora_down.view(-1, lora_down.shape[-1])
                        
                        # Matrix multiplication: up @ down
                        lora_delta = torch.mm(lora_up, lora_down) * self.lora_scale
                        
                        # Reshape back to original parameter shape if necessary
                        if lora_delta.shape != param.shape:
                            lora_delta = lora_delta.view(param.shape)
                    else:
                        print(f"Unsupported LoRA tensor dimensions for {name}: up={lora_up.shape}, down={lora_down.shape}")
                        continue
                    
                    # Apply LoRA to the parameter
                    if lora_delta.shape == param.shape:
                        param.data = self.original_weights[full_name] + lora_delta
                        applied_count += 1
                    else:
                        print(f"Shape mismatch for {name}: {lora_delta.shape} vs {param.shape}")
                        
                except Exception as e:
                    print(f"Error applying LoRA to {name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Applied LoRA to {applied_count} parameters in {prefix}")
    
    def _apply_legacy_lora(self, legacy_weights: Dict[str, torch.Tensor], character_name: str) -> None:
        """
        Apply legacy format LoRA weights
        """
        # This is a simplified version - you may need to adapt based on your specific legacy format
        print(f"Legacy LoRA application for {character_name} - simplified implementation")
        # Legacy format handling would go here based on your specific format
    
    def _remove_lora_weights(self) -> None:
        """
        Remove LoRA weights by restoring original weights
        """
        try:
            # Restore all original weights
            for full_name, original_weight in self.original_weights.items():
                # Parse the full name to get module and parameter name
                if '.' in full_name:
                    prefix, param_name = full_name.split('.', 1)
                    
                    # Find the parameter and restore it
                    if 'unet' in prefix:
                        module = self.model.model.diffusion_model
                    elif 'text_encoder' in prefix:
                        if hasattr(self.model, 'cond_stage_model'):
                            module = self.model.cond_stage_model
                        else:
                            continue
                    else:
                        continue
                    
                    # Navigate to the parameter and restore original weight
                    param = module
                    for part in param_name.split('.'):
                        if hasattr(param, part):
                            param = getattr(param, part)
                        else:
                            break
                    else:
                        if hasattr(param, 'data'):
                            param.data = original_weight.clone()
                            
            # Clear the original weights cache
            self.original_weights.clear()
            print("Restored all original weights")
            
        except Exception as e:
            print(f"Error removing LoRA weights: {e}")
    
    def get_active_characters(self) -> List[str]:
        """
        Get list of currently active character names
        """
        return list(self.current_loras.keys())
    
    def set_lora_scale(self, scale: float) -> None:
        """
        Set the global LoRA scaling factor
        
        Args:
            scale: Scaling factor for LoRA weights (default: 1.0)
        """
        self.lora_scale = scale
        print(f"LoRA scale set to {scale}")
    
    def validate_lora_application(self) -> Dict[str, bool]:
        """
        Validate that LoRA weights have been properly applied to the model
        
        Returns:
            Dictionary mapping character names to validation status
        """
        validation_results = {}
        
        if not self.current_loras:
            print("No LoRA adapters currently applied")
            return validation_results
        
        for character_name in self.current_loras.keys():
            # Check if we have stored original weights (indicates LoRA was applied)
            character_weights = [key for key in self.original_weights.keys() if character_name in key]
            
            if character_weights:
                # Sample a few parameters to verify they've been modified
                modified_count = 0
                total_checked = 0
                
                # Check first few parameters for modification
                for full_name in character_weights[:5]:  # Check up to 5 parameters
                    total_checked += 1
                    try:
                        # Navigate to the current parameter
                        if 'unet' in full_name:
                            module = self.model.model.diffusion_model
                        elif 'text_encoder' in full_name:
                            if hasattr(self.model, 'cond_stage_model'):
                                module = self.model.cond_stage_model
                            else:
                                continue
                        else:
                            continue
                        
                        param_name = full_name.split('.', 1)[1]
                        param = module
                        for part in param_name.split('.'):
                            if hasattr(param, part):
                                param = getattr(param, part)
                            else:
                                break
                        else:
                            if hasattr(param, 'data'):
                                # Check if current weight differs from original
                                original_weight = self.original_weights[full_name]
                                if not torch.equal(param.data, original_weight):
                                    modified_count += 1
                                        
                    except Exception as e:
                        print(f"  Error checking parameter {full_name}: {e}")
                        continue
                
                # Consider LoRA applied if at least some parameters were modified
                is_valid = modified_count > 0
                validation_results[character_name] = is_valid
                
                if is_valid:
                    print(f"LoRA validation for {character_name}: PASSED ({modified_count}/{total_checked} params modified, {len(character_weights)} total)")
                else:
                    print(f"LoRA validation for {character_name}: FAILED (no parameters modified)")
            else:
                validation_results[character_name] = False
                print(f"LoRA validation for {character_name}: FAILED (no original weights stored)")
        
        return validation_results
    
    def get_lora_influence_summary(self) -> str:
        """
        Get a summary of current LoRA influence on the model
        """
        if not self.current_loras:
            return "No LoRA adapters currently active"
        
        summary = f"Active LoRA adapters: {len(self.current_loras)}\n"
        summary += f"LoRA scale: {self.lora_scale}\n"
        summary += f"Characters: {list(self.current_loras.keys())}\n"
        summary += f"Modified parameters: {len(self.original_weights)}"
        
        return summary
    
    def _find_lora_layers(self, module, target_layers=None):
        """
        Find layers that typically have LoRA applied (attention layers, linear layers)
        """
        if target_layers is None:
            target_layers = ['to_q', 'to_k', 'to_v', 'to_out', 'ff.net', 'proj_in', 'proj_out']
        
        lora_layers = {}
        for name, child in module.named_modules():
            for target in target_layers:
                if target in name and hasattr(child, 'weight'):
                    lora_layers[name] = child
        
        return lora_layers
