#!/usr/bin/env python3
"""
Minimal test script for AnimateDiff + DreamBooth LoRA combination
"""

import os
import torch
import subprocess
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from peft import PeftModel

def frames_to_video(frames_dir, output_video_path, fps=8):
    """
    Convert frames to video using ffmpeg
    
    Args:
        frames_dir (str): Directory containing frame_%03d.png files
        output_video_path (str): Path for output video file
        fps (int): Frames per second for the video
    """
    try:
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%03d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            output_video_path
        ]
        
        print(f"Converting frames to video: {output_video_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Video created successfully: {output_video_path}")
            return True
        else:
            print(f"ffmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to convert frames to video.")
        return False
    except Exception as e:
        print(f"Error converting frames to video: {e}")
        return False

def generate_variations(NAME):
    """
    Generate all prompt variations with different pre/postfixes
    """
    base_prompts = [
        f"sks{NAME} woman waving hello, detailed, high quality",
        f"sks{NAME} woman walking in a park, cinematic, smooth motion", 
        f"sks{NAME} woman smiling and nodding, portrait style"
    ]
    
    # Define postfixes
    postfixes = {
        "none": "",
        "vipe": ", extreme detail, high quality, HD, 32K, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant, intricate, trending on artstation",
        "vipe+background": ", extreme detail, high quality, HD, 32K, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant, intricate, detailed background, background",
        "background": ", detailed background, background",
        "background2": ", wide shot, establishing shot, scenic view, immersive environment, background in focus, balanced framing",
        "gemini": [
            "with an ancient stone bridge and an ivy-covered castle in the background.",
            "Golden hour light filters through autumn leaves, mist hangs low, a cinematic lens captures a rich, painterly bokeh.",
            "sun dappled park bench, bokeh background, warm summer evening light, cinematic feel"
        ]
    }
    
    # Define prefixes  
    prefixes = postfixes
    
    variations = []
    variation_names = []  # Track the names for each variation
    
    # First: no prefix/postfix
    for i, prompt in enumerate(base_prompts):
        variations.append(prompt)
        variation_names.append(f"postfix_none_base{i+1}")
    
    # Then for each postfix (except none and gemini)
    for postfix_name in ["vipe", "vipe+background", "background", "background2"]:
        for i, prompt in enumerate(base_prompts):
            variations.append(prompt + postfixes[postfix_name])
            variation_names.append(f"postfix_{postfix_name}_base{i+1}")
    
    # Gemini postfix (use enumerated answers)
    for i, prompt in enumerate(base_prompts):
        variations.append(prompt + ", " + postfixes["gemini"][i])
        variation_names.append(f"postfix_gemini_base{i+1}")
    
    # Then for each prefix (gemini only)
    for i, prompt in enumerate(base_prompts):
        variations.append(prefixes["gemini"][i] + prompt)
        variation_names.append(f"prefix_gemini_base{i+1}")
    
    return variations, variation_names

def test_animatediff_with_lora():
    """
    Test AnimateDiff pipeline with DreamBooth LoRA applied
    """
    
    # Hardcoded paths - adjust these for your setup
    NAME = "sarah"
    LORA_MODEL_PATH = f"/graphics/scratch2/students/webereli/test/models/{NAME}"
    OUTPUT_DIR = "/graphics/scratch2/students/webereli/playground/post_and_prefixes/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate all variations
    test_prompts, variation_names = generate_variations(NAME)
    
    print("=== PROMPT VARIATIONS ===")
    for i, (prompt, name) in enumerate(zip(test_prompts, variation_names)):
        print(f"{i+1:2d}: {name}")
        print(f"    {prompt}")
    print(f"\nTotal variations: {len(test_prompts)}")
    
    print("=== ANIMATEDIFF + LORA TEST ===")
    print(f"LoRA Model Path: {LORA_MODEL_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if LoRA model exists
    if not os.path.exists(LORA_MODEL_PATH):
        print(f"Error: LoRA model path does not exist: {LORA_MODEL_PATH}")
        return

    # Generate test videos - LOAD FRESH PIPELINE FOR EACH TEST
    for i, (prompt, variation_name) in enumerate(zip(test_prompts, variation_names)):
        print(f"\n=== Test {i+1}/{len(test_prompts)}: {variation_name} ===")
        print(f"Prompt: {prompt}")
        
        try:
            print("--- Loading Fresh AnimateDiff Pipeline ---")
            
            # Force cleanup before loading new pipeline
            torch.cuda.empty_cache()
            
            # Load AnimateDiff pipeline
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2", 
                torch_dtype=torch.float16
            )
            
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            
            pipe = AnimateDiffPipeline.from_pretrained(
                model_id, 
                motion_adapter=adapter, 
                torch_dtype=torch.float16,
            ).to(DEVICE)
            
            # Configure scheduler
            scheduler = DDIMScheduler.from_pretrained(
                model_id,
                subfolder="scheduler",
                clip_sample=False,
                timestep_spacing="linspace",
                beta_schedule="linear",
                steps_offset=1,
            )
            pipe.scheduler = scheduler
            
            print("--- Applying LoRA ---")
            
            # Apply LoRA with better error handling
            unet_path = os.path.join(LORA_MODEL_PATH, "unet")
            text_encoder_path = os.path.join(LORA_MODEL_PATH, "text_encoder")
            
            if os.path.exists(unet_path):
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="peft")
                    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_path)
                print("  UNet LoRA applied")
            else:
                print(f"Error: UNet LoRA path not found: {unet_path}")
                continue
            
            if os.path.exists(text_encoder_path):
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="peft")
                    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_path)
                print("  Text Encoder LoRA applied")
            else:
                print(f"Error: Text Encoder LoRA path not found: {text_encoder_path}")
                continue
            
            # enable FreeInit
            # Refer to the enable_free_init documentation for a full list of configurable parameters
            pipe.enable_free_init(method="butterworth", use_fast_sampling=True)
            # Enable optimizations
            pipe.enable_vae_slicing()
            pipe.enable_model_cpu_offload()

            # Fixed seed for each generation
            seed = 2169387807
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            
            print("--- Generating Video ---")
            
            # Generate video
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    negative_prompt="bad quality, worse quality, blurry",
                    num_frames=16,
                    height=512,
                    width=512,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=generator,
                )
            
            frames = result.frames[0]
            print(f"Generated {len(frames)} frames")

            # disable FreeInit
            pipe.disable_free_init()
            
            # Save frames as individual images
            test_dir = os.path.join(OUTPUT_DIR, variation_name)
            os.makedirs(test_dir, exist_ok=True)
            
            for frame_idx, frame in enumerate(frames):
                frame_path = os.path.join(test_dir, f"frame_{frame_idx:03d}.png")
                frame.save(frame_path)
            
            print(f"Frames saved to: {test_dir}")
            
            # Convert frames to video
            video_path = os.path.join(OUTPUT_DIR, f"{variation_name}.mp4")
            frames_to_video(test_dir, video_path, fps=8)
            
        except Exception as e:
            print(f"Error during test {i+1}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # CRITICAL: Clean up pipeline after each generation
            print("--- Cleaning up pipeline ---")
            try:
                del pipe
                del adapter
                del scheduler
                torch.cuda.empty_cache()
                print("  Pipeline cleaned")
            except:
                print("  Error during cleanup")
    
    print("\n=== All Tests Completed ===")
    print(f"Videos generated in: {OUTPUT_DIR}")
    print("Generated files:")
    for variation_name in variation_names:
        video_file = os.path.join(OUTPUT_DIR, f"{variation_name}.mp4")
        if os.path.exists(video_file):
            print(f"  {variation_name}.mp4")
        else:
            print(f"  {variation_name}.mp4 (conversion failed)")

if __name__ == "__main__":
    test_animatediff_with_lora()