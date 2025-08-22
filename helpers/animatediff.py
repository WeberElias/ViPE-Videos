import os
import torch
import numpy as np
from PIL import Image
import cv2
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from peft import PeftModel

def generate_animatediff_video(args, anim_args, animation_prompts, root, characters=None):
    """
    Generate video using AnimateDiff with one video segment per line/prompt.
    Each line/prompt gets its own complete video.
    All videos are concatenated to form the final video.
    
    Args:
        args: Generation arguments (from DeforumArgs)
        anim_args: Animation arguments (from DeforumAnimArgs) 
        animation_prompts: Dictionary of {frame_num: prompt_data}
        root: Root object containing model and device info
        characters: List of all Character objects
        
    Returns:
        str: Path to the generated video file
    """
    
    # Configuration - use AnimateDiff-compatible frame counts
    MAX_FRAMES_PER_SEGMENT = 64  # AnimateDiff max
    MIN_FRAMES_PER_SEGMENT = 16  # AnimateDiff min (changed from 8)
    
    print("=== ANIMATEDIFF VIDEO GENERATION ===")
    print(f"Total lines/prompts to process: {len(animation_prompts)}")
    
    # Get all unique lines/prompts - each will become one video segment
    lines = _extract_lines_from_animation_prompts(animation_prompts, anim_args.max_frames)
    
    print(f"Found {len(lines)} unique lines/prompts:")
    for i, line in enumerate(lines):
        print(f"  Line {i+1}: '{line['prompt'][:50]}...' (duration: {line['duration']} frames)")
    
    # Generate one video segment for each line/prompt
    all_video_segments = []
    
    for line_idx, line in enumerate(lines):
        print(f"\n--- GENERATING VIDEO FOR LINE {line_idx + 1}/{len(lines)} ---")
        print(f"Prompt: {line['prompt']}")
        print(f"Target duration: {line['duration']} frames")
        
        # Calculate how many frames to generate for this video segment
        # Use the line duration but cap it for memory/quality reasons
        frames_to_generate = min(line['duration'], MAX_FRAMES_PER_SEGMENT)
        frames_to_generate = max(frames_to_generate, MIN_FRAMES_PER_SEGMENT)
        
        print(f"Generating {frames_to_generate} frames for this video segment...")
        
        try:
            # Generate complete video for this line/prompt
            video_frames = _generate_video_for_line(
                prompt=line['prompt'],
                num_frames=frames_to_generate,
                characters=characters,  # Pass all characters instead of line-specific ones
                args=args,
                root=root,
                line_number=line_idx  # Add the line number parameter
            )
            
            if video_frames and len(video_frames) > 0:
                print(f"Successfully generated {len(video_frames)} frames for line {line_idx + 1}")
                
                # Store this complete video segment
                video_segment = {
                    'line_number': line_idx + 1,
                    'prompt': line['prompt'],
                    'frames': video_frames,
                    'target_duration': line['duration'],
                    'generated_frames': len(video_frames)
                }
                all_video_segments.append(video_segment)
                
            else:
                print(f"ERROR: No frames generated for line {line_idx + 1}")
                return None
                
        except Exception as e:
            print(f"ERROR generating video for line {line_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    if not all_video_segments:
        print("ERROR: No video segments were generated!")
        return None
    
    # Now stitch all video segments together
    print(f"\n--- STITCHING {len(all_video_segments)} VIDEO SEGMENTS TOGETHER ---")
    final_video_frames = _stitch_video_segments_together(all_video_segments)
    
    if not final_video_frames:
        print("ERROR: Failed to stitch video segments together!")
        return None
    
    # Save the final concatenated video to disk
    print(f"\n--- SAVING {len(final_video_frames)} FRAMES TO DISK ---")
    saved_frame_count = _save_frames_to_disk(final_video_frames, args)
    
    if saved_frame_count > 0:
        print(f"SUCCESS: Saved {saved_frame_count} frames to {args.outdir}")
        print(f"Frames saved with pattern: {args.timestring}_*.png")
        return args.outdir
    else:
        print("ERROR: Failed to save frames to disk")
        return None


def _extract_lines_from_animation_prompts(animation_prompts, max_frames):
    """
    Extract individual lines/prompts from the animation_prompts dictionary.
    Each unique prompt becomes one line that will get its own video.
    
    Args:
        animation_prompts: Dictionary of {frame_num: prompt_data}
        max_frames: Total frames in the animation
        
    Returns:
        List of line dictionaries with prompt, duration, characters
    """
    import pandas as pd
    
    # Create frame-by-frame prompt series to understand timing
    prompt_series = pd.Series([np.nan for _ in range(max_frames)])
    for frame_num, prompt_data in animation_prompts.items():
        prompt_series[int(frame_num)] = prompt_data
    prompt_series = prompt_series.ffill().bfill()
    
    # Extract unique lines and their durations
    lines = []
    current_prompt = None
    current_characters = None
    start_frame = 0
    
    for frame_idx in range(max_frames):
        frame_prompt_data = prompt_series[frame_idx]
        
        # Extract prompt text and characters
        if isinstance(frame_prompt_data, dict):
            prompt_text = frame_prompt_data.get('prompt', str(frame_prompt_data))
            characters = frame_prompt_data.get('characters', [])
        else:
            prompt_text = str(frame_prompt_data)
            characters = []
        
        # Check if we moved to a new line/prompt
        if current_prompt is not None and prompt_text != current_prompt:
            # Save the previous line
            lines.append({
                'prompt': current_prompt,
                'characters': current_characters,
                'start_frame': start_frame,
                'end_frame': frame_idx,
                'duration': frame_idx - start_frame
            })
            start_frame = frame_idx
        
        current_prompt = prompt_text
        current_characters = characters
    
    # Add the final line
    if current_prompt is not None:
        lines.append({
            'prompt': current_prompt,
            'characters': current_characters,
            'start_frame': start_frame,
            'end_frame': max_frames,
            'duration': max_frames - start_frame
        })
    
    return lines


def _generate_video_for_line(prompt, num_frames, characters, args, root, line_number=None):
    """
    Generate a complete video for one line/prompt using AnimateDiff.
    
    Args:
        prompt: Text prompt for this line
        num_frames: Number of frames to generate
        characters: List of Character objects (all characters)
        args: Generation arguments
        root: Root object with model and device
        line_number: The line number being processed (0-indexed)
        
    Returns:
        List of PIL Images or None if generation failed
    """
    
    try:
        print(f"Loading AnimateDiff pipeline for: '{prompt[:50]}...'")
        
        # CONSISTENT FRAME COUNT APPROACH:
        # Always use the same frame count to avoid LoRA conflicts
        # We'll adjust the final video length by duplicating/trimming frames
        CONSISTENT_FRAME_COUNT = 32  # Use 32 for all generations
        
        original_num_frames = num_frames
        num_frames = CONSISTENT_FRAME_COUNT
        
        print(f"Using consistent frame count: {num_frames} (target: {original_num_frames})")
        
        # Change this back to variable frame counts
        # AnimateDiff works best with specific frame counts
        # Common working values: 16, 24, 32, 48, 64
        valid_frame_counts = [16, 24, 32, 48, 64]

        # Find the closest valid frame count
        original_num_frames = num_frames
        num_frames = min(valid_frame_counts, key=lambda x: abs(x - num_frames))

        if original_num_frames != num_frames:
            print(f"Adjusted frame count from {original_num_frames} to {num_frames} for AnimateDiff compatibility")
        
        # Force GPU memory cleanup before loading new pipeline
        torch.cuda.empty_cache()
        
        # Load fresh AnimateDiff pipeline for each line to avoid LoRA conflicts
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2", 
            torch_dtype=torch.float16
        )
        
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        
        pipe = AnimateDiffPipeline.from_pretrained(
            model_id, 
            motion_adapter=adapter, 
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(root.device)
        
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
        
        # Apply LoRA for characters that appear in this line
        if characters and line_number is not None:
            # Filter characters that appear in this specific line
            line_characters = [char for char in characters if char.appears_in_line(line_number)]
            if line_characters:
                _apply_character_loras(pipe, line_characters)
        
        # Enable memory optimizations
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        
        # Generation parameters
        seed = getattr(args, 'seed', 42)
        generator = torch.Generator(device=root.device).manual_seed(seed)
        
        negative_prompt = getattr(args, 'negative_prompt', "bad quality, worse quality, low resolution, blurry")
        height = getattr(args, 'H', 512)
        width = getattr(args, 'W', 512) 
        steps = getattr(args, 'steps', 25)
        guidance_scale = getattr(args, 'scale', 7.5)
        
        # Ensure dimensions are divisible by 8 (required by diffusion models)
        height = height - (height % 8)
        width = width - (width % 8)
        
        print(f"Generation parameters:")
        print(f"  Frames: {num_frames} (target: {original_num_frames})")
        print(f"  Size: {width}x{height}")
        print(f"  Steps: {steps}")
        print(f"  Guidance: {guidance_scale}")
        print(f"  Seed: {seed}")
        
        # Generate complete video for this line/prompt
        print("Generating video...")
        
        print("DEBUG ----------------------------"+ prompt)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        
        frames = result.frames[0]
        print(f"Generated {len(frames)} frames successfully")
        
        # Adjust frame count to match target duration
        if len(frames) != original_num_frames:
            if len(frames) < original_num_frames:
                print(f"Extending video from {len(frames)} to {original_num_frames} frames")
                last_frame = frames[-1]
                while len(frames) < original_num_frames:
                    frames.append(last_frame.copy())
            else:
                print(f"Trimming video from {len(frames)} to {original_num_frames} frames")
                frames = frames[:original_num_frames]
        
        # Clean up pipeline completely to free memory for next line
        del pipe
        del adapter
        del scheduler
        torch.cuda.empty_cache()
        
        return frames
        
    except Exception as e:
        print(f"Error generating video for line: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            del pipe
            del adapter
            torch.cuda.empty_cache()
        except:
            pass
        
        return None


def _stitch_video_segments_together(video_segments):
    """
    Stitch all video segments together into one final video.
    Each segment represents one line/prompt.
    
    Args:
        video_segments: List of video segment dictionaries
        
    Returns:
        List of PIL Images representing the final concatenated video
    """
    
    print("Stitching video segments together...")
    final_frames = []
    
    for segment in video_segments:
        line_number = segment['line_number']
        frames = segment['frames']
        target_duration = segment['target_duration']
        generated_frames = segment['generated_frames']
        
        print(f"Adding video for line {line_number}: {generated_frames} frames")
        
        # Add all frames from this video segment
        final_frames.extend(frames)
        
        # If the generated video is shorter than the target duration,
        # extend it by repeating the last frame
        if generated_frames < target_duration:
            frames_to_extend = target_duration - generated_frames
            last_frame = frames[-1]
            
            print(f"  Extending line {line_number} by {frames_to_extend} frames")
            for _ in range(frames_to_extend):
                final_frames.append(last_frame)
    
    print(f"Final video: {len(final_frames)} total frames from {len(video_segments)} lines")
    return final_frames


def _apply_character_loras(pipe, characters):
    """
    Apply LoRA models for characters to the pipeline.
    
    Args:
        pipe: AnimateDiff pipeline
        characters: List of Character objects that should appear in this line
    """
    
    try:
        # Filter characters that have trained models
        characters_with_models = [
            char for char in characters 
            if hasattr(char, 'model_path') and char.model_path and os.path.exists(char.model_path)
        ]
        
        if not characters_with_models:
            print("No characters with trained models found for this line")
            return
        
        print(f"Applying LoRA for characters: {[char.name for char in characters_with_models]}")
        
        # Only apply the first character's LoRA to avoid conflicts
        character = characters_with_models[0]
        base_model_path = character.model_path
        unet_path = os.path.join(base_model_path, "unet")
        text_encoder_path = os.path.join(base_model_path, "text_encoder")
        
        try:
            print(f"  Applying UNet LoRA for {character.name} from {unet_path}")
            if os.path.exists(unet_path):
                pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_path)
            else:
                print(f"    Warning: UNet LoRA path does not exist: {unet_path}")
                return
            
            print(f"  Applying Text Encoder LoRA for {character.name} from {text_encoder_path}")
            if os.path.exists(text_encoder_path):
                pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_path)
            else:
                print(f"    Warning: Text Encoder LoRA path does not exist: {text_encoder_path}")
                return
            
            print(f"  Successfully applied LoRA for {character.name}")
            
        except Exception as e:
            print(f"  Error applying LoRA for {character.name}: {e}")
            # Don't raise the exception, just continue without LoRA
            
    except Exception as e:
        print(f"Error applying character LoRAs: {e}")


def _save_frames_to_disk(frames, args):
    """
    Save all video frames to disk with proper naming for FFmpeg.
    
    Args:
        frames: List of PIL Images
        args: Generation arguments containing outdir, timestring, bit_depth_output
        
    Returns:
        int: Number of frames successfully saved
    """
    
    try:
        # Ensure output directory exists
        os.makedirs(args.outdir, exist_ok=True)
        
        saved_count = 0
        
        for frame_idx, frame_image in enumerate(frames):
            filename = f"{args.timestring}_{frame_idx:05}.png"
            
            try:
                # Convert frame based on bit depth
                bit_depth = getattr(args, 'bit_depth_output', 8)
                
                if bit_depth == 8:
                    # PIL Image for 8-bit PNG
                    frame_image.save(os.path.join(args.outdir, filename))
                elif bit_depth == 32:
                    # Convert to float32 array and save as EXR
                    frame_array = np.array(frame_image).astype(np.float32) / 255.0
                    filename = filename.replace(".png", ".exr")
                    cv2.imwrite(
                        os.path.join(args.outdir, filename), 
                        cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    )
                else:
                    # 16-bit PNG
                    frame_array = (np.array(frame_image) * 256).astype(np.uint16)
                    try:
                        from numpngw import write_png
                        write_png(os.path.join(args.outdir, filename), frame_array)
                    except ImportError:
                        print("numpngw not available, falling back to 8-bit")
                        frame_image.save(os.path.join(args.outdir, filename))
                
                saved_count += 1
                
                # Display progress occasionally
                if frame_idx % 10 == 0:
                    print(f"  Saved frame {frame_idx}: {filename}")
                    
            except Exception as e:
                print(f"Error saving frame {frame_idx}: {e}")
                continue
        
        return saved_count
        
    except Exception as e:
        print(f"Error in save_frames_to_disk: {e}")
        return 0