import torch
from PIL import Image
import requests
import numpy as np
import torchvision.transforms.functional as TF
from pytorch_lightning import seed_everything
import os
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser
from torch import autocast
from contextlib import nullcontext
from einops import rearrange, repeat

from diffusers import StableDiffusionPipeline, AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from peft import PeftModel

from .prompt import get_uc_and_c
from .k_samplers import sampler_fn, make_inject_timing_fn
from scipy.ndimage import gaussian_filter

from .callback import SamplerCallback

from .conditioning import exposure_loss, make_mse_loss, get_color_palette, make_clip_loss_fn
from .conditioning import make_rgb_color_match_loss, blue_loss_fn, threshold_by, make_aesthetics_loss_fn, mean_loss_fn, var_loss_fn, exposure_loss
from .model_wrap import CFGDenoiserWithGrad
from .load_images import load_img, load_mask_latent, prepare_mask, prepare_overlay_mask

def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def generate_motion_adapter_pipeline(args, root, frame=0, return_latent=False, return_sample=False, return_c=False):
    """
    MotionAdapter-based video generation pipeline for coherent frame sequences
    """
    
    # Ensure frame is an integer
    if not isinstance(frame, int):
        frame = 0
        print(f"Warning: frame parameter was not an integer, defaulting to 0")
    
    print("DEBUG 3 ------------------------------------------------------- \n -----------------------------------------------")
    print("Sequence for prompt: " + args.prompt)
    
    # Calculate the duration for this specific prompt based on animation_prompts
    prompt_start_frame = frame
    prompt_end_frame = None
    
    # Find this prompt's duration by looking at animation_prompts
    if hasattr(args, 'prompts') and args.prompts:
        # Get sorted frame numbers
        frame_numbers = sorted([int(k) for k in args.prompts.keys()])
        
        # Find current prompt's position
        current_prompt_frame = None
        for f_num in frame_numbers:
            prompt_data = args.prompts[f_num]
            if isinstance(prompt_data, dict):
                prompt_text = prompt_data.get('prompt', prompt_data)
            else:
                prompt_text = prompt_data
            
            if prompt_text == args.prompt:
                current_prompt_frame = f_num
                break
        
        if current_prompt_frame is not None:
            # Find the next prompt's frame number to determine duration
            current_index = frame_numbers.index(current_prompt_frame)
            if current_index + 1 < len(frame_numbers):
                prompt_end_frame = frame_numbers[current_index + 1]
            else:
                # This is the last prompt, use max_frames
                if hasattr(args, 'max_frames'):
                    prompt_end_frame = args.max_frames
                else:
                    prompt_end_frame = current_prompt_frame + 32  # default fallback
            
            prompt_start_frame = current_prompt_frame
    
    # Calculate actual number of frames for this prompt
    if prompt_end_frame is not None:
        num_frames = min(prompt_end_frame - prompt_start_frame, 64)  # Cap at 64 for MotionAdapter
    else:
        print("-------------------------------------- FALL BACK FOR PROMPT DURATION")
        num_frames = getattr(args, 'num_frames', 32)  # Default fallback
    
    # Ensure minimum frames
    num_frames = max(num_frames, 8)  # Minimum 8 frames
    
    print(f"Prompt duration: frames {prompt_start_frame} to {prompt_end_frame} ({num_frames} frames)")
    
    cache_key = f"{args.prompt}_{args.seed}_{num_frames}"
    
    # If frames are already cached, return ALL frames (for saving to disk)
    if hasattr(root, 'motion_adapter_frames') and cache_key in root.motion_adapter_frames:
        results = root.motion_adapter_frames[cache_key]
        print(f"Found cached frames: {len(results)} frames")
        
        # For MotionAdapter, we want to return all frames when called from render_animation
        if frame == 0:
            return results  # Return all frames for saving
        
        # Return specific frame if requested
        if frame < len(results):
            current_frame = results[frame]
        else:
            current_frame = results[-1]
        
        # Return format that matches what render.py expects
        if return_sample and return_latent and return_c:
            return [None, None, None, current_frame]
        elif return_sample and return_latent:
            return [None, None, current_frame]
        elif return_sample:
            return None, current_frame
        elif return_latent:
            return [None, current_frame]
        elif return_c:
            return [None, current_frame]
        else:
            return [current_frame]
    
    # Generate video only on first call (frame 0)
    if frame != 0:
        print(f"Warning: Motion adapter already should have generated all frames. Returning empty result for frame {frame}")
        return [None]
    
    print(f"Generating video with MotionAdapter for prompt: {args.prompt}")
    print(f"Generating {num_frames} frames for this prompt segment...")
    
    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2", 
        torch_dtype=torch.float16
    )
    
    # Use SD 1.5 based model (you can change this to match your needs)
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"  # or "runwayml/stable-diffusion-v1-5"
    
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id, 
        motion_adapter=adapter, 
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(root.device)
    
    # Configure scheduler for better quality
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler
    
    # Apply LoRA if available
    if hasattr(root, 'lora_manager') and hasattr(root, 'characters'):
        current_characters = []
        for character in root.characters:
            if character.unique_identifier.lower() in args.prompt.lower():
                current_characters.append(character)
        
        if current_characters:
            for character in current_characters:
                if hasattr(root.lora_manager, 'lora_cache') and character.name in root.lora_manager.lora_cache:
                    lora_data = root.lora_manager.lora_cache[character.name]
                    unet_dir = lora_data['unet_dir']
                    text_encoder_dir = lora_data['text_encoder_dir']
                    
                    # Apply UNet LoRA
                    if os.path.exists(unet_dir):
                        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_dir)
                    
                    # Apply Text Encoder LoRA
                    if os.path.exists(text_encoder_dir):
                        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_dir)
                    
                    break
    
    # Enable memory optimizations
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    
    # Set up generator for reproducibility
    generator = torch.Generator(device=root.device).manual_seed(args.seed) if hasattr(args, 'seed') else None
    
    # Create negative prompt if not provided
    negative_prompt = getattr(args, 'negative_prompt', "bad quality, worse quality, low resolution, blurry")
    
    # Generate video
    with torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=args.H,
            width=args.W,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            generator=generator,
        )
    
    frames = result.frames[0]  # Get the first (and only) video sequence
    
    # Convert frames to the expected format
    results = []
    
    for i, frame_img in enumerate(frames):
        # Convert PIL to format expected by the rest of the system
        if args.bit_depth_output == 8:
            processed_frame = frame_img  # PIL Image for 8-bit
        elif args.bit_depth_output == 32:
            frame_array = np.array(frame_img).astype(np.float32) / 255.0
            processed_frame = frame_array
        else:  # 16-bit
            frame_array = (np.array(frame_img) * 256).astype(np.uint16)
            processed_frame = frame_array
        
        results.append(processed_frame)
    
    # Store frames in a way that render.py can access them for animation
    if not hasattr(root, 'motion_adapter_frames'):
        root.motion_adapter_frames = {}
    
    # Store frames with a key based on prompt (simple cache)
    root.motion_adapter_frames[cache_key] = results
    root.motion_adapter_current_frame = frame
    root.motion_adapter_total_frames = len(results)
    
    print(f"Generated and cached {len(results)} frames")
    
    # Return all frames for saving to disk
    return results

def generate_simple_pipeline(args, root, frame=0, return_latent=False, return_sample=False, return_c=False):
    """
    Simple diffusers-based generation pipeline
    """

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(root.device)
    
    if hasattr(root, 'lora_manager') and hasattr(root, 'characters'):
        # Get current characters from the prompt
        current_characters = []
        for character in root.characters:
            if character.unique_identifier.lower() in args.prompt.lower():
                current_characters.append(character)
        
        if current_characters:
            for character in current_characters:
                if hasattr(root.lora_manager, 'lora_cache') and character.name in root.lora_manager.lora_cache:
                    lora_data = root.lora_manager.lora_cache[character.name]
                    unet_dir = lora_data['unet_dir']
                    text_encoder_dir = lora_data['text_encoder_dir']
                    
                    # Apply UNet LoRA
                    if os.path.exists(unet_dir):
                        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_dir)
                    
                    # Apply Text Encoder LoRA
                    if os.path.exists(text_encoder_dir):
                        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_dir)
                    
                    break
    
    # Set progress bar
    pipe.set_progress_bar_config(disable=True)
    
    # Generate image
    with torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            height=args.H,
            width=args.W,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            num_images_per_prompt=args.n_samples
        )
    
    # Convert results to match original format expectations
    results = []
    
    for image in result.images:
        # Convert PIL to format expected by the rest of the system
        if args.bit_depth_output == 8:
            processed_image = image  # PIL Image for 8-bit
        elif args.bit_depth_output == 32:
            image_array = np.array(image).astype(np.float32) / 255.0
            processed_image = image_array
        else:  # 16-bit
            image_array = (np.array(image) * 256).astype(np.uint16)
            processed_image = image_array
        
        results.append(processed_image)
    
    # Return format that matches what render.py expects
    if return_sample and return_latent and return_c:
        # Return [latent, sample, conditioning, image1, image2, ...]
        return [None, None, None] + results  # Dummy values for latent, sample, c
    elif return_sample and return_latent:
        # Return [latent, sample, image1, image2, ...]
        return [None, None] + results
    elif return_sample:
        # This is what render_animation expects: sample, image
        return None, results[0] if results else None
    elif return_latent:
        # Return [latent, image1, image2, ...]
        return [None] + results
    elif return_c:
        # Return [conditioning, image1, image2, ...]
        return [None] + results
    else:
        # Just return the images
        return results
        

def generate_original_pipeline(args, root, frame=0, return_latent=False, return_sample=False, return_c=False):
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    sampler = PLMSSampler(root.model) if args.sampler == 'plms' else DDIMSampler(root.model)
    if root.model.parameterization == "v":
        model_wrap = CompVisVDenoiser(root.model)
    else:
        model_wrap = CompVisDenoiser(root.model)
    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        with precision_scope("cuda"):
            init_latent = root.model.get_first_stage_encoding(root.model.encode_first_stage(args.init_sample))
    elif args.use_init and args.init_image != None and args.init_image != '':
        init_image, mask_image = load_img(args.init_image, 
                                          shape=(args.W, args.H),  
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_image = init_image.to(root.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            init_latent = root.model.get_first_stage_encoding(root.model.encode_first_stage(init_image))  # move to latent space        

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        #print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        #print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    # Mask functions
    if args.use_mask:
        assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
        assert args.use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"


        mask = prepare_mask(args.mask_file if mask_image is None else mask_image, 
                            init_latent.shape, 
                            args.mask_contrast_adjust, 
                            args.mask_brightness_adjust,
                            args.invert_mask)
        
        if (torch.all(mask == 0) or torch.all(mask == 1)) and args.use_alpha_as_mask:
            raise Warning("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")
        
        mask = mask.to(root.device)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    else:
        mask = None

    assert not ( (args.use_mask and args.overlay_mask) and (args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

    # Init MSE loss image
    init_mse_image = None
    if args.init_mse_scale and args.init_mse_image != None and args.init_mse_image != '':
        init_mse_image, mask_image = load_img(args.init_mse_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_mse_image = init_mse_image.to(root.device)
        init_mse_image = repeat(init_mse_image, '1 ... -> b ...', b=batch_size)

    assert not ( args.init_mse_scale != 0 and (args.init_mse_image is None or args.init_mse_image == '') ), "Need an init image when init_mse_scale != 0"

    t_enc = int((1.0-args.strength) * args.steps)

    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(args.steps)
    args.clamp_schedule = dict(zip(k_sigmas.tolist(), np.linspace(args.clamp_start,args.clamp_stop,args.steps+1)))
    k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

    if args.sampler in ['plms','ddim']:
        sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, ddim_discretize='uniform', verbose=False)

    if args.colormatch_scale != 0:
        assert args.colormatch_image is not None, "If using color match loss, colormatch_image is needed"
        colormatch_image, _ = load_img(args.colormatch_image)
        colormatch_image = colormatch_image.to('cpu')
        del(_)
    else:
        colormatch_image = None

    # Loss functions
    if args.init_mse_scale != 0:
        if args.decode_method == "linear":
            mse_loss_fn = make_mse_loss(root.model.linear_decode(root.model.get_first_stage_encoding(root.model.encode_first_stage(init_mse_image.to(root.device)))))
        else:
            mse_loss_fn = make_mse_loss(init_mse_image)
    else:
        mse_loss_fn = None

    if args.colormatch_scale != 0:
        _,_ = get_color_palette(root, args.colormatch_n_colors, colormatch_image, verbose=True) # display target color palette outside the latent space
        if args.decode_method == "linear":
            grad_img_shape = (int(args.W/args.f), int(args.H/args.f))
            colormatch_image = root.model.linear_decode(root.model.get_first_stage_encoding(root.model.encode_first_stage(colormatch_image.to(root.device))))
            colormatch_image = colormatch_image.to('cpu')
        else:
            grad_img_shape = (args.W, args.H)
        color_loss_fn = make_rgb_color_match_loss(root,
                                                  colormatch_image, 
                                                  n_colors=args.colormatch_n_colors, 
                                                  img_shape=grad_img_shape,
                                                  ignore_sat_weight=args.ignore_sat_weight)
    else:
        color_loss_fn = None

    if args.clip_scale != 0:
        clip_loss_fn = make_clip_loss_fn(root, args)
    else:
        clip_loss_fn = None

    if args.aesthetics_scale != 0:
        aesthetics_loss_fn = make_aesthetics_loss_fn(root, args)
    else:
        aesthetics_loss_fn = None

    if args.exposure_scale != 0:
        exposure_loss_fn = exposure_loss(args.exposure_target)
    else:
        exposure_loss_fn = None

    loss_fns_scales = [
        [clip_loss_fn,              args.clip_scale],
        [blue_loss_fn,              args.blue_scale],
        [mean_loss_fn,              args.mean_scale],
        [exposure_loss_fn,          args.exposure_scale],
        [var_loss_fn,               args.var_scale],
        [mse_loss_fn,               args.init_mse_scale],
        [color_loss_fn,             args.colormatch_scale],
        [aesthetics_loss_fn,        args.aesthetics_scale]
    ]

    # Conditioning gradients not implemented for ddim or PLMS
    assert not( any([cond_fs[1]!=0 for cond_fs in loss_fns_scales]) and (args.sampler in ["ddim","plms"]) ), "Conditioning gradients not implemented for ddim or plms. Please use a different sampler."

    callback = SamplerCallback(args=args,
                            root=root,
                            mask=mask, 
                            init_latent=init_latent,
                            sigmas=k_sigmas,
                            sampler=sampler,
                            verbose=False).callback 

    clamp_fn = threshold_by(threshold=args.clamp_grad_threshold, threshold_type=args.grad_threshold_type, clamp_schedule=args.clamp_schedule)

    grad_inject_timing_fn = make_inject_timing_fn(args.grad_inject_timing, model_wrap, args.steps)

    cfg_model = CFGDenoiserWithGrad(model_wrap, 
                                    loss_fns_scales, 
                                    clamp_fn, 
                                    args.gradient_wrt, 
                                    args.gradient_add_to, 
                                    args.cond_uncond_sync,
                                    decode_method=args.decode_method,
                                    grad_inject_timing_fn=grad_inject_timing_fn, # option to use grad in only a few of the steps
                                    grad_consolidate_fn=None, # function to add grad to image fn(img, grad, sigma)
                                    verbose=False)

    results = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with root.model.ema_scope():
                for prompts in data:
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    if args.prompt_weighting:
                        uc, c = get_uc_and_c(prompts, root.model, args, frame)
                    else:
                        uc = root.model.get_learned_conditioning(batch_size * [""])
                        c = root.model.get_learned_conditioning(prompts)


                    if args.scale == 1.0:
                        uc = None
                    if args.init_c != None:
                        c = args.init_c

                    if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]:
                        samples = sampler_fn(
                            c=c, 
                            uc=uc, 
                            args=args, 
                            model_wrap=cfg_model, 
                            init_latent=init_latent, 
                            t_enc=t_enc, 
                            device=root.device, 
                            cb=callback,
                            verbose=False)
                    else:
                        # args.sampler == 'plms' or args.sampler == 'ddim':
                        if init_latent is not None and args.strength > 0:
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(root.device))
                        else:
                            z_enc = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=root.device)
                        if args.sampler in ['plms','ddim']: # no "decode" function in plms, so use "sample"
                            shape = [args.C, args.H // args.f, args.W // args.f]
                            samples, _ = sampler.sample(S=args.steps,
                                                            conditioning=c,
                                                            batch_size=args.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=args.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=args.ddim_eta,
                                                            x_T=z_enc,
                                                            img_callback=callback)
                        else:
                            raise Exception(f"Sampler {args.sampler} not recognised.")

                    
                    if return_latent:
                        results.append(samples.clone())

                    x_samples = root.model.decode_first_stage(samples)

                    if args.use_mask and args.overlay_mask:
                        # Overlay the masked image after the image is generated
                        if args.init_sample_raw is not None:
                            img_original = args.init_sample_raw
                        elif init_image is not None:
                            img_original = init_image
                        else:
                            raise Exception("Cannot overlay the masked image without an init image to overlay")

                        if args.mask_sample is None or args.using_vid_init:
                            args.mask_sample = prepare_overlay_mask(args, root, img_original.shape)

                        x_samples = img_original * args.mask_sample + x_samples * ((args.mask_sample * -1.0) + 1)

                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        def uint_number(datum, number):
                            if number == 8:
                                datum = Image.fromarray(datum.astype(np.uint8))
                            elif number == 32:
                                datum = datum.astype(np.float32)
                            else:
                                datum = datum.astype(np.uint16)
                            return datum
                        if args.bit_depth_output == 8:
                            exponent_for_rearrange = 1
                        elif args.bit_depth_output == 32:
                            exponent_for_rearrange = 0
                        else:
                            exponent_for_rearrange = 2
                        x_sample = 255.**exponent_for_rearrange * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = uint_number(x_sample, args.bit_depth_output)
                        results.append(image)
    return results

def generate(args, root, frame=0, return_latent=False, return_sample=False, return_c=False):
    """
    Main generate function - toggle between pipelines here
    """
    
    # TOGGLE BETWEEN PIPELINES BY COMMENTING/UNCOMMENTING THESE LINES:
    
    # Use MotionAdapter pipeline for coherent video generation
    print("\n __________ Using MotionAdapter pipeline _____________")
    return generate_motion_adapter_pipeline(args, root, frame, return_latent, return_sample, return_c)
    
    # Use simple diffusers pipeline
    #print("\n __________ Using simple pipeline _____________")
    #return generate_simple_pipeline(args, root, frame, return_latent, return_sample, return_c)
    
    # Use original complex LDM pipeline  
    #print("\n __________ Using original pipeline _____________")
    #return generate_original_pipeline(args, root, frame, return_latent, return_sample, return_c)
