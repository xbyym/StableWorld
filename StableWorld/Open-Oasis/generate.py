"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import os
import argparse
from pprint import pprint

import cv2
import imageio
import numpy as np
import torch
from torch import autocast
from tqdm import tqdm
from einops import rearrange
from safetensors.torch import load_model
from torchvision.io import write_video

from dit import DiT_models
from vae import VAE_models
from utils import load_prompt, load_actions, sigmoid_beta_schedule

assert torch.cuda.is_available()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y", "t"):
        return True
    if v.lower() in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def chw_to_gray_u8(chw: torch.Tensor) -> np.ndarray:
    """
    chw: [C, H, W] tensor, with value range either [-1, 1] or [0, 1]
    Return: HxW uint8 grayscale image
    """
    x = chw.detach().float()
    x_min, x_max = x.min().item(), x.max().item()

    if x_min >= -1.0 and x_max <= 1.0:
        x = ((x + 1.0) * 127.5).clamp(0, 255.0)
    else:
        x = (x * 255.0).clamp(0, 255.0)

    if x.shape[0] == 3:
        r, g, b = x[0], x[1], x[2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 255)
    else:
        gray = x[0].clamp(0, 255)

    return gray.byte().cpu().numpy()


def orb_ransac_score_chw(
    chwA: torch.Tensor,
    chwB: torch.Tensor,
    ratio_thresh: float = 0.8,
    min_good: int = 30,
    nfeatures: int = 3000,
) -> float:
    """
    ORB + RANSAC inlier ratio, with similarity roughly in [0, 1]
    """
    img1 = chw_to_gray_u8(chwA)
    img2 = chw_to_gray_u8(chwB)

    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)

    if d1 is None or d2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 5:
        return 0.0

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])

    H, maskH = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    F, maskF = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

    inH = int(maskH.sum()) if maskH is not None else 0
    inF = int(maskF.sum()) if isinstance(maskF, np.ndarray) and maskF.size > 0 else 0

    denom = max(len(good), 1)
    score = max(inH / denom, inF / denom)

    if len(good) < min_good:
        score *= len(good) / float(min_good)

    return float(score)


def save_chw_png(t: torch.Tensor, path: str):
    """
    t: [C, H, W], can be on GPU or CPU; dtype can be uint8 or float
    Save as PNG (RGB). If it is single-channel, expand it to pseudo-RGB.
    """
    x = t.detach()
    if x.is_floating_point():
        x = (x.clamp(0, 1) * 255).to(torch.uint8)
    if x.device.type != "cpu":
        x = x.cpu()

    if x.ndim != 3:
        raise ValueError(f"expect CHW, got {tuple(x.shape)}")

    c, h, w = x.shape
    if c == 3:
        img = x.permute(1, 2, 0).numpy()
    else:
        img = x[0].unsqueeze(-1).repeat(1, 1, 3).numpy()

    imageio.imwrite(path, img)


def decode_latents_to_frames(latents: torch.Tensor, vae, scaling_factor: float) -> torch.Tensor:
    """
    latents: [B, T, C, h, w]
    return:  [B, T, C, H, W] in [0, 1]
    """
    b, t = latents.shape[:2]
    flat = rearrange(latents, "b t c h w -> (b t) (h w) c")

    # Make sure the input dtype matches the VAE decode weights
    lat_dtype = vae.post_quant_conv.weight.dtype
    flat = (flat / scaling_factor).to(lat_dtype)

    with torch.no_grad():
        decoded = (vae.decode(flat) + 1) / 2

    decoded = rearrange(decoded, "(b t) c h w -> b t c h w", b=b, t=t)
    return decoded


def main(args):
    device = args.device

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ===== load DiT checkpoint =====
    model = DiT_models["DiT-S/2"]()
    print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(args.oasis_ckpt)}...")
    if args.oasis_ckpt.endswith(".pt"):
        ckpt = torch.load(args.oasis_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt, strict=False)
    elif args.oasis_ckpt.endswith(".safetensors"):
        load_model(model, args.oasis_ckpt)
    else:
        raise ValueError("Unsupported oasis_ckpt format. Use .pt or .safetensors")
    model = model.to(device).eval()

    # ===== load VAE checkpoint =====
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(args.vae_ckpt)}...")
    if args.vae_ckpt.endswith(".pt"):
        vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu", weights_only=True)
        vae.load_state_dict(vae_ckpt)
    elif args.vae_ckpt.endswith(".safetensors"):
        load_model(vae, args.vae_ckpt)
    else:
        raise ValueError("Unsupported vae_ckpt format. Use .pt or .safetensors")
    vae = vae.to(device).eval()

    # ===== sampling params =====
    n_prompt_frames = args.n_prompt_frames
    total_frames = args.num_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1, device=device)
    noise_abs_max = args.noise_abs_max
    stabilization_level = args.stabilization_level
    sim_threshold = args.sim_threshold

    # Override the model window size if --max-frames is provided
    if args.max_frames is not None:
        model.max_frames = args.max_frames

    print(f"model.max_frames = {model.max_frames}")
    print(f"evict_mode = {args.evict_mode}")

    # ===== prompt & actions =====
    x = load_prompt(
        args.prompt_path,
        video_offset=args.video_offset,
        n_prompt_frames=n_prompt_frames,
    )
    actions = load_actions(
        args.actions_path,
        action_offset=args.video_offset,
    )[:, :total_frames]

    x = x.to(device)
    actions = actions.to(device)

    # ===== vae encoding =====
    b = x.shape[0]
    h_img, w_img = x.shape[-2:]
    scaling_factor = 0.07843137255

    x = rearrange(x, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        with autocast("cuda", dtype=torch.half):
            x = vae.encode(x * 2 - 1).mean * scaling_factor

    x = rearrange(
        x,
        "(b t) (h w) c -> b t c h w",
        b=b,
        t=n_prompt_frames,
        h=h_img // vae.patch_size,
        w=w_img // vae.patch_size,
    )
    x = x[:, :n_prompt_frames]

    # ===== context buffer (with possible frame eviction) & full output =====
    x_ctx = x.clone()                         # used for current window inference
    actions_ctx = actions[:, :n_prompt_frames].clone()
    x_all = x.clone()                         # stores the full generated sequence

    # history records the global frame id of each frame in x_ctx
    history = list(range(n_prompt_frames))

    # frames stores the pixel frame corresponding to each global frame id for ORB comparison
    frames = {}
    prompt_decoded = decode_latents_to_frames(x, vae, scaling_factor)  # [B, T, C, H, W]
    for fid in range(n_prompt_frames):
        # Default B=1; if B>1, use the first batch item for similarity evaluation
        frames[fid] = prompt_decoded[0, fid].detach().clone()

    # ===== alphas for DDIM =====
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "t -> t 1 1 1")

    # ===== sampling loop (generate frame by frame) =====
    for i in tqdm(range(n_prompt_frames, total_frames), desc="Sampling"):
        # Append the current action
        actions_ctx = torch.cat([actions_ctx, actions[:, i:i + 1]], dim=1)

        # Initialize a noise latent for the new frame
        chunk = torch.randn((b, 1, *x_ctx.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x_ctx = torch.cat([x_ctx, chunk], dim=1)

        # Evict one frame if the window exceeds the maximum size
        if x_ctx.shape[1] > model.max_frames:
            L = x_ctx.shape[1]

            if args.evict_mode:
                # Dynamic eviction logic:
                # Compare the oldest frame with a middle/later frame
                if L >= 13:
                    idx2 = 13
                else:
                    idx2 = max(1, (2 * L) // 3)

                gid0 = history[0]
                gid2 = history[idx2]

                first_pix = frames[gid0]
                compare2 = frames[gid2]

                sim2 = orb_ransac_score_chw(first_pix, compare2)

                if args.verbose_evict:
                    print(f"[Frame {i}] compare gid0={gid0} vs gid2={gid2}, sim2={sim2:.4f}")

                if sim2 > sim_threshold:
                    # If similarity is high, drop the middle/later frame
                    drop_pos = idx2
                else:
                    # Otherwise, drop the oldest frame
                    drop_pos = 0
            else:
                # Original sliding window / FIFO: always drop the oldest frame
                drop_pos = 0

            keep = [j for j in range(L) if j != drop_pos]
            keep_idx = torch.tensor(keep, device=device, dtype=torch.long)

            x_ctx = torch.index_select(x_ctx, 1, keep_idx)
            actions_ctx = torch.index_select(actions_ctx, 1, keep_idx)

            drop_gid = history[drop_pos]
            del history[drop_pos]
            frames.pop(drop_gid, None)

        # Add the current new frame global id into history
        history.append(i)

        # ===== DDIM denoising for the last frame =====
        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            L = min(x_ctx.shape[1], model.max_frames)
            x_win = x_ctx[:, -L:]          # [B, L, C, h, w]
            act_win = actions_ctx[:, -L:]  # [B, L, D]

            # Set the first L-1 frames to stabilization_level-1,
            # and the last frame to the current DDIM step
            t_ctx = torch.full(
                (b, L - 1),
                stabilization_level - 1,
                dtype=torch.long,
                device=device,
            )
            t_cur = torch.full(
                (b, 1),
                int(noise_range[noise_idx].item()),
                dtype=torch.long,
                device=device,
            )
            t = torch.cat([t_ctx, t_cur], dim=1)

            t_next_cur = torch.full(
                (b, 1),
                int(noise_range[noise_idx - 1].item()),
                dtype=torch.long,
                device=device,
            )
            t_next_cur = torch.where(t_next_cur < 0, t_cur, t_next_cur)
            t_next = torch.cat([t_ctx, t_next_cur], dim=1)

            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    v = model(x_win, t, act_win)

            x_start = alphas_cumprod[t].sqrt() * x_win - (1 - alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_win - x_start) / (
                (1 / alphas_cumprod[t] - 1).sqrt()
            )

            alpha_next = alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])  # freeze history frames
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])

            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()

            # Write back the last frame
            x_ctx[:, -1:] = x_pred[:, -1:]

        # Save this frame to the full output sequence
        x_all = torch.cat([x_all, x_ctx[:, -1:].clone()], dim=1)

        # Decode the current new frame for future ORB comparison
        if args.evict_mode:
            new_frame = decode_latents_to_frames(x_ctx[:, -1:], vae, scaling_factor)  # [B, 1, C, H, W]
            frames[i] = new_frame[0, 0].detach().clone()

            if args.save_debug_frames:
                os.makedirs(args.debug_frame_dir, exist_ok=True)
                save_chw_png(frames[i], os.path.join(args.debug_frame_dir, f"frame_{i:06d}.png"))

    # ===== final decode all frames =====
    x = rearrange(x_all, "b t c h w -> (b t) (h w) c")
    lat_dtype = vae.post_quant_conv.weight.dtype
    with torch.no_grad():
        x = (vae.decode((x / scaling_factor).to(lat_dtype)) + 1) / 2
    x = rearrange(x, "(b t) c h w -> b t h w c", b=b, t=x_all.shape[1])

    # ===== save video =====
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    output_path = args.output_path

    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "output.mp4")
    elif os.path.splitext(output_path)[1] == "":
        output_path = output_path + ".mp4"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    write_video(output_path, x[0].cpu(), fps=args.fps)
    print(f"generation saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--oasis-ckpt",
        type=str,
        default="./checkpoints/oasis500m.safetensors",
        help="Path to Oasis DiT checkpoint.",
    )
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default="./checkpoints/vit-l-20.safetensors",
        help="Path to Oasis ViT-VAE checkpoint.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1200,
        help="How many frames should the output be?",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="./sample_data/sample_image_0.png",
        help="Path to image or video to condition generation on.",
    )
    parser.add_argument(
        "--actions-path",
        type=str,
        default="./sample_data/example.actions.pt",
        help="File to load actions from (.actions.pt or .one_hot_actions.pt)",
    )
    parser.add_argument(
        "--video-offset",
        type=int,
        default=None,
        help="If loading prompt from video, index of frame to start reading from.",
    )
    parser.add_argument(
        "--n-prompt-frames",
        type=int,
        default=1,
        help="If the prompt is a video, how many frames to condition on.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./outputs/output.mp4",
        help="Path where generated video should be saved.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="What framerate should be used to save the output?",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=10,
        help="How many DDIM steps?",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='CUDA device, e.g. "cuda:0"',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Override model.max_frames. Set None if you do not want to override.",
    )
    parser.add_argument(
        "--noise-abs-max",
        type=float,
        default=20.0,
        help="Clamp range for initial Gaussian noise.",
    )
    parser.add_argument(
        "--stabilization-level",
        type=int,
        default=15,
        help="Stabilization timestep for context frames.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.78,
        help="Similarity threshold for ORB-based eviction.",
    )
    parser.add_argument(
        "--evict-mode",
        type=str2bool,
        default=True,
        help="True: use dynamic ORB eviction; False: use vanilla FIFO eviction.",
    )
    parser.add_argument(
        "--verbose-evict",
        type=str2bool,
        default=False,
        help="Whether to print ORB eviction details.",
    )
    parser.add_argument(
        "--save-debug-frames",
        type=str2bool,
        default=False,
        help="Whether to save decoded frames for debugging.",
    )
    parser.add_argument(
        "--debug-frame-dir",
        type=str,
        default="./debug_frames",
        help="Where to save debug frames if save_debug_frames=True.",
    )

    args = parser.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
    print("\n✅ All done.")