from typing import List, Optional
import numpy as np
import torch
import time
import copy

from einops import rearrange
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
from utils.visualize import process_video
import torch.nn.functional as F
from demo_utils.constant import ZERO_VAE_CACHE
from tqdm import tqdm

def get_current_action(mode="universal"):

    CAM_VALUE = 0.1
    if mode == 'universal':
        print()
        print('-'*30)
        print("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM\n (I: up, K: down, J: left, L: right, U: no move)")
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print('-'*30)
        CAMERA_VALUE_MAP = {
            "i":  [CAM_VALUE, 0],
            "k":  [-CAM_VALUE, 0],
            "j":  [0, -CAM_VALUE],
            "l":  [0, CAM_VALUE],
            "u":  [0, 0]
        }
        KEYBOARD_IDX = { 
            "w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0], "d": [0, 0, 0, 1],
            "q": [0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_mouse = input('Please input the mouse action (e.g. `U`):\n').strip().lower()
                idx_keyboard = input('Please input the keyboard action (e.g. `W`):\n').strip().lower()
                if idx_mouse in CAMERA_VALUE_MAP.keys() and idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    elif mode == 'gta_drive':
        print()
        print('-'*30)
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print('-'*30)
        CAMERA_VALUE_MAP = {
            "a":  [0, -CAM_VALUE],
            "d":  [0, CAM_VALUE],
            "q":  [0, 0]
        }
        KEYBOARD_IDX = { 
            "w": [1, 0], "s": [0, 1],
            "q": [0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                indexes = input('Please input the actions (split with ` `):\n(e.g. `W` for forward, `W A` for forward and left)\n').strip().lower().split(' ')
                idx_mouse = []
                idx_keyboard = []
                for i in indexes:
                    if i in CAMERA_VALUE_MAP.keys():
                        idx_mouse += [i]
                    elif i in KEYBOARD_IDX.keys():
                        idx_keyboard += [i]
                if len(idx_mouse) == 0:
                    idx_mouse += ['q']
                if len(idx_keyboard) == 0:
                    idx_keyboard += ['q']
                assert idx_mouse in [['a'], ['d'], ['q']] and idx_keyboard in [['q'], ['w'], ['s']]
                flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse[0]]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard[0]]).cuda()
    elif mode == 'templerun':
        print()
        print('-'*30)
        print("PRESS [W, S, A, D, Z, C, Q] FOR ACTIONS\n (W: jump, S: slide, A: left side, D: right side, Z: turn left, C: turn right, Q: no move)")
        print('-'*30)
        KEYBOARD_IDX = { 
            "w": [0, 1, 0, 0, 0, 0, 0], "s": [0, 0, 1, 0, 0, 0, 0],
            "a": [0, 0, 0, 0, 0, 1, 0], "d": [0, 0, 0, 0, 0, 0, 1],
            "z": [0, 0, 0, 1, 0, 0, 0], "c": [0, 0, 0, 0, 1, 0, 0],
            "q": [1, 0, 0, 0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_keyboard = input('Please input the action: \n(e.g. `W` for forward, `Z` for turning left)\n').strip().lower()
                if idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    
    if mode != 'templerun':
        return {
            "mouse": mouse_cond,
            "keyboard": keyboard_cond
        }
    return {
        "keyboard": keyboard_cond
    }

def cond_current(conditional_dict, current_start_frame, num_frame_per_block, replace=None, mode='universal'):
    
    new_cond = {}
    
    new_cond["cond_concat"] = conditional_dict["cond_concat"][:, :, current_start_frame: current_start_frame + num_frame_per_block]
    new_cond["visual_context"] = conditional_dict["visual_context"]
    if replace != None:
        if current_start_frame == 0:
            last_frame_num = 1 + 4 * (num_frame_per_block - 1)
        else:
            last_frame_num = 4 * num_frame_per_block
        final_frame = 1 + 4 * (current_start_frame + num_frame_per_block-1)
        if mode != 'templerun':
            conditional_dict["mouse_cond"][:, -last_frame_num + final_frame: final_frame] = replace['mouse'][None, None, :].repeat(1, last_frame_num, 1)
        conditional_dict["keyboard_cond"][:, -last_frame_num + final_frame: final_frame] = replace['keyboard'][None, None, :].repeat(1, last_frame_num, 1)
    if mode != 'templerun':
        new_cond["mouse_cond"] = conditional_dict["mouse_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]
    new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]

    if replace != None:
        return new_cond, conditional_dict
    else:
        return new_cond

import cv2


def chw_to_gray_u8(chw: torch.Tensor) -> np.ndarray:
    """
    chw: [C, H, W] tensor, value range can be [-1, 1] or [0, 1]
    Returns: HxW uint8 grayscale image (0~255)
    """
    x = chw.detach().cpu().float()

    # Automatically detect the input range and normalize
    if x.min() < 0:  # assume range is [-1, 1]
        x = ((x + 1.0) * 127.5).clamp(0, 255.0)
    else:            # assume range is [0, 1]
        x = (x * 255.0).clamp(0, 255.0)

    # Convert to grayscale
    if x.shape[0] == 3:
        r, g, b = x[0], x[1], x[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = x[0]

    return gray.byte().numpy()


def orb_ransac_score_chw(chwA: torch.Tensor,
                         chwB: torch.Tensor,
                         ratio_thresh: float = 0.8,
                         min_good: int = 30) -> float:
    """
    Use ORB + RANSAC to estimate the inlier ratio of feature matches
    as a similarity score in [0, 1].

    - Compute both Homography and Fundamental Matrix,
      and take the larger inlier ratio.
    """
    img1 = chw_to_gray_u8(chwA)
    img2 = chw_to_gray_u8(chwB)

    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=7)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)

    good = []
    for matches in knn:
        if len(matches) < 2:
            continue
        m, n = matches
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 7:
        return 0.0

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])

    # Estimate Homography / Fundamental Matrix with RANSAC
    H, maskH = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    F, maskF = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

    inH = int(maskH.sum()) if maskH is not None else 0
    inF = int(maskF.sum()) if isinstance(maskF, np.ndarray) and maskF.size > 0 else 0

    denom = max(len(good), 1)
    ratioH = inH / denom
    ratioF = inF / denom

    # If there are too few matches, downscale the score
    # to avoid occasional overly high scores
    if len(good) < min_good:
        scale = len(good) / float(min_good)
        return float(max(ratioH, ratioF) * scale)

    return float(max(ratioH, ratioF))


def get_decoded_frame_by_latent(videos: list, latent_id: int, sub: int = 2):
    """
    From `videos`, retrieve the corresponding decoded single frame (CHW tensor)
    according to the latent frame index.

    - videos[blk] has shape [1, T_blk, 3, H, W],
      where T_blk is 9 for the first block and 12 for all later blocks
    - Each latent block contains 3 latent frames: pos = 0, 1, 2
    - `sub` selects which sub-frame to use among the 4x upsampled frames,
      where sub ∈ {0,1,2,3}
    - For the last latent in the first block, only one frame is available
    """
    blk = latent_id // 3           # which block the latent belongs to
    pos = latent_id % 3            # position within the block (0, 1, or 2)

    v = videos[blk]                # [1, T_blk, 3, H, W]
    T_blk = v.shape[1]

    if blk == 0:
        # First block: 4 + 4 + 1 = 9
        if pos == 0:          # first latent -> off: 0..3
            off = min(sub, 3)
        elif pos == 1:        # second latent -> off: 4..7
            off = min(4 + sub, 7)
        else:                 # third latent -> only off: 8
            off = 8
    else:
        # Other blocks: 4 + 4 + 4 = 12
        off = pos * 4 + min(sub, 3)   # 0..3, 4..7, 8..11

    # Safety fallback to avoid out-of-range indexing
    if off >= T_blk:
        off = T_blk - 1

    # Extract a single frame [3, H, W]
    chw = v[0, off]   # note: video layout is [1, T_blk, 3, H, W]
    return chw


def decide_and_update_window_ids_tri_9(
    window_ids: list,     # ascending order, usually length = 12 (also supports >= 9)
    videos: list,         # decoded video blocks, each with shape [1, T_blk, 3, H, W]
    sim_threshold: float = 0.75,   # ORB+RANSAC inlier ratio threshold
    sub: int = 2,         # which sub-frame of each latent to compare (recommended: 1 or 2)
) -> tuple[bool, list, float, dict]:
    """
    Rules (similarity is based on positions inside the window, not actual frame numbers):

      Take latent frames win[2], win[5], and win[8], and map them to decoded frames.
      Let s25 = sim(win[2], win[5]), s28 = sim(win[2], win[8])

      1) s25 >= thr and s28 <  thr  -> evict win[3:6]   (3,4,5)
      2) s25 >= thr and s28 >= thr  -> evict win[6:9]   (6,7,8)
      3) s25 <  thr and s28 <  thr  -> evict win[0:3]   (0,1,2)
      Otherwise / boundary cases / insufficient length -> fallback to evicting the oldest 3 frames

    Returns:
      evict_middle: whether this is a middle-section eviction
                    (True: evict 3–5 or 6–8; False: evict 0–2)
      new_ids:      updated window frame id list (same length as input)
      min_sim:      min(s25, s28), useful for logging
      debug:        {'s25': ..., 's28': ..., 'case': 'A/B/C/FALLBACK'}
    """
    L = len(window_ids)
    assert L >= 6, "Window must contain at least 6 frames"
    last_id = window_ids[-1]

    # Need to access positions 2, 5, and 8
    if L < 7:
        # Fallback: window too short, evict the oldest 3 frames
        evict_middle = 0
        new_ids = window_ids[3:] + [last_id + 1, last_id + 2, last_id + 3]
        return evict_middle, new_ids, 0.0, {'case': 'FALLBACK_L<9'}

    # --- Retrieve the three frames (based on positions inside the sliding window) ---
    id2 = window_ids[2]
    id5 = window_ids[5]

    # Map them to decoded frames and extract single-frame CHW tensors
    img2 = get_decoded_frame_by_latent(videos, id2, sub=sub)
    img5 = get_decoded_frame_by_latent(videos, id5, sub=sub)

    # --- Compute similarity for the two pairs ---
    s25 = orb_ransac_score_chw(img2, img5)  # range: 0~1, larger means more similar
    min_sim = s25

    # --- Branch according to the three cases ---
    # A: high similarity between 2 and 5 -> evict 3..5
    if (s25 >= sim_threshold):
        kept = window_ids[:3] + window_ids[6:]     # keep 0,1,2 and 6..end
        evict_middle = 1
    else:
        # Other mixed cases: fallback to evicting the oldest part for stability
        kept = window_ids[3:]
        evict_middle = 0
 

    # --- Append 3 new frame ids at the tail (same as original logic) ---
    need = 3
    next_ids = [last_id + i + 1 for i in range(need)]
    new_ids = kept + next_ids

    # Ensure the window length remains unchanged
    assert len(new_ids) == L, f"Window length should remain {L}, got {len(new_ids)}"
    return evict_middle, new_ids, min_sim

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device="cuda",
            generator=None,
            vae_decoder=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
            
        self.vae_decoder = vae_decoder
        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict,
        initial_latent = None,
        return_latents = False,
        mode = 'universal',
        profile = False,
        evict_mode = False,
        Threshold=0.78
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        
        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape
        
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache=None
        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block


        evict_middle=0
        all_num_frames = [self.num_frame_per_block] * num_blocks

        window_ids=[0,1,2,3,4,5,6,7,8]

        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            
        # Step 3
        for current_num_frames in tqdm(all_num_frames):

            noisy_input = noise[
                :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            if profile:
                torch.cuda.synchronize()
                diffusion_start.record()
            valid_len = current_start_frame
            if valid_len >= len(window_ids) and evict_mode:

                evict_middle, window_ids, sim_min = decide_and_update_window_ids_tri_9(
                    window_ids=window_ids,
                    videos=videos,         
                    sim_threshold=Threshold,     
                )

            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        evict_middle=evict_middle
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),# .flatten(0, 1),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        evict_middle=evict_middle
                    )

            B, C, F_blk, H, W = denoised_pred.shape   # 例如 [1,16,3,44,80]
            assert B == 1


            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.args.context_noise

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                evict_middle=evict_middle
            )

            current_start_frame += current_num_frames
            denoised_pred = denoised_pred.transpose(1,2)
            video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
            videos += [video]

        if return_latents:
            return output
        else:
            return videos

    # def inference(
    #     self,
    #     noise: torch.Tensor,
    #     conditional_dict,
    #     initial_latent = None,
    #     return_latents = False,
    #     mode = 'universal',
    #     profile = False,
    # ) -> torch.Tensor:
    #     """
    #     Perform inference on the given noise and text prompts.
    #     Inputs:
    #         noise (torch.Tensor): The input noise tensor of shape
    #             (batch_size, num_output_frames, num_channels, height, width).
    #         text_prompts (List[str]): The list of text prompts.
    #         initial_latent (torch.Tensor): The initial latent tensor of shape
    #             (batch_size, num_input_frames, num_channels, height, width).
    #             If num_input_frames is 1, perform image to video.
    #             If num_input_frames is greater than 1, perform video extension.
    #         return_latents (bool): Whether to return the latents.
    #     Outputs:
    #         video (torch.Tensor): The generated video tensor of shape
    #             (batch_size, num_output_frames, num_channels, height, width).
    #             It is normalized to be in the range [0, 1].
    #     """
        
    #     assert noise.shape[1] == 16
    #     batch_size, num_channels, num_frames, height, width = noise.shape
        
    #     assert num_frames % self.num_frame_per_block == 0
    #     num_blocks = num_frames // self.num_frame_per_block

    #     num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
    #     num_output_frames = num_frames + num_input_frames  # add the initial latent frames

    #     output = torch.zeros(
    #         [batch_size, num_channels, num_output_frames, height, width],
    #         device=noise.device,
    #         dtype=noise.dtype
    #     )
    #     videos = []
    #     vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
    #     for j in range(len(vae_cache)):
    #         vae_cache[j] = None

    #     self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache=None
    #     # Step 1: Initialize KV cache to all zeros
    #     if self.kv_cache1 is None:
    #         self._initialize_kv_cache(
    #             batch_size=batch_size,
    #             dtype=noise.dtype,
    #             device=noise.device
    #         )
    #         self._initialize_kv_cache_mouse_and_keyboard(
    #             batch_size=batch_size,
    #             dtype=noise.dtype,
    #             device=noise.device
    #         )
            
    #         self._initialize_crossattn_cache(
    #             batch_size=batch_size,
    #             dtype=noise.dtype,
    #             device=noise.device
    #         )
    #     else:
    #         # reset cross attn cache
    #         for block_index in range(self.num_transformer_blocks):
    #             self.crossattn_cache[block_index]["is_init"] = False
    #         # reset kv cache
    #         for block_index in range(len(self.kv_cache1)):
    #             self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #     # Step 2: Cache context feature
    #     current_start_frame = 0
    #     if initial_latent is not None:
    #         timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
    #         # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
    #         assert num_input_frames % self.num_frame_per_block == 0
    #         num_input_blocks = num_input_frames // self.num_frame_per_block

    #         for _ in range(num_input_blocks):
    #             current_ref_latents = \
    #                 initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
    #             output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                
    #             self.generator(
    #                 noisy_image_or_video=current_ref_latents,
    #                 conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
    #                 timestep=timestep * 0,
    #                 kv_cache=self.kv_cache1,
    #                 kv_cache_mouse=self.kv_cache_mouse,
    #                 kv_cache_keyboard=self.kv_cache_keyboard,
    #                 crossattn_cache=self.crossattn_cache,
    #                 current_start=current_start_frame * self.frame_seq_length,
    #             )
    #             current_start_frame += self.num_frame_per_block


    #     # Step 3: Temporal denoising loop
    #     all_num_frames = [self.num_frame_per_block] * num_blocks
    #     if profile:
    #         diffusion_start = torch.cuda.Event(enable_timing=True)
    #         diffusion_end = torch.cuda.Event(enable_timing=True)
    #     for current_num_frames in tqdm(all_num_frames):

    #         noisy_input = noise[
    #             :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

    #         # Step 3.1: Spatial denoising loop
    #         if profile:
    #             torch.cuda.synchronize()
    #             diffusion_start.record()
    #         for index, current_timestep in enumerate(self.denoising_step_list):
    #             # set current timestep
    #             timestep = torch.ones(
    #                 [batch_size, current_num_frames],
    #                 device=noise.device,
    #                 dtype=torch.int64) * current_timestep

    #             if index < len(self.denoising_step_list) - 1:
    #                 _, denoised_pred = self.generator(
    #                     noisy_image_or_video=noisy_input,
    #                     conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
    #                     timestep=timestep,
    #                     kv_cache=self.kv_cache1,
    #                     kv_cache_mouse=self.kv_cache_mouse,
    #                     kv_cache_keyboard=self.kv_cache_keyboard,
    #                     crossattn_cache=self.crossattn_cache,
    #                     current_start=current_start_frame * self.frame_seq_length
    #                 )
    #                 next_timestep = self.denoising_step_list[index + 1]
    #                 noisy_input = self.scheduler.add_noise(
    #                     rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),# .flatten(0, 1),
    #                     torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
    #                     next_timestep * torch.ones(
    #                         [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
    #                 )
    #                 noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
    #             else:
    #                 # for getting real output
    #                 _, denoised_pred = self.generator(
    #                     noisy_image_or_video=noisy_input,
    #                     conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
    #                     timestep=timestep,
    #                     kv_cache=self.kv_cache1,
    #                     kv_cache_mouse=self.kv_cache_mouse,
    #                     kv_cache_keyboard=self.kv_cache_keyboard,
    #                     crossattn_cache=self.crossattn_cache,
    #                     current_start=current_start_frame * self.frame_seq_length
    #                 )

    #         # Step 3.2: record the model's output
    #         output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

    #         # Step 3.3: rerun with timestep zero to update KV cache using clean context
    #         context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
    #         self.generator(
    #             noisy_image_or_video=denoised_pred,
    #             conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
    #             timestep=context_timestep,
    #             kv_cache=self.kv_cache1,
    #             kv_cache_mouse=self.kv_cache_mouse,
    #             kv_cache_keyboard=self.kv_cache_keyboard,
    #             crossattn_cache=self.crossattn_cache,
    #             current_start=current_start_frame * self.frame_seq_length,
    #         )

    #         # Step 3.4: update the start and end frame indices
    #         current_start_frame += current_num_frames

    #         denoised_pred = denoised_pred.transpose(1,2)
    #         video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
    #         videos += [video]

    #         if profile:
    #             torch.cuda.synchronize()
    #             diffusion_end.record()
    #             diffusion_time = diffusion_start.elapsed_time(diffusion_end)
    #             print(f"diffusion_time: {diffusion_time}", flush=True)
    #             fps = video.shape[1]*1000/ diffusion_time
    #             print(f"  - FPS: {fps:.2f}")

    #     if return_latents:
    #         return output
    #     else:
    #         return videos

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 15 * 1 * self.frame_seq_length # 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_keyboard = kv_cache_keyboard  # always store the clean cache
        self.kv_cache_mouse = kv_cache_mouse  # always store the clean cache

        

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache


class CausalInferenceStreamingPipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device="cuda",
            vae_decoder=None,
            generator=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.vae_decoder = vae_decoder

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880 # 1590 # HW/4

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        output_folder = None,
        name = None,
        mode = 'universal'
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        
        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape
        
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None
        # Set up profiling if requested
        self.kv_cache1=self.kv_cache_keyboard=self.kv_cache_mouse=self.crossattn_cache=None
        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=True),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            current_actions = get_current_action(mode=mode)
            new_act, conditional_dict = cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=current_actions, mode=mode)
            # Step 3.1: Spatial denoising loop

            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),# .flatten(0, 1),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=new_act,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # Step 3.4: update the start and end frame indices
            denoised_pred = denoised_pred.transpose(1,2)
            video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
            videos += [video]
            video = rearrange(video, "B T C H W -> B T H W C")
            video = ((video.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
            video = np.ascontiguousarray(video)
            mouse_icon = 'assets/images/mouse.png'
            if mode != 'templerun':
                config = (
                    conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                    conditional_dict["mouse_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                )
            else:
                config = (
                    conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy()
                )
            process_video(video.astype(np.uint8), output_folder+f'/{name}_current.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)
            current_start_frame += current_num_frames

            if input("Continue? (Press `n` to break)").strip() == "n":
                break
                
        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)
        mouse_icon = 'assets/images/mouse.png'
        if mode != 'templerun':
            config = (
                conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                conditional_dict["mouse_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
            )
        else:
            config = (
                conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy()
            )
        process_video(video.astype(np.uint8), output_folder+f'/{name}_icon.mp4', config, mouse_icon, mouse_scale=0.1, mode=mode)
        process_video(video.astype(np.uint8), output_folder+f'/{name}.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)

        if return_latents:
            return output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 15 * 1 * self.frame_seq_length # 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_keyboard = kv_cache_keyboard  # always store the clean cache
        self.kv_cache_mouse = kv_cache_mouse  # always store the clean cache

        

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
