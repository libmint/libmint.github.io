아래는 **3090(24GB)에서 OOM을 최대한 피하도록 generate_prequant.py를 “통째로” 수정한 버전**입니다.
그대로 **파일 하나로 저장해서 실행**하면 됩니다. (원본 스크립트 기반: HF `generate_prequant.py`)

* ✅ **FP32로 큰 텐서를 만들던 부분 제거**(noise, zeros)
* ✅ **CPU↔GPU 불필요 왕복 제거**(interpolate `.cpu()` 제거)
* ✅ **CFG 계산을 in-place로 변경**(피크 메모리 감소)
* ✅ **`torch.inference_mode()` 적용**
* ✅ **SDPA(Flash/mem-efficient) 백엔드 우선 사용**
* ✅ **step마다 `empty_cache()` 남발 제거**(오히려 단편화 유발 가능)

---

## 1) 실행 전 export (복붙)

```bash
# (중요) CUDA 메모리 단편화 완화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# (선택) PyTorch가 flash/mem-efficient SDPA를 더 잘 쓰게 유도하는 경우가 있음
export NVIDIA_TF32_OVERRIDE=1
```

---

## 2) 수정된 전체 코드 (그대로 `generate_prequant_3090.py` 로 저장)

```python
#!/usr/bin/env python3
"""
Generate videos using PRE-QUANTIZED bitsandbytes NF4 models.

Unlike generate_bnb.py which re-quantizes at runtime, this script loads
pre-quantized weights directly. No base model weights are needed.

Prerequisites:
- Pre-quantized models in {ckpt_dir}/{high,low}_noise_model_bnb_nf4/
- Each should contain model.safetensors (or model.pt) + config.json

Usage:
python generate_prequant_3090.py \
  --image examples/00/image.jpg \
  --prompt "A cinematic video of the scene" \
  --frame_num 81 \
  --size 480*832
"""

import argparse
import gc
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from einops import rearrange
from load_prequant import load_quantized_model
from wan.configs.wan_i2v_A14B import i2v_A14B as cfg
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Memory / attention backend tweaks (important for 24GB GPUs) ----
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass
# -------------------------------------------------------------------


class WanI2V_PreQuant:
    """Image-to-video pipeline using pre-quantized NF4 models."""

    def __init__(
        self,
        checkpoint_dir: str,
        device_id: int = 0,
        t5_cpu: bool = True,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = cfg
        self.t5_cpu = t5_cpu
        self.num_train_timesteps = cfg.num_train_timesteps
        self.boundary = cfg.boundary
        self.param_dtype = cfg.param_dtype
        self.vae_stride = cfg.vae_stride
        self.patch_size = cfg.patch_size
        self.sample_neg_prompt = cfg.sample_neg_prompt

        # Load T5 encoder (not quantized)
        logger.info("Loading T5 encoder...")
        local_tokenizer = os.path.join(checkpoint_dir, "tokenizer")
        tokenizer_path = local_tokenizer if os.path.isdir(local_tokenizer) else cfg.t5_tokenizer

        self.text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, cfg.t5_checkpoint),
            tokenizer_path=tokenizer_path,
            shard_fn=None,
        )

        # Load VAE (not quantized)
        logger.info("Loading VAE...")
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, cfg.vae_checkpoint),
            device=self.device,
        )

        # Load PRE-QUANTIZED diffusion models
        logger.info("Loading pre-quantized NF4 diffusion models...")
        low_noise_dir = os.path.join(checkpoint_dir, cfg.low_noise_checkpoint + "_bnb_nf4")
        high_noise_dir = os.path.join(checkpoint_dir, cfg.high_noise_checkpoint + "_bnb_nf4")

        for d in [low_noise_dir, high_noise_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Pre-quantized model not found: {d}\n"
                    "Run: python scripts/quantize_and_package.py first"
                )

        # Load to CPU first; swap to GPU per timestep
        self.low_noise_model = load_quantized_model(low_noise_dir, device="cpu")
        self.high_noise_model = load_quantized_model(high_noise_dir, device="cpu")
        logger.info("Model loading complete!")

    def _prepare_model_for_timestep(self, t, boundary):
        """Prepare and return the required model for the current timestep."""
        if t.item() >= boundary:
            required_model_name = "high_noise_model"
            offload_model_name = "low_noise_model"
        else:
            required_model_name = "low_noise_model"
            offload_model_name = "high_noise_model"

        required_model = getattr(self, required_model_name)
        offload_model = getattr(self, offload_model_name)

        # Offload unused model to CPU
        try:
            if next(offload_model.parameters()).device.type == "cuda":
                offload_model.to("cpu")
                # empty_cache()는 여기서는 한 번만 (너무 자주 부르면 단편화 유발 가능)
                torch.cuda.empty_cache()
        except StopIteration:
            pass

        # Load required model to GPU
        try:
            if next(required_model.parameters()).device.type == "cpu":
                required_model.to(self.device)
        except StopIteration:
            pass

        return required_model

    def generate(
        self,
        input_prompt: str,
        img: Image.Image,
        action_path: str = None,
        max_area: int = 720 * 1280,
        frame_num: int = 81,
        shift: float = 5.0,
        sampling_steps: int = 40,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seed: int = -1,
    ):
        """Generate video from image and text prompt."""
        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]
            guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale

        # Image tensor on GPU
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img_tensor.shape[1:]
        aspect_ratio = h / w

        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.vae_stride[1]
            // self.patch_size[1]
            * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.vae_stride[2]
            // self.patch_size[2]
            * self.patch_size[2]
        )

        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # ---- IMPORTANT: noise in param_dtype (not fp32) ----
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=self.param_dtype,
            generator=seed_g,
            device=self.device,
        )

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Encode text
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # Camera preparation
        dit_cond_dict = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            Ks = get_Ks_transformed(Ks, 480, 832, h, w, h, w)
            Ks = Ks[0]

            len_c2ws = len(c2ws)
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)
            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)

            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "b (f h w) c -> b c f h w",
                f=lat_f,
                h=lat_h,
                w=lat_w,
            ).to(self.param_dtype)
            dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

        # ---- Encode image (avoid CPU round-trip + avoid fp32 zeros) ----
        img_resized = torch.nn.functional.interpolate(
            img_tensor[None], size=(h, w), mode="bicubic"
        ).transpose(0, 1).to(dtype=self.param_dtype)

        zeros = torch.zeros(3, F - 1, h, w, device=self.device, dtype=self.param_dtype)

        y = self.vae.encode([torch.concat([img_resized, zeros], dim=1)])[0]
        y = y.to(dtype=self.param_dtype)
        y = torch.concat([msk, y])

        # Diffusion sampling
        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.inference_mode():
            boundary = self.boundary * self.num_train_timesteps
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps

            latent = noise

            arg_c = {
                "context": [context[0]],
                "seq_len": max_seq_len,
                "y": [y],
                "dit_cond_dict": dit_cond_dict,
            }
            arg_null = {
                "context": context_null,
                "seq_len": max_seq_len,
                "y": [y],
                "dit_cond_dict": dit_cond_dict,
            }

            # Pre-load first model
            first_model_name = "high_noise_model" if timesteps[0].item() >= boundary else "low_noise_model"
            getattr(self, first_model_name).to(self.device)
            logger.info(f"Loaded {first_model_name} to GPU")

            for _, t in enumerate(tqdm(timesteps, desc="Sampling")):
                model = self._prepare_model_for_timestep(t, boundary)
                s = guide_scale[1] if t.item() >= boundary else guide_scale[0]

                latent_model_input = [latent]
                timestep = t[None].to(self.device)

                # uncond
                uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                # cond
                cond = model(latent_model_input, t=timestep, **arg_c)[0]

                # in-place CFG: cond = uncond + s*(cond - uncond)
                cond.sub_(uncond)
                cond.mul_(s)
                cond.add_(uncond)
                del uncond

                temp_x0 = sample_scheduler.step(
                    cond.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latent = temp_x0.squeeze(0)

                del cond, temp_x0

            # Offload models
            self.low_noise_model.cpu()
            self.high_noise_model.cpu()
            torch.cuda.empty_cache()

            # Decode video
            videos = self.vae.decode([latent])

        # cleanup
        del noise, latent, zeros, img_resized
        gc.collect()
        torch.cuda.synchronize()

        return videos[0]


def save_video(frames: torch.Tensor, output_path: str, fps: int = 16):
    """Save video frames to file."""
    import imageio

    frames = ((frames + 1) / 2 * 255).clamp(0, 255).byte()
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()
    imageio.mimwrite(output_path, frames, fps=fps, codec="libx264")
    logger.info(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate videos with pre-quantized NF4 models (3090-friendly)")
    script_dir = str(Path(__file__).parent)

    parser.add_argument("--ckpt_dir", type=str, default=script_dir)
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--action_path", type=str, default=None, help="Camera control path")
    parser.add_argument("--size", type=str, default="480*832", help="Output resolution")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sampling_steps", type=int, default=40)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--t5_cpu", action="store_true", default=True)

    args = parser.parse_args()

    h, w = map(int, args.size.split("*"))
    max_area = h * w

    img = Image.open(args.image).convert("RGB")
    pipeline = WanI2V_PreQuant(checkpoint_dir=args.ckpt_dir, t5_cpu=args.t5_cpu)

    logger.info("Generating video...")
    video = pipeline.generate(
        input_prompt=args.prompt,
        img=img,
        action_path=args.action_path,
        max_area=max_area,
        frame_num=args.frame_num,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
    )
    save_video(video, args.output)


if __name__ == "__main__":
    main()
```

---

## 3) 실행 예시 (복붙)

```bash
python generate_prequant_3090.py \
  --ckpt_dir . \
  --image examples/00/image.jpg \
  --prompt "A cinematic video of the scene" \
  --frame_num 81 \
  --sampling_steps 40 \
  --size 480*832 \
  --output out.mp4
```

---

## 4) 그래도 OOM이면 “다음 1개만” 확인하면 끝입니다

OOM이 **어느 구간**에서 나는지에 따라 “확실한 마지막 패치”가 달라요.

* Sampling 진행바 돌다가 OOM → **attention/모델 forward 피크**
* Sampling 끝나고 OOM → **VAE decode 피크**

지금 상태에서 여전히 터지면, 터미널에 찍힌 **OOM 에러 마지막 25줄**만 붙여주세요.
(진행바가 `Sampling` 중이었는지도 같이)
그러면 그 위치에 맞춰 **VAE 타일 디코드 / attention 강제 backend / chunking** 중 “딱 맞는” 최종 패치를 바로 드릴게요.
