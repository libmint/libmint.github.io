좋아요. 그럼 **“바로 복붙해서 쓰는” 완성본 2개 파일**로 드릴게요.

* ✅ `generate_prequant_3090.py` : **3090-friendly + lazy-load(모델 1개만 메모리 상주) + VRAM 피크 감소 패치**
* ✅ `load_prequant.py` : **RAM OOM 핵심 원인인 `load_file()`(전체 dict 로드) 제거 → `safe_open` 스트리밍 로딩으로 교체**

> 당신이 겪은 “`model.safetensors`에서 RAM OOM”은 보통 `load_file()`이 **state_dict 전체를 한 번에 RAM에 올리는 순간 피크가 튀어서** 터집니다. 이걸 스트리밍으로 바꾸면 피크가 확 내려갑니다. ([Hugging Face][1])

---

## 0) 실행 전 export (복붙)

```bash
# RAM 파편화/피크 완화에 도움되는 경우 많음
export MALLOC_ARENA_MAX=2

# (GPU 파편화 완화도 같이)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

---

## 1) `load_prequant.py` (전체 복붙해서 교체)

> **기존 `load_prequant.py` 파일을 아래 내용으로 통째로 덮어쓰기** 하세요.

```python
#!/usr/bin/env python3
"""
Load pre-quantized bitsandbytes NF4 models without needing base weights.

This module provides utilities to load pre-quantized WanModel weights
directly from safetensors/pt files, without downloading or loading
the original FP16/BF16 base model weights.

Key change (3090 / 64GB RAM friendly):
- SAFETENSORS are loaded via safe_open streaming (tensor-by-tensor),
  instead of load_file() which loads the entire state_dict into RAM.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.functional import QuantState

# ✅ streaming safetensors
from safetensors import safe_open

# Add parent to path for wan imports
sys.path.insert(0, str(Path(__file__).parent))

from wan.modules.model import WanModel


def replace_linears_with_bnb_nf4(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    compress_statistics: bool = True,
    quant_type: str = "nf4",
) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    """
    Replace all nn.Linear layers with empty bnb.nn.Linear4bit layers.
    Creates the structure needed to load pre-quantized weights.
    """
    replaced = 0
    layer_shapes: Dict[str, Tuple[int, int]] = {}

    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    for name, module in linear_layers:
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        layer_shapes[name] = (module.in_features, module.out_features)

        nf4_linear = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
        )
        setattr(parent, child_name, nf4_linear)
        replaced += 1

    return replaced, layer_shapes


def build_model_from_config(config: Dict[str, Any]) -> WanModel:
    """Build a WanModel instance from config dictionary."""
    model = WanModel(
        model_type=config.get("model_type", "i2v"),
        patch_size=tuple(config.get("patch_size", (1, 2, 2))),
        text_len=config.get("text_len", 512),
        in_dim=config.get("in_dim", 16),
        dim=config.get("dim", 2048),
        ffn_dim=config.get("ffn_dim", 8192),
        freq_dim=config.get("freq_dim", 256),
        text_dim=config.get("text_dim", 4096),
        out_dim=config.get("out_dim", 16),
        num_heads=config.get("num_heads", 16),
        num_layers=config.get("num_layers", 32),
        window_size=tuple(config.get("window_size", (-1, -1))),
        qk_norm=config.get("qk_norm", True),
        cross_attn_norm=config.get("cross_attn_norm", True),
        eps=config.get("eps", 1e-6),
    )
    return model


def reconstruct_params4bit_from_components(
    weight_components: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> bnb.nn.Params4bit:
    """
    Reconstruct a Params4bit object from serialized components using QuantState.from_dict.
    """
    qs_dict = {
        "absmax": weight_components["absmax"],
        "quant_map": weight_components["quant_map"],
    }

    if "nested_absmax" in weight_components:
        qs_dict["nested_absmax"] = weight_components["nested_absmax"]
        qs_dict["nested_quant_map"] = weight_components["nested_quant_map"]

    if "quant_state_data" in weight_components:
        qs_dict["quant_state.bitsandbytes__nf4"] = weight_components["quant_state_data"]

    quant_state = QuantState.from_dict(qs_dict, device=torch.device(device))
    quantized_weight = weight_components["weight"].to(device)

    param = bnb.nn.Params4bit(
        data=quantized_weight,
        requires_grad=False,
        quant_state=quant_state,
        bnb_quantized=True,  # already quantized
    )
    return param


def _stream_load_safetensors(
    model: nn.Module,
    weights_path: str,
    layer_shapes: Dict[str, Tuple[int, int]],
    device: str = "cpu",
) -> nn.Module:
    """
    ✅ RAM-friendly: stream tensors from safetensors one-by-one (no full state_dict in RAM).
    Loads:
      - quantized Linear4bit weights via component tensors
      - remaining non-quant params/buffers via in-place copy
    """

    # quick lookup maps for non-quant keys
    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())

    # find all Linear4bit module names
    linear4bit_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            linear4bit_names.add(name)

    def is_quant_component(base: str, k: str) -> bool:
        return k in {
            f"{base}.weight",
            f"{base}.weight.absmax",
            f"{base}.weight.quant_map",
            f"{base}.weight.nested_absmax",
            f"{base}.weight.nested_quant_map",
            f"{base}.weight.quant_state.bitsandbytes__nf4",
        }

    with safe_open(weights_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())

        # 1) Load quantized Linear4bit layers (layer-by-layer)
        loaded_count = 0
        for name, module in model.named_modules():
            if not isinstance(module, bnb.nn.Linear4bit):
                continue
            if name not in layer_shapes:
                continue

            # required
            k_weight = f"{name}.weight"
            k_absmax = f"{name}.weight.absmax"
            k_qmap = f"{name}.weight.quant_map"
            k_qstate = f"{name}.weight.quant_state.bitsandbytes__nf4"

            if k_weight not in keys:
                continue

            missing_req = [k for k in [k_absmax, k_qmap, k_qstate] if k not in keys]
            if missing_req:
                raise RuntimeError(f"Missing required quant components for {name}: {missing_req}")

            comps: Dict[str, torch.Tensor] = {
                "weight": f.get_tensor(k_weight),
                "absmax": f.get_tensor(k_absmax),
                "quant_map": f.get_tensor(k_qmap),
                "quant_state_data": f.get_tensor(k_qstate),
            }

            # optional double-quant
            k_nabs = f"{name}.weight.nested_absmax"
            k_nqmap = f"{name}.weight.nested_quant_map"
            if k_nabs in keys and k_nqmap in keys:
                comps["nested_absmax"] = f.get_tensor(k_nabs)
                comps["nested_quant_map"] = f.get_tensor(k_nqmap)

            param = reconstruct_params4bit_from_components(comps, device=device)
            module.weight = param
            loaded_count += 1

            # bias if exists
            k_bias = f"{name}.bias"
            if k_bias in keys and module.bias is not None:
                module.bias.data.copy_(f.get_tensor(k_bias).to(device))

            del comps, param

        print(f" Loaded {loaded_count} quantized linear layers (streaming safetensors)")

        # 2) Load remaining non-quant params/buffers via streaming in-place copy
        for k in keys:
            # skip linear4bit quant components + its bias (already handled)
            skip = False
            for ln in linear4bit_names:
                if k.startswith(ln + ".") and (is_quant_component(ln, k) or k == f"{ln}.bias"):
                    skip = True
                    break
            if skip:
                continue

            t = f.get_tensor(k)

            if k in param_map:
                dst = param_map[k]
                if dst.shape != t.shape:
                    raise RuntimeError(f"Shape mismatch for {k}: model {tuple(dst.shape)} vs ckpt {tuple(t.shape)}")
                dst.data.copy_(t.to(dst.device))
            elif k in buffer_map:
                dst = buffer_map[k]
                if dst.shape != t.shape:
                    raise RuntimeError(f"Shape mismatch for buffer {k}: model {tuple(dst.shape)} vs ckpt {tuple(t.shape)}")
                dst.data.copy_(t.to(dst.device))
            else:
                # ignore unexpected keys
                pass

    return model


def load_quantized_state(
    model: nn.Module,
    weights_path: str,
    layer_shapes: Dict[str, Tuple[int, int]],
    device: str = "cpu",
) -> nn.Module:
    """
    Load quantized weights into a model with bnb.Linear4bit layers.
    - safetensors: streaming safe_open (RAM-friendly)
    - pt/bin: fallback torch.load (may spike RAM)
    """
    if weights_path.endswith(".safetensors"):
        return _stream_load_safetensors(model, weights_path, layer_shapes, device=device)

    # Fallback for .pt (may use more RAM)
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)

    weight_components = defaultdict(dict)
    other_keys = {}
    quant_suffixes = [
        ".absmax",
        ".quant_map",
        ".nested_absmax",
        ".nested_quant_map",
        ".quant_state.bitsandbytes__nf4",
    ]

    for key, tensor in sd.items():
        base_key = None
        component = None

        if ".weight.absmax" in key:
            base_key = key.replace(".weight.absmax", "")
            component = "absmax"
        elif ".weight.quant_map" in key:
            base_key = key.replace(".weight.quant_map", "")
            component = "quant_map"
        elif ".weight.nested_absmax" in key:
            base_key = key.replace(".weight.nested_absmax", "")
            component = "nested_absmax"
        elif ".weight.nested_quant_map" in key:
            base_key = key.replace(".weight.nested_quant_map", "")
            component = "nested_quant_map"
        elif ".weight.quant_state.bitsandbytes__nf4" in key:
            base_key = key.replace(".weight.quant_state.bitsandbytes__nf4", "")
            component = "quant_state_data"
        elif key.endswith(".weight"):
            potential_base = key[:-7]
            has_quant_metadata = any(f"{potential_base}.weight{suffix}" in sd for suffix in quant_suffixes)
            if has_quant_metadata:
                base_key = potential_base
                component = "weight"
            else:
                other_keys[key] = tensor
                continue
        else:
            other_keys[key] = tensor
            continue

        if base_key and component:
            weight_components[base_key][component] = tensor

    loaded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            if name in weight_components and name in layer_shapes:
                components = weight_components[name]
                if "weight" in components:
                    param = reconstruct_params4bit_from_components(components, device=device)
                    module.weight = param
                    loaded_count += 1

                bias_key = f"{name}.bias"
                if bias_key in other_keys and module.bias is not None:
                    module.bias.data.copy_(other_keys[bias_key].to(device))

    if other_keys:
        model.load_state_dict(other_keys, strict=False)

    print(f" Loaded {loaded_count} quantized linear layers (pt fallback)")
    del sd
    return model


def load_quantized_model(model_dir: str, device: str = "cpu") -> WanModel:
    """
    Load a pre-quantized WanModel from a directory containing:
      - config.json
      - model.safetensors (preferred) or model.pt
    """
    model_dir = str(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Build base model
    model = build_model_from_config(config)
    model.eval()

    # Replace nn.Linear -> Linear4bit shells
    compute_dtype = torch.bfloat16
    if "compute_dtype" in config:
        if str(config["compute_dtype"]).lower() in ("float16", "fp16"):
            compute_dtype = torch.float16
        elif str(config["compute_dtype"]).lower() in ("bfloat16", "bf16"):
            compute_dtype = torch.bfloat16

    replaced, layer_shapes = replace_linears_with_bnb_nf4(
        model,
        compute_dtype=compute_dtype,
        compress_statistics=bool(config.get("double_quant", True)),
        quant_type=str(config.get("quant_type", "nf4")),
    )
    print(f" Replaced {replaced} Linear layers with Linear4bit shells")

    # Find weights file
    weights_path = None
    for cand in ["model.safetensors", "model.pt", "pytorch_model.bin"]:
        p = os.path.join(model_dir, cand)
        if os.path.exists(p):
            weights_path = p
            break
    if weights_path is None:
        raise FileNotFoundError(f"No weights found in {model_dir} (expected model.safetensors)")

    # Load weights (streaming for safetensors)
    model = load_quantized_state(model, weights_path, layer_shapes, device=device)
    model.to(device)
    return model
```

---

## 2) `generate_prequant_3090.py` (전체 복붙해서 새 파일로 저장)

> 기존 `generate_prequant.py`는 그대로 두고, 아래를 **새 파일로** 저장하세요.

```python
#!/usr/bin/env python3
"""
3090-friendly generate script:
- uses pre-quantized NF4 weights
- reduces VRAM peaks (inference_mode + in-place CFG + dtype fixes)
- avoids RAM spikes by lazy-loading ONLY one diffusion model at a time
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

# ---- Memory / attention backend tweaks (helpful on 24GB GPUs) ----
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass
# ------------------------------------------------------------------


class WanI2V_PreQuant:
    """Image-to-video pipeline using pre-quantized NF4 models."""

    def __init__(self, checkpoint_dir: str, device_id: int = 0, t5_cpu: bool = True):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = cfg
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = cfg.num_train_timesteps
        self.boundary = cfg.boundary
        self.param_dtype = cfg.param_dtype
        self.vae_stride = cfg.vae_stride
        self.patch_size = cfg.patch_size
        self.sample_neg_prompt = cfg.sample_neg_prompt

        # T5
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

        # VAE
        logger.info("Loading VAE...")
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, cfg.vae_checkpoint),
            device=self.device,
        )

        # Diffusion model dirs (lazy load: only one in RAM at a time)
        self.low_noise_dir = os.path.join(checkpoint_dir, cfg.low_noise_checkpoint + "_bnb_nf4")
        self.high_noise_dir = os.path.join(checkpoint_dir, cfg.high_noise_checkpoint + "_bnb_nf4")
        for d in [self.low_noise_dir, self.high_noise_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Pre-quantized model not found: {d}\n"
                    "Check you cloned the nf4 repo correctly."
                )

        self._loaded_name = None  # "low" or "high"
        self._model = None  # currently loaded model (only one)

        logger.info("Init complete. Diffusion models will be loaded lazily (one at a time).")

    def _unload_model(self):
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
            del self._model
            self._model = None
        self._loaded_name = None
        gc.collect()
        torch.cuda.empty_cache()

    def _load_model(self, which: str):
        assert which in ("low", "high")
        if self._loaded_name == which and self._model is not None:
            # ensure on GPU
            try:
                if next(self._model.parameters()).device.type != "cuda":
                    self._model.to(self.device)
            except StopIteration:
                self._model.to(self.device)
            return self._model

        # unload previous model first (RAM peak prevention)
        self._unload_model()

        model_dir = self.low_noise_dir if which == "low" else self.high_noise_dir
        logger.info(f"Loading {which} diffusion model from disk (CPU->GPU): {model_dir}")

        # load on CPU first (RAM-friendly load_prequant uses streaming)
        m = load_quantized_model(model_dir, device="cpu")
        m.to(self.device)

        self._model = m
        self._loaded_name = which
        return self._model

    def _prepare_model_for_timestep(self, t, boundary):
        which = "high" if t.item() >= boundary else "low"
        return self._load_model(which)

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
        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]
            guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale

        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img_tensor.shape[1:]
        aspect_ratio = h / w

        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
        )

        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # ✅ noise in param_dtype (not fp32)
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
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # encode text
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

        # camera cond (optional)
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

        # ✅ encode image without CPU round-trip; zeros in param_dtype on GPU
        img_resized = torch.nn.functional.interpolate(img_tensor[None], size=(h, w), mode="bicubic").transpose(0, 1)
        img_resized = img_resized.to(dtype=self.param_dtype)
        zeros = torch.zeros(3, F - 1, h, w, device=self.device, dtype=self.param_dtype)

        y = self.vae.encode([torch.concat([img_resized, zeros], dim=1)])[0]
        y = y.to(dtype=self.param_dtype)
        y = torch.concat([msk, y])

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

            arg_c = {"context": [context[0]], "seq_len": max_seq_len, "y": [y], "dit_cond_dict": dit_cond_dict}
            arg_null = {"context": context_null, "seq_len": max_seq_len, "y": [y], "dit_cond_dict": dit_cond_dict}

            for t in tqdm(timesteps, desc="Sampling"):
                model = self._prepare_model_for_timestep(t, boundary)
                s = guide_scale[1] if (not isinstance(guide_scale, float) and t.item() >= boundary) else (
                    guide_scale[0] if (not isinstance(guide_scale, float) and t.item() < boundary) else guide_scale
                )

                latent_model_input = [latent]
                timestep = t[None].to(self.device)

                # uncond then cond
                uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                cond = model(latent_model_input, t=timestep, **arg_c)[0]

                # ✅ in-place CFG (reduces peak allocs)
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

            # unload diffusion model before decode to free VRAM/RAM
            self._unload_model()

            videos = self.vae.decode([latent])

        del noise, latent, zeros, img_resized
        gc.collect()
        torch.cuda.synchronize()

        return videos[0]


def save_video(frames: torch.Tensor, output_path: str, fps: int = 16):
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
    parser.add_argument("--size", type=str, default="480*832", help="Output resolution H*W")
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

## 3) 실행 (복붙)

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

## 4) 만약 그래도 RAM OOM이면 (진짜 확인 1줄)

아래에서 “OOM-killer가 python을 죽였는지”가 바로 나옵니다.

```bash
dmesg -T | tail -n 80
```

그 출력만 붙여주시면, 다음 단계로는 **Linux overcommit / ulimit / xet 캐시 경로 변경(HF xet)** 쪽까지 포함해서 “당신 머신에 맞는” 최종 고정 세팅을 바로 드릴게요.

[1]: https://huggingface.co/cahlen/lingbot-world-base-cam-nf4/blob/main/load_prequant.py "load_prequant.py · cahlen/lingbot-world-base-cam-nf4 at main"
