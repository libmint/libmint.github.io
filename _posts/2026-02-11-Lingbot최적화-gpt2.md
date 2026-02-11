좋습니다. **RAM OOM이 `model.safetensors` 로딩 시점에서 나는 경우는 99% “가중치를 한 번에 dict로 통째로 읽어서(=추가 복사) 피크가 2배 이상 튀는 구조”** 때문입니다.
스왑 64GB 올려도, 리눅스 OOM-killer는 **순간 피크가 크면 그냥 죽여버려서** 스왑이 별 도움이 안 되는 케이스가 많아요.

여기서 확실히 먹히는 해결은 2개입니다:

1. **high/low 두 모델을 동시에 로드하지 않기 (lazy load)**
2. **safetensors를 “스트리밍(1 텐서씩)”으로 로드해서 RAM 피크를 없애기** ← 핵심

아래는 (2) **스트리밍 로딩을 강제로 쓰게 만드는 `load_prequant.py` 완성본**입니다. 그대로 복붙해서 교체하세요.

---

## 0) 실행 전 export (RAM OOM 대응용)

```bash
# 파이썬이 큰 메모리 블록을 덜 쪼개게 해서 RAM 파편화 완화(도움되는 경우 많음)
export MALLOC_ARENA_MAX=2

# (옵션) swap이 있어도 OOM-killer가 죽일 수 있어서, 로그 확인용
# dmesg -T | tail -n 50
```

`PYTORCH_CUDA_ALLOC_CONF`는 GPU 파편화 옵션이라 **RAM OOM에는 영향이 거의 없습니다.**

---

## 1) `load_prequant.py` 전체 교체 (스트리밍 safetensors 로딩)

프로젝트에 있는 `load_prequant.py`를 아래로 **통째로 교체**하세요.
(원래 파일 이름이 다르면, `from load_prequant import load_quantized_model` 를 쓰는 그 파일을 교체)

```python
import json
import os
from pathlib import Path

import torch

# safetensors streaming loader
from safetensors import safe_open

# NOTE:
# 아래 import는 원래 load_prequant.py가 하던 방식대로 유지해야 합니다.
# 당신 레포에 맞게 기존 load_prequant.py 안의 "모델 생성" 부분 import를 그대로 쓰세요.
# 예) from wan.modules.xxx import WanModel ...
# 여기서는 "기존에 쓰던 build_model_from_config"를 호출한다고 가정합니다.

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _find_weight_file(model_dir: str):
    # 보통 prequant 폴더에는 model.safetensors가 있음
    st = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(st):
        return st
    pt = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(pt):
        return pt
    pt2 = os.path.join(model_dir, "model.pt")
    if os.path.exists(pt2):
        return pt2
    raise FileNotFoundError(f"No weights found in {model_dir} (expected model.safetensors)")

def _build_model_from_dir(model_dir: str):
    """
    ✅ 여기가 중요: 기존 load_prequant.py가 하던 "config.json 읽고 모델 instantiate" 하는 로직을
    그대로 넣어야 합니다.

    - 당신 레포의 원래 load_prequant.py에 모델 생성 코드가 이미 있을 겁니다.
    - 그 부분을 이 함수 안으로 옮겨주세요.
    """
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found: {cfg_path}")

    cfg = _load_json(cfg_path)

    # ---- [여기를 기존 로직으로 교체] ----
    # 예시(가짜):
    # model = SomeModelClass.from_config(cfg)
    # -----------------------------------
    raise NotImplementedError(
        "You must move your original model-instantiation code here "
        "(from your old load_prequant.py)."
    )

def _stream_load_safetensors_into_model(model: torch.nn.Module, safetensors_path: str, device: str = "cpu"):
    """
    ✅ RAM 피크를 낮추는 핵심:
    - safetensors를 load_file()로 dict로 통째로 올리지 않음
    - safe_open으로 키를 돌며 텐서 1개씩 읽어서 param/buffer에 바로 copy
    => 'state_dict 전체 복사본'이 사라져서 피크가 크게 줄어듭니다.
    """
    # param/buffer lookup 테이블 (name -> Tensor)
    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())

    # 모델이 이미 파라미터 메모리를 잡고 있으니, 여기에 한 개씩 덮어쓰기
    with safe_open(safetensors_path, framework="pt", device=device) as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)  # CPU 텐서 1개만 메모리에 올라옴

            if k in param_map:
                # in-place copy (추가 메모리 최소화)
                dst = param_map[k]
                if dst.shape != t.shape:
                    raise RuntimeError(f"Shape mismatch for {k}: model {tuple(dst.shape)} vs ckpt {tuple(t.shape)}")
                dst.data.copy_(t)
            elif k in buffer_map:
                dst = buffer_map[k]
                if dst.shape != t.shape:
                    raise RuntimeError(f"Shape mismatch for buffer {k}: model {tuple(dst.shape)} vs ckpt {tuple(t.shape)}")
                dst.data.copy_(t)
            else:
                # 가끔 불필요 키가 있을 수 있음
                # print(f"[WARN] unexpected key: {k}")
                pass

    return model

def load_quantized_model(model_dir: str, device: str = "cpu"):
    """
    generate_prequant.py에서 호출하는 함수 시그니처를 유지합니다.

    - model을 instantiate 한 뒤
    - weights를 "스트리밍"으로 주입
    """
    model_dir = str(model_dir)
    weight_path = _find_weight_file(model_dir)

    # 1) 모델 instantiate (RAM: 모델 파라미터 메모리만)
    model = _build_model_from_dir(model_dir)
    model.eval()

    # 2) weights 로드
    if weight_path.endswith(".safetensors"):
        # ✅ 스트리밍 로딩(피크 최소)
        model = _stream_load_safetensors_into_model(model, weight_path, device="cpu")
    else:
        # bin/pt는 구조상 스트리밍이 어렵고 보통 피크가 큼
        # 그래도 map_location=cpu로 RAM 피크를 조금 줄임
        sd = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        del sd

    # 3) 원하는 디바이스로 이동(여기서는 generate 쪽에서 스왑하니 보통 "cpu" 유지 권장)
    model.to(device)
    return model
```

### ⭐️ 딱 1군데만 “당신 레포에 맞게” 바꿔야 하는 곳

위 코드에서 `_build_model_from_dir()` 안의 `NotImplementedError` 부분을
**당신의 원래 `load_prequant.py`에서 “모델 instantiate 하던 코드”로 그대로 옮기면 끝**입니다.

> 즉, “모델 만드는 법”은 레포마다 다르니 그 부분만 원래대로 유지하고,
> **가중치 로딩만 스트리밍 방식으로 바꾼다**가 핵심입니다.

---

## 2) (추가로 강추) high/low 동시 로딩 금지(lazy load)

RAM OOM이 로딩에서 났다면, 스트리밍만으로도 해결되는 경우가 많지만,
더 안전하게 하려면 **generate_prequant.py에서 high/low를 둘 다 __init__에서 로드하지 말고**, timestep에서 하나씩만 로드하게 바꾸는 게 베스트입니다(제가 앞에서 준 lazy-load 방식).

---

## 3) 바로 확인할 것 (OOM-killer 로그)

실행 직후 아래 보면 “왜 죽였는지”가 정확히 나옵니다.

```bash
dmesg -T | tail -n 80
```

여기에 `Out of memory: Killed process ... python` 같이 뜨면 RAM OOM이 맞고, 위 패치가 정답 루트입니다.

---

## 다음 액션 (질문 없이 바로 진행 가능)

당장 당신 쪽에서 필요한 건 `_build_model_from_dir()`에 들어갈 “모델 생성 코드”뿐입니다.

* 지금 프로젝트의 **기존 `load_prequant.py` 내용 전체를 여기 붙여주시면**,
  제가 그걸 기반으로 **위 스트리밍 로더가 “완전히 합쳐진 최종 load_prequant.py 완성본”**을 만들어서 그대로 복붙 가능하게 드릴게요.

(지금처럼 RAM OOM이면, 이 스트리밍 로딩으로 해결되는 확률이 매우 높습니다.)
