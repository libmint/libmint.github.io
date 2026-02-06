RTX 3090(24GB VRAM)은 **LingBot-World-Base(14B)** 모델을 돌리기에 매우 훌륭한 환경이지만, 모델의 원래 크기가 24GB를 거의 다 채우기 때문에 **양자화(Quantization)**와 **플래시 어텐션(Flash Attention)** 설정이 필수입니다.

현시점(2026년) 기준으로 가장 안정적인 로컬 설치 및 실행 방법을 단계별로 정리해 드립니다.

---

### 1. 사전 준비 (Environment)

가장 먼저 Python 환경과 CUDA 설정을 마쳐야 합니다. (Ubuntu 22.04 이상 권장)

* **Python:** 3.10 이상
* **CUDA:** 12.1 또는 12.4
* **PyTorch:** 2.4.0 이상

```bash
# 가상환경 생성 및 활성화
conda create -n lingbot python=3.10 -y
conda activate lingbot

# PyTorch 설치 (CUDA 12.1 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

---

### 2. 소스코드 복사 및 패키지 설치

LingBot-World 공식 저장소를 가져오고 필요한 라이브러리를 설치합니다.

```bash
git clone https://github.com/robbyant/lingbot-world.git
cd lingbot-world

# 필수 패키지 설치
pip install -r requirements.txt

# RTX 3090 성능 최적화를 위한 Flash Attention 설치 (필수)
pip install flash-attn --no-build-isolation

```

---

### 3. 모델 웨이트 다운로드 (Hugging Face)

3090 사용자는 **Base-Cam** 또는 **Base-Act** 모델 중 하나를 선택하면 됩니다. 처음에는 시점 제어가 가능한 Cam 모델을 추천합니다.

```bash
pip install "huggingface_hub[cli]"

# 14B 모델 다운로드 (약 30GB 이상의 여유 하드 디스크 공간 필요)
huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./models/base-cam

```

---

### 4. RTX 3090을 위한 실행 최적화 (가장 중요)

3090의 24GB VRAM 안에 모델(14B)과 VAE, 그리고 시뮬레이션 버퍼를 모두 넣으려면 **`--fp8`** 또는 **`--quant 4bit`** 옵션을 사용해야 합니다.

#### A. 텍스트/이미지로 세계 생성 (Inference)

```bash
# FP8 정밀도로 실행하여 VRAM 사용량을 약 16~18GB로 제한
python generate.py \
    --model_path ./models/base-cam \
    --prompt "A neon-lit cyberpunk street in Seoul, raining" \
    --fp8 \
    --infer_steps 50 \
    --save_path ./outputs/test_world.mp4

```

#### B. 실시간 인터랙션 모드 (Simulation)

사용자 키보드 입력(WASD)을 받아 실시간으로 세계를 그리는 모드입니다.

```bash
python interact.py \
    --model_path ./models/base-cam \
    --fp8 \
    --enable_tiled_vae \
    --resolution 480p

```

> **Tip:** 3090에서는 720p보다 **480p**로 시작하는 것이 끊김 없는(Real-time) 경험에 유리합니다.

---

### 5. 3090 사용자를 위한 트러블슈팅

1. **Out of Memory (OOM) 발생 시:** `--enable_tiled_vae` 옵션을 추가하세요. VRAM 부하가 큰 비디오 디코딩 과정을 쪼개서 처리합니다.
2. **속도가 너무 느릴 때:** `--xformers` 옵션을 설치하고 추가하면 메모리 효율이 더 좋아집니다.
3. **정교한 물리 시뮬레이션이 깨질 때:** 4-bit 양자화보다는 **FP8(8-bit)**을 사용하세요. 3090은 FP8 연산 성능이 우수하여 품질 저하 없이 속도를 챙길 수 있습니다.

---

### 다음 단계 제안

혹시 특정 사진(예: 직접 찍은 서울 거리 사진)을 넣어서 그 안을 돌아다니는 **Image-to-World** 기능을 바로 테스트해보고 싶으신가요? 관련 파라미터 설정을 안내해 드릴 수 있습니다.
