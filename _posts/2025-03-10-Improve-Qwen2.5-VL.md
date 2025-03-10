아래는 원본 코드와 동일한 인터페이스를 유지하면서, 모델만 **Qwen2.5-VL**로 변경하여 수정한 전체 코드입니다.

### ✅ 최종 수정된 코드 (Qwen2.5-VL 모델 사용)

```python
# -*- coding: utf-8 -*-

import os
import re
import torch
import pickle
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--port", type=int, default=8080)
args = parser.parse_args()

print("Loaded model:", args.model_name_or_path)

# Processor 로딩 (Qwen2.5-VL)
processor = AutoProcessor.from_pretrained(
    args.model_name_or_path,
    do_image_splitting=False,
    min_pixels=256*28*28,
    max_pixels=1280*28*28
)

# Data Collator 정의 (기존과 동일한 기능)
class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids("")

    def __call__(self, example_list):
        texts = []
        images = []
        for example in example_list:
            image_list = example["images"]

            messages = []
            conversations = example["conversations"]
            for conv in conversations:
                item = {}
                item["role"] = conv["role"]
                raw_content = conv["content"]
                raw_content_split = re.split(r'()', raw_content)
                content_list = [
                    {"type": "image"} if seg == "" else {"type": "text", "text": seg}
                    for seg in raw_content_split if seg.strip()
                ]
                item = {"role": conv["role"], "content": content_list}
                messages.append(item)

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text.strip())
            images.append(image_list)

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        return batch

# 모델 로딩 (Qwen2.5-VL)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
).to(args.device)

data_collator = MyDataCollator(processor)

class InputData(BaseModel):
    id: str
    conversations: list
    images: str  # base64로 인코딩된 pickle.dumps(image_list)의 결과물

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(example: InputData):
    example_dict = example.dict()

    # 이미지 디코딩 및 PIL 이미지로 변환
    image_list_bin = base64.b64decode(example_dict["images"])
    image_list_pil = pickle.loads(image_list_bin)

    converted_example = {
        "id": example_dict["id"],
        "images": image_list_pil,
        "conversations": example_dict["conversations"]
    }

    batch = data_collator([converted_example])
    batch = {k: v.to(args.device) for k, v in batch.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **batch,
            max_new_tokens=256,
            min_new_tokens=3,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=True,
            temperature=1.2,
        )

    generated_text = processor.batch_decode(
        generated_ids[:, batch["input_ids"].size(1):],
        skip_special_tokens=True
    )[0]

    input_token_count = batch["input_ids"].size(1)
    output_token_count = generated_ids.size(1) - input_token_count

    return {
        "text": generated_text,
        "prompt_tokens": input_token_count,
        "completion_tokens": output_token_count
    }

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict_endpoint(example: InputData):
    return predict(example)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
```

---

### 📌 주요 수정 사항 요약:

- **모델 변경:** 기존의 `Idefics2ForConditionalGeneration`을 `Qwen2_5_VLForConditionalGeneration`으로 교체했습니다.
- **Processor 변경**: `AutoProcessor`를 Qwen2.5-VL 모델에 맞게 로드했습니다.
- **토큰 처리 방식 유지**: 기존과 동일하게 `` 토큰을 사용합니다.
- **입력 데이터 형식 유지**: 기존의 Base64 인코딩된 pickle 이미지 입력 방식을 유지했습니다.
- **FastAPI 인터페이스 유지**: 기존 인터페이스와 완전히 동일하게 유지했습니다.

---

### 🚩 테스트 방법 예시:

```python
import requests, pickle, base64
from PIL import Image

url = "http://localhost:8080/predict"

# 이미지 준비 및 인코딩 (예시 경로를 실제 경로로 변경하세요.)
img_paths = ["./img1.png", "./img2.png"]
pil_images = [Image.open(p).convert("RGB") for p in img_paths]
encoded_images_str = base64.b64encode(pickle.dumps(pil_images)).decode("utf-8")

example_payload = {
  "id": "test123",
  "images": encoded_images_str,
  "conversations": [
      {"role": "user", "content": " 사진 속에 무엇이 있나요?"},
      {"role": "assistant", "content": "Thought: 사진 분석 중입니다.\n\nAction: 개가 보입니다."}
  ]
}

response = requests.post(url, json=example_payload)
print(response.json())
```

---

## 🚩 주의사항 및 설치 방법 요약:

다음 패키지를 설치해야 합니다:

```bash
pip install transformers torch torchvision fastapi uvicorn pillow accelerate einops sentencepiece flash-attn --upgrade
```

Flash Attention 2를 사용하려면 다음과 같이 설치하세요:

```bash
pip install -U flash-attn --no-build-isolation
```
