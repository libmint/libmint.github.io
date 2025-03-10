`hf_online_server.py` 파일을 `Qwen2.5-VL-7B-Instruct` 모델을 사용하는 코드로 변경하려면, 모델 로드 및 처리 부분을 수정해야 합니다. 아래는 수정된 코드 예제입니다.

### 수정된 코드

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# 모델 및 프로세서 로드
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # FP16으로 메모리 최적화
    device_map="auto"          # GPU 자동 할당
)

processor = AutoProcessor.from_pretrained(model_name)

# 입력 데이터 준비
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# 입력 데이터 처리
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

# 추론 실행
generated_ids = model.generate(**inputs, max_new_tokens=128)

# 결과 디코딩
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)
```

### 변경 사항 설명

1. **모델 및 프로세서 로드**: `Qwen2_5_VLForConditionalGeneration` 모델과 `AutoProcessor`를 사용하여 `Qwen2.5-VL-7B-Instruct` 모델을 로드합니다.

2. **입력 데이터 처리**: `process_vision_info` 함수를 사용하여 이미지 입력을 처리합니다.

3. **추론 실행**: 모델에 입력을 제공하고 결과를 생성합니다.

4. **결과 디코딩**: 생성된 토큰 ID를 텍스트로 디코딩합니다.

이 코드는 `Qwen2.5-VL-7B-Instruct` 모델을 사용하여 이미지와 텍스트 입력을 처리하고 결과를 생성합니다. 

### 추가 팁

- **라이브러리 설치**: `qwen-vl-utils` 라이브러리를 설치해야 합니다. 이는 `pip install qwen-vl-utils[decord]==0.0.8` 명령어로 설치할 수 있습니다[2].
- **환경 설정**: `transformers` 라이브러리를 최신 버전으로 설치해야 합니다. 이는 `pip install git+https://github.com/huggingface/transformers accelerate` 명령어로 설치할 수 있습니다[2].

from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorWithPadding
import torch

# 모델 및 프로세서 로드
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # FP16으로 메모리 최적화
    device_map="auto"          # GPU 자동 할당
)

processor = AutoProcessor.from_pretrained(model_name)

# 추가 토큰 등록
special_tokens_dict = {"additional_special_tokens": ["<image>"]}
num_added_toks = processor.tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(processor.tokenizer))

# 커스텀 데이터 콜레이터
class MyDataCollator(DataCollatorWithPadding):
    def __init__(self, processor, padding=True, max_length=None, pad_to_multiple_of=None):
        super().__init__(processor.tokenizer, padding=padding, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of)
        self.processor = processor

    def __call__(self, features):
        texts = []
        images = []
        for feature in features:
            text = feature["text"]
            image = feature.get("image", None)
            
            # 텍스트에 <image> 토큰 추가
            if image is not None:
                text += " <image>"
            
            texts.append(text)
            images.append(image)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )

        # 레이블 생성 (필요에 따라 수정)
        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels

        return batch

# 데이터 콜레이터 생성
data_collator = MyDataCollator(processor)

# 예시 데이터
data = [
    {"text": "Describe this image.", "image": "file:///path/to/image.jpg"},
    {"text": "Another text without image."}
]

# 데이터 콜레이터 사용
batch = data_collator(data)

# 추론 실행
generated_ids = model.generate(**batch, max_new_tokens=128)

# 결과 디코딩
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch["input_ids"], generated_ids)
]

output_text = processor.tokenizer.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)

