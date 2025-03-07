UIX-Qwen2 모델로 추론하는 코드

```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image

# 모델과 프로세서 로드
model = Qwen2VLForConditionalGeneration.from_pretrained("neulab/UIX-Qwen2", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("neulab/UIX-Qwen2")

# 로컬 이미지 파일 로드
image_path = "path/to/your/local/image.jpg"
image = Image.open(image_path)

# 메시지 구성
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {"type": "text", "text": "이 이미지에 대해 설명해주세요."}
        ]
    }
]

# 추론 준비
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
)

# GPU로 입력 이동
inputs = inputs.to(model.device)

# 추론 실행
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# 결과 디코딩
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0])
```
