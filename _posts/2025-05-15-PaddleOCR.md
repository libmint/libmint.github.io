

---

## ✅ 완성된 `localizer.py`

```python
import os
import json
import uuid
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

# PaddleOCR 엔진 초기화: 한국어 + 영어 동시 인식
ocr_engine = PaddleOCR(use_angle_cls=True, lang='korean')  # 'korean'은 영어도 포함합니다

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def extract_elements_with_ocr(image_path):
    """
    PaddleOCR을 이용해 이미지 내의 텍스트 요소를 추출합니다.
    반환 형식은 id, name(텍스트), bbox 좌표 포함
    """
    result = ocr_engine.ocr(image_path, cls=True)

    elements = []
    for line in result:
        for box, (text, conf) in line:
            if text.strip():
                x_min = int(min([p[0] for p in box]))
                y_min = int(min([p[1] for p in box]))
                x_max = int(max([p[0] for p in box]))
                y_max = int(max([p[1] for p in box]))
                elements.append({
                    "id": len(elements) + 1,
                    "name": text.strip(),
                    "bbox": [x_min, y_min, x_max, y_max]
                })
    return elements

def draw_elements_on_image(image_path, elements, save_dir):
    """
    요소들을 사각형 박스로 이미지에 표시하고, 요소 ID를 숫자로 기록합니다.
    """
    os.makedirs(save_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()

    for e in elements:
        draw.rectangle(e["bbox"], outline="red", width=2)
        draw.text((e["bbox"][0], e["bbox"][1] - 18), str(e["id"]), fill="red", font=font)

    output_path = os.path.join(save_dir, f"som_{uuid.uuid4().hex}.png")
    image.save(output_path)
    return output_path

def localize_screen_with_ocr(image_path, save_dir, output_json):
    """
    이미지로부터 UI 요소를 추출하고 SOM 이미지를 생성한 후, JSON으로 결과 저장
    """
    elements = extract_elements_with_ocr(image_path)
    marked_path = draw_elements_on_image(image_path, elements, save_dir)
    result = {
        "original_screenshot_path": image_path,
        "som_screenshot_path": marked_path,
        "elements": elements
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result
```

---

## ✅ 설치 안내 (필수)

```bash
pip install paddleocr
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

※ Windows 사용자는 `paddlepaddle==2.5.0` 등 직접 버전 지정이 필요할 수 있습니다.

---

## 🧪 출력 예시

* `som_output/som_xxxxx.png`: 각 요소에 숫자가 표시된 이미지
* `localized_xxxxx.json`:

```json
{
  "id": 1,
  "name": "설정",
  "bbox": [140, 100, 220, 135]
}
```

---

