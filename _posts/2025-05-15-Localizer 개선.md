

---
## Localizer
```
import os
import json
import uuid
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

# OCR 초기화 (한국어 + 영어 동시 인식)
ocr_engine = PaddleOCR(use_angle_cls=True, lang='korean')

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def extract_elements_with_ocr(image_path):
    """
    PaddleOCR을 사용해 이미지 내의 텍스트 요소 추출
    각 요소는 ID, name(텍스트), bbox 좌표 포함
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
    요소들의 bounding box 및 ID를 이미지에 표시 후 저장
    마킹된 이미지는 som_output 디렉터리에 저장
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

def process_image(image_path, som_output_dir):
    """
    단일 이미지 처리:
    - OCR 요소 추출
    - SOM 이미지 저장
    - JSON은 이미지와 동일한 폴더에 저장
    """
    elements = extract_elements_with_ocr(image_path)
    marked_path = draw_elements_on_image(image_path, elements, som_output_dir)

    json_data = {
        "image_path": image_path,
        "elements": elements
    }

    json_path = os.path.splitext(image_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return json_path, marked_path

def process_directory_recursively(root_dir, som_output_dir="som_output"):
    """
    시작 폴더부터 모든 하위 폴더의 이미지를 재귀적으로 처리
    - 각 이미지의 JSON은 해당 폴더에 저장
    - SOM 이미지는 지정 폴더에 저장
    """
    supported_ext = (".png", ".jpg", ".jpeg")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(supported_ext):
                image_path = os.path.join(root, file)
                print(f"[+] Processing {image_path}")
                json_path, marked_path = process_image(image_path, som_output_dir)
                print(f"[✓] Saved JSON to {json_path}, marked image to {marked_path}")

```

## ✅ 완성된 `main.py`

```python
import argparse
import os
import json
from datetime import datetime

from model_utils import request_model
from aw_data_converter.data_utils import convert_existing_data_with_prompts, fill_unknown_answers
from gui_tasks.element_description_task import ElementDescriptionTask
from gui_tasks.dense_captioning_task import DenseCaptioningTask
from gui_tasks.qa_task import QATask
from gui_tasks.set_of_mark_task import SetOfMarkTask
from gui_tasks.state_transition_captioning_task import StateTransitionCaptioningTask


def run_selected_tasks(args):
    image_paths = []
    if args.image_dir:
        image_paths = [os.path.join(args.image_dir, f)
                       for f in os.listdir(args.image_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    else:
        image_paths = [args.image]

    results = []
    is_output_dir = os.path.isdir(args.output) or args.output.endswith("/")

    for img in image_paths:
        print(f"[+] Processing {img}")
        elements = []
        marked_image = img

        if args.use_localizer:
            from localizer import localize_screen_with_ocr
            base_name = os.path.splitext(os.path.basename(img))[0]
            local_output_json = f"localizer_{base_name}.json"
            localization = localize_screen_with_ocr(img, "som_output", local_output_json)
            elements = localization["elements"]
            marked_image = localization["som_screenshot_path"]

        result = {
            "screenshot": marked_image,
            "metadata": {},
            "perception_tasks": {},
            "created_at": datetime.utcnow().isoformat()
        }

        if args.all or args.task == "element":
            if args.use_localizer and elements:
                result["perception_tasks"]["element_description"] = elements
            else:
                raw = ElementDescriptionTask(marked_image).run(request_model)
                result["perception_tasks"]["element_description"] = json.loads(raw)

        if args.all or args.task == "caption":
            caption = DenseCaptioningTask(marked_image, elements).run(request_model)
            result["perception_tasks"]["dense_captioning"] = caption

        if args.all or args.task == "qa":
            qa = QATask(marked_image, args.question, elements).run(request_model)
            result["perception_tasks"]["qa"] = {"question": args.question, "answer": qa}

        if args.all or args.task == "mark":
            marks = SetOfMarkTask(marked_image, elements).run(request_model)
            result["perception_tasks"]["set_of_mark"] = marks

        if (args.all or args.task == "state") and args.before and args.after:
            state_caption = StateTransitionCaptioningTask(args.before, args.after).run(request_model)
            result["perception_tasks"]["state_transition_captioning"] = {
                "before_image": args.before,
                "after_image": args.after,
                "caption": state_caption
            }

        if is_output_dir:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img))[0]
            task_prefix = args.task if not args.all else "all"
            out_path = os.path.join(args.output, f"{task_prefix}_{base_name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[✓] Saved: {out_path}")
        else:
            results.append(result)

    if not is_output_dir:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2, ensure_ascii=False)
        print(f"[✓] Results saved: {args.output}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--image")
    run.add_argument("--image_dir")
    run.add_argument("--before")
    run.add_argument("--after")
    run.add_argument("--question", default="What is the main purpose of this screen?")
    run.add_argument("--output", required=True)
    run.add_argument("--use_localizer", action="store_true", help="Use OCR-based element localizer")
    group = run.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", choices=["element", "caption", "qa", "mark", "state"])
    group.add_argument("--all", action="store_true")

    convert = subparsers.add_parser("convert")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)

    fill = subparsers.add_parser("fill")
    fill.add_argument("--input", required=True)
    fill.add_argument("--image_dir", required=True)
    fill.add_argument("--output", required=True)

    localize = subparsers.add_parser("localize")
    localize.add_argument("--image_dir", required=True, help="Directory of screenshots (recursive)")
    localize.add_argument("--output_dir", default="som_output", help="Where to save SOM marked images")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.mode == "run":
        run_selected_tasks(args)

    elif args.mode == "convert":
        convert_existing_data_with_prompts(args.input, args.output)

    elif args.mode == "fill":
        fill_unknown_answers(args.input, args.image_dir, args.output)

    elif args.mode == "localize":
        from localizer import process_directory_recursively
        process_directory_recursively(args.image_dir, args.output_dir)
```

---

## ✅ 실행 예시

```bash
python main.py localize \
  --image_dir gui_screens/ \
  --output_dir som_output/
```

이 명령은 다음을 수행합니다:

* `gui_screens/` 하위 폴더까지 모두 검색
* 이미지마다:

  * 마킹된 SOM 이미지를 `som_output/`에 저장
  * `same_folder/image.json` 파일로 요소 정보 저장

---
