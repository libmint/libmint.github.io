

---

## ✅ 전체 프로젝트 구조

```
gui_project/
├── main.py
├── model_utils.py
├── localizer.py
├── aw_data_converter/
│   └── data_utils.py
├── gui_tasks/
│   ├── base_task.py
│   ├── element_description_task.py
│   ├── dense_captioning_task.py
│   ├── qa_task.py
│   ├── set_of_mark_task.py
│   └── state_transition_captioning_task.py
```

---

## 🔹 `main.py`

✅ OCR 기반 localizer 사용 포함, 각 task에 elements 전달
```
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
    for img in image_paths:
        print(f"[+] Processing {img}")

        # Step 1: Use Localizer if enabled
        elements = []
        marked_image = img
        if args.use_localizer:
            from localizer import localize_screen_with_ocr
            local_output_json = f"localizer_{os.path.basename(img)}.json"
            localization = localize_screen_with_ocr(img, "som_output", local_output_json)
            elements = localization["elements"]
            marked_image = localization["som_screenshot_path"]

        result = {
            "screenshot": marked_image,
            "metadata": {},
            "perception_tasks": {},
            "created_at": datetime.utcnow().isoformat()
        }

        # Step 2: Perform Tasks
        if args.all or args.task == "element":
            if args.use_localizer and elements:
                result["perception_tasks"]["element_description"] = elements
            else:
                element_raw = ElementDescriptionTask(marked_image).run(request_model)
                try:
                    result["perception_tasks"]["element_description"] = json.loads(element_raw)
                except:
                    result["perception_tasks"]["element_description"] = {"error": "Invalid JSON"}

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

        results.append(result)

    # Save final output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results if len(results) > 1 else results[0], f, indent=2, ensure_ascii=False)
    print(f"[✓] Results saved: {args.output}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Run
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

    # Convert
    convert = subparsers.add_parser("convert")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)

    # Fill
    fill = subparsers.add_parser("fill")
    fill.add_argument("--input", required=True)
    fill.add_argument("--image_dir", required=True)
    fill.add_argument("--output", required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == "run":
        run_selected_tasks(args)
    elif args.mode == "convert":
        convert_existing_data_with_prompts(args.input, args.output)
    elif args.mode == "fill":
        fill_unknown_answers(args.input, args.image_dir, args.output)

```

---

## 🔹 `model_utils.py`

```python
import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_ENDPOINT = os.getenv("API_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def request_model(system_prompt, user_prompt, image_data_list):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}] + image_data_list}
    ]
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3
    }
    response = requests.post(API_ENDPOINT, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
```

---

## 🔹 `localizer.py`

```python
import os
import json
import uuid
from PIL import Image, ImageDraw
import pytesseract

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def extract_elements_with_ocr(image_path):
    image = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    elements = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            elements.append({"name": text, "bbox": [x, y, x + w, y + h]})
    return elements

def draw_elements_on_image(image_path, elements, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for e in elements:
        draw.rectangle(e["bbox"], outline="red", width=2)
        draw.text((e["bbox"][0], e["bbox"][1] - 15), e["name"], fill="red")
    output_path = os.path.join(save_dir, f"som_{uuid.uuid4().hex}.png")
    image.save(output_path)
    return output_path

def localize_screen_with_ocr(image_path, save_dir, output_json):
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

## 🔹 `aw_data_converter/data_utils.py`

✅ 기존 포맷 변환 및 unknown 채움 로직
👉 [이전 답변에서 제공됨](https://chat.openai.com/share)

---

## 🔹 `gui_tasks/base_task.py`

```python
from abc import ABC, abstractmethod
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class BaseGUITask(ABC):
    def __init__(self, image_path, elements=None):
        self.image_path = image_path
        self.image_data = encode_image(image_path)
        self.elements = elements or []

    @abstractmethod
    def task_name(self): pass
    @abstractmethod
    def system_prompt(self): pass
    @abstractmethod
    def user_prompt(self): pass

    def run(self, request_model_func):
        image_data = [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{self.image_data}"}
        }]
        return request_model_func(self.system_prompt(), self.user_prompt(), image_data)
```

---

## 🔹 `element_description_task.py`

```python
from gui_tasks.base_task import BaseGUITask

class ElementDescriptionTask(BaseGUITask):
    def task_name(self): return "element_description"
    def system_prompt(self): return "You are a UI screen analyzer that returns UI elements."
    def user_prompt(self):
        return (
            "Return the list of UI elements detected in this screen including the following fields:\n"
            "- 'type', 'text', 'function', 'visual', 'position', 'box'\n"
            "Use the elements provided below as guidance:\n\n"
            + "\n".join(f"{e['name']}: {e['bbox']}" for e in self.elements)
        )
```

---

## 🔹 `dense_captioning_task.py`

```python
from gui_tasks.base_task import BaseGUITask

class DenseCaptioningTask(BaseGUITask):
    def task_name(self): return "dense_captioning"
    def system_prompt(self): return "You are a GUI captioning assistant."
    def user_prompt(self):
        prompt = "Describe this GUI screen in detail including key UI components."
        if self.elements:
            prompt += "\nRelevant elements include:\n"
            for e in self.elements:
                prompt += f"- {e['name']} at {e['bbox']}\n"
        return prompt
```

---

## 🔹 `qa_task.py`

```python
from gui_tasks.base_task import BaseGUITask

class QATask(BaseGUITask):
    def __init__(self, image_path, question, elements):
        super().__init__(image_path, elements)
        self.question = question

    def task_name(self): return "qa"
    def system_prompt(self): return "You are a screen-level question answering assistant."

    def user_prompt(self):
        prompt = f"For each UI element below, answer: '{self.question}'\n\n"
        for e in self.elements:
            prompt += f"Element: {e['name']} at {e['bbox']}\n"
        return prompt
```

---

## 🔹 `set_of_mark_task.py`

```python
from gui_tasks.base_task import BaseGUITask

class SetOfMarkTask(BaseGUITask):
    def task_name(self): return "set_of_mark"
    def system_prompt(self): return "You are a GUI element identifier for marked elements."

    def user_prompt(self):
        prompt = "Identify visually marked or highlighted elements.\n"
        if self.elements:
            prompt += "\nProvided elements:\n"
            for e in self.elements:
                prompt += f"- {e['name']} at {e['bbox']}\n"
        return prompt
```

---

## 🔹 `state_transition_captioning_task.py`

```python
from gui_tasks.base_task import encode_image

class StateTransitionCaptioningTask:
    def __init__(self, before_image, after_image):
        self.before_image = encode_image(before_image)
        self.after_image = encode_image(after_image)

    def task_name(self): return "state_transition_captioning"

    def run(self, request_model_func):
        image_data = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.before_image}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.after_image}"}}
        ]
        return request_model_func(
            "You are a GUI transition analyzer.",
            "Describe what changes occurred between the two screenshots.",
            image_data
        )
```

---
main update
```
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
    for img in image_paths:
        print(f"[+] Processing {img}")

        # Step 1: Use Localizer if enabled
        elements = []
        marked_image = img
        if args.use_localizer:
            from localizer import localize_screen_with_ocr
            local_output_json = f"localizer_{os.path.basename(img)}.json"
            localization = localize_screen_with_ocr(img, "som_output", local_output_json)
            elements = localization["elements"]
            marked_image = localization["som_screenshot_path"]

        result = {
            "screenshot": marked_image,
            "metadata": {},
            "perception_tasks": {},
            "created_at": datetime.utcnow().isoformat()
        }

        # Step 2: Perform Tasks
        if args.all or args.task == "element":
            if args.use_localizer and elements:
                result["perception_tasks"]["element_description"] = elements
            else:
                element_raw = ElementDescriptionTask(marked_image).run(request_model)
                try:
                    result["perception_tasks"]["element_description"] = json.loads(element_raw)
                except:
                    result["perception_tasks"]["element_description"] = {"error": "Invalid JSON"}

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

        results.append(result)

    # Save final output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results if len(results) > 1 else results[0], f, indent=2, ensure_ascii=False)
    print(f"[✓] Results saved: {args.output}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Run mode
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

    # Convert mode
    convert = subparsers.add_parser("convert")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)

    # Fill mode
    fill = subparsers.add_parser("fill")
    fill.add_argument("--input", required=True)
    fill.add_argument("--image_dir", required=True)
    fill.add_argument("--output", required=True)

    # Localize mode
    localize = subparsers.add_parser("localize")
    localize.add_argument("--image_dir", required=True, help="Directory of screenshots to localize")
    localize.add_argument("--output_dir", default="som_output", help="Directory for marked images")
    localize.add_argument("--json_dir", default="som_json", help="Directory for element JSONs")

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
        from localizer import localize_screen_with_ocr
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.json_dir, exist_ok=True)

        image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img_name in image_files:
            img_path = os.path.join(args.image_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            json_path = os.path.join(args.json_dir, f"localized_{base_name}.json")
            print(f"[+] Localizing {img_path}")
            localize_screen_with_ocr(img_path, args.output_dir, json_path)

        print(f"[✓] Localized {len(image_files)} images.")

```

localizer update
```
import os
import json
import uuid
from PIL import Image, ImageDraw, ImageFont
import pytesseract

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def extract_elements_with_ocr(image_path):
    """OCR을 사용해 텍스트 요소를 감지하고 bounding box 정보 추출"""
    image = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    elements = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            elements.append({
                "id": len(elements) + 1,
                "name": text,
                "bbox": [x, y, x + w, y + h]
            })
    return elements

def draw_elements_on_image(image_path, elements, save_dir):
    """마킹된 SOM 이미지 생성 (숫자 표시 포함)"""
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
    """단일 스크린샷에 대해 요소 추출 + 마킹 이미지 저장 + JSON 저장"""
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
