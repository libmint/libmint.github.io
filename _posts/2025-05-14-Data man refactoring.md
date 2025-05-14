
---

## ✅ 프로젝트 구조

```
gui_project/
│
├── main.py
├── gui_tasks/
│   ├── base_task.py
│   ├── element_description_task.py
│   ├── dense_captioning_task.py
│   ├── qa_task.py
│   ├── set_of_mark_task.py
│   └── state_transition_captioning_task.py
```

---

## ✅ `main.py`

```python
# main.py
import os
import json
import base64
import requests
import argparse
from datetime import datetime
from dotenv import load_dotenv

from gui_tasks.element_description_task import ElementDescriptionTask
from gui_tasks.dense_captioning_task import DenseCaptioningTask
from gui_tasks.qa_task import QATask
from gui_tasks.set_of_mark_task import SetOfMarkTask
from gui_tasks.state_transition_captioning_task import StateTransitionCaptioningTask

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_ENDPOINT = os.getenv("API_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

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

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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
        result = {
            "screenshot": img,
            "metadata": {},
            "perception_tasks": {},
            "created_at": datetime.utcnow().isoformat()
        }

        if args.all or args.task == "element":
            element_raw = ElementDescriptionTask(img).run(request_model)
            try:
                result["perception_tasks"]["element_description"] = json.loads(element_raw)
            except:
                result["perception_tasks"]["element_description"] = {"error": "Invalid JSON"}

        if args.all or args.task == "caption":
            caption = DenseCaptioningTask(img).run(request_model)
            result["perception_tasks"]["dense_captioning"] = caption

        if args.all or args.task == "qa":
            qa = QATask(img, args.question).run(request_model)
            result["perception_tasks"]["qa"] = {
                "question": args.question,
                "answer": qa
            }

        if args.all or args.task == "mark":
            marks = SetOfMarkTask(img).run(request_model)
            result["perception_tasks"]["set_of_mark"] = marks

        if (args.all or args.task == "state") and args.before and args.after:
            transition = StateTransitionCaptioningTask(args.before, args.after).run(request_model)
            result["perception_tasks"]["state_transition_captioning"] = {
                "before_image": args.before,
                "after_image": args.after,
                "caption": transition
            }

        results.append(result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results if len(results) > 1 else results[0], f, indent=2, ensure_ascii=False)
    print(f"[✓] Results saved: {args.output}")

def convert_existing_data_with_prompts(input_path, output_path, screen_width=1080, screen_height=1920):
    with open(input_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)

    new_data = []
    for entry in old_data:
        elements = []
        for e in entry["elements"]:
            x1, y1, x2, y2 = e["bbox"]
            abs_box = [int(x1 * screen_width), int(y1 * screen_height),
                       int(x2 * screen_width), int(y2 * screen_height)]
            elements.append({
                "type": e["data_type"].split('.')[-1].lower(),
                "text": e["instruction"],
                "function": "unknown",
                "visual": "unknown",
                "position": "unknown",
                "box": abs_box
            })
        new_data.append({
            "screenshot": entry["img_filename"],
            "metadata": {},
            "perception_tasks": {
                "element_description": elements,
                "dense_captioning": {"prompt": "Describe all visible elements and layout.", "answer": "unknown"},
                "qa": {"question": "What is the main purpose of this screen?", "answer": "unknown"},
                "set_of_mark": {"prompt": "Identify marked elements.", "answer": "unknown"},
                "state_transition_captioning": None
            },
            "created_at": "converted"
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    print(f"[✓] Converted and saved: {output_path}")

def fill_unknown_answers(data_path, image_dir, output_path):
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for entry in dataset:
        image_path = os.path.join(image_dir, entry["screenshot"])
        if not os.path.exists(image_path):
            continue
        b64img = encode_image(image_path)
        tasks = entry["perception_tasks"]

        if tasks["dense_captioning"]["answer"] == "unknown":
            tasks["dense_captioning"]["answer"] = request_model("You are a GUI captioning assistant.",
                                                                tasks["dense_captioning"]["prompt"],
                                                                [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64img}"}}])
        if tasks["qa"]["answer"] == "unknown":
            tasks["qa"]["answer"] = request_model("You are a screen-level question answering assistant.",
                                                  tasks["qa"]["question"],
                                                  [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64img}"}}])
        if tasks["set_of_mark"]["answer"] == "unknown":
            tasks["set_of_mark"]["answer"] = request_model("You are a GUI element identifier.",
                                                           tasks["set_of_mark"]["prompt"],
                                                           [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64img}"}}])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"[✓] Answers filled and saved: {output_path}")

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

## ✅ `gui_tasks/` 디렉터리 코드

### 1. `base_task.py`

```python
from abc import ABC, abstractmethod
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class BaseGUITask(ABC):
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_data = encode_image(image_path)

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

### 2. `element_description_task.py`

```python
from gui_tasks.base_task import BaseGUITask

class ElementDescriptionTask(BaseGUITask):
    def task_name(self): return "element_description"
    def system_prompt(self): return "You are a UI screen analyzer that returns all visible UI elements in JSON format."
    def user_prompt(self):
        return (
            "Return a complete JSON array of all visible UI elements in the screenshot.\n"
            "Each object must include:\n"
            "- 'type'\n"
            "- 'text'\n"
            "- 'function'\n"
            "- 'visual'\n"
            "- 'position'\n"
            "- 'box'"
        )
```

### 3. `dense_captioning_task.py`

```python
from gui_tasks.base_task import BaseGUITask

class DenseCaptioningTask(BaseGUITask):
    def task_name(self): return "dense_captioning"
    def system_prompt(self): return "You are a GUI captioning assistant."
    def user_prompt(self): return "Describe all visible elements and layout in this UI screenshot."
```

### 4. `qa_task.py`

```python
from gui_tasks.base_task import BaseGUITask

class QATask(BaseGUITask):
    def __init__(self, image_path, question):
        super().__init__(image_path)
        self.question = question
    def task_name(self): return "qa"
    def system_prompt(self): return "You are a screen-level question answering assistant."
    def user_prompt(self): return f"Answer the following question about the screen: {self.question}"
```

### 5. `set_of_mark_task.py`

```python
from gui_tasks.base_task import BaseGUITask

class SetOfMarkTask(BaseGUITask):
    def task_name(self): return "set_of_mark"
    def system_prompt(self): return "You are a GUI element identifier."
    def user_prompt(self): return "List all visually marked elements in this screenshot."
```

### 6. `state_transition_captioning_task.py`

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
