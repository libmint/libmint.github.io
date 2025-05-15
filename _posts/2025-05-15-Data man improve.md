

---

## 📁 구조 개요

```
main.py
gui_tasks/
├── element_description_task.py
├── dense_captioning_task.py
├── qa_task.py
├── set_of_mark_task.py
├── task_utils.py
```

---

## ✅ 1. `main.py`

```python
import argparse
import os
import json
from datetime import datetime

from model_utils import request_model
from gui_tasks.element_description_task import ElementDescriptionTask
from gui_tasks.dense_captioning_task import DenseCaptioningTask
from gui_tasks.qa_task import QATask
from gui_tasks.set_of_mark_task import SetOfMarkTask
from gui_tasks.state_transition_captioning_task import StateTransitionCaptioningTask
from gui_tasks.task_utils import save_result_json


def run_selected_tasks(args):
    image_paths = []
    for root, _, files in os.walk(args.image_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, f))

    for img in image_paths:
        print(f"[+] Processing {img}")
        base_name = os.path.splitext(os.path.basename(img))[0]
        folder = os.path.dirname(img)
        localized_json_path = os.path.join(folder, f"{base_name}_localized.json")

        if not os.path.exists(localized_json_path):
            print(f"[!] Skipping {img}: No corresponding localized JSON found.")
            continue

        if args.task == "element":
            result = {
                "screenshot": img,
                "perception_tasks": {
                    "element_description": ElementDescriptionTask(img).run()
                }
            }
            save_result_json(img, result, "element")

        elif args.task == "caption":
            result = {
                "screenshot": img,
                "perception_tasks": {
                    "dense_captioning": DenseCaptioningTask(img).run(request_model)
                }
            }
            save_result_json(img, result, "caption")

        elif args.task == "qa":
            result = {
                "screenshot": img,
                "perception_tasks": {
                    "qa": {
                        "question": args.question,
                        "answer": QATask(img, args.question).run(request_model)
                    }
                }
            }
            save_result_json(img, result, "qa")

        elif args.task == "mark":
            result = {
                "screenshot": img,
                "perception_tasks": {
                    "set_of_mark": SetOfMarkTask(img).run(request_model)
                }
            }
            save_result_json(img, result, "mark")

        elif args.task == "state" and args.before and args.after:
            result = {
                "screenshot": img,
                "perception_tasks": {
                    "state_transition_captioning": {
                        "before_image": args.before,
                        "after_image": args.after,
                        "caption": StateTransitionCaptioningTask(args.before, args.after).run(request_model)
                    }
                }
            }
            save_result_json(img, result, "state")

        elif args.all:
            result = {
                "screenshot": img,
                "perception_tasks": {
                    "element_description": ElementDescriptionTask(img).run(),
                    "dense_captioning": DenseCaptioningTask(img).run(request_model),
                    "qa": {
                        "question": args.question,
                        "answer": QATask(img, args.question).run(request_model)
                    },
                    "set_of_mark": SetOfMarkTask(img).run(request_model)
                },
                "created_at": datetime.utcnow().isoformat()
            }
            save_result_json(img, result, "all")


def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--image_dir", required=True)
    run.add_argument("--before")
    run.add_argument("--after")
    run.add_argument("--question", default="이 버튼은 무슨 기능인가요?")
    run.add_argument("--output", required=True)
    group = run.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", choices=["element", "caption", "qa", "mark", "state"])
    group.add_argument("--all", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == "run":
        run_selected_tasks(args)
```

---

## ✅ 2. `gui_tasks/task_utils.py`

```python
import os
import json

def save_result_json(image_path, result: dict, task_name: str):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    folder = os.path.dirname(image_path)
    out_path = os.path.join(folder, f"{base_name}_{task_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[✓] Saved: {out_path}\\n")
```

---

## ✅ 3. `element_description_task.py`

```python
import os
import json

class ElementDescriptionTask:
    def __init__(self, image_path):
        self.image_path = image_path
        self.elements = self.load_elements()

    def load_elements(self):
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        folder = os.path.dirname(self.image_path)
        json_path = os.path.join(folder, f"{base_name}_localized.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Localized JSON not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f).get("elements", [])

    def run(self, request_model=None):
        return self.elements
```

---

## ✅ 4. `dense_captioning_task.py`

```python
import os
import json

class DenseCaptioningTask:
    def __init__(self, image_path):
        self.image_path = image_path
        self.elements = self.load_elements()

    def load_elements(self):
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        folder = os.path.dirname(self.image_path)
        json_path = os.path.join(folder, f"{base_name}_localized.json")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f).get("elements", [])

    def run(self, request_model):
        prompt = "Describe the UI shown in this image in detail."
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{request_model.encode_image(self.image_path)}"}}]}]
        return request_model(messages)
```

---

## ✅ 5. `qa_task.py`

```python
import os
import json

class QATask:
    def __init__(self, image_path, question):
        self.image_path = image_path
        self.question = question
        self.elements = self.load_elements()

    def load_elements(self):
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        folder = os.path.dirname(self.image_path)
        json_path = os.path.join(folder, f"{base_name}_localized.json")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f).get("elements", [])

    def run(self, request_model):
        prompt = f"Question: {self.question}"
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{request_model.encode_image(self.image_path)}"}}]}]
        return request_model(messages)
```

---

## ✅ 6. `set_of_mark_task.py`

```python
import os
import json

class SetOfMarkTask:
    def __init__(self, image_path):
        self.image_path = image_path
        self.elements = self.load_elements()

    def load_elements(self):
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        folder = os.path.dirname(self.image_path)
        json_path = os.path.join(folder, f"{base_name}_localized.json")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f).get("elements", [])

    def run(self, request_model):
        prompt = "Identify the visually emphasized or selected UI elements in this image."
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{request_model.encode_image(self.image_path)}"}}]}]
        return request_model(messages)
```

---
