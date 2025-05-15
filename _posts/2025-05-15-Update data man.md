
---

## ✅ `model_utils.py`

```python
import os
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
API_BASE = os.getenv("API_BASE", "http://10.10.10.90:9000/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_localized_json_path(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(os.path.dirname(image_path), f"{base}_localized.json")

def load_elements(image_path):
    json_path = get_localized_json_path(image_path)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[!] Localized JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f).get("elements", [])

def request_model(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048
    }
    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    response = requests.post(API_BASE, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")
```

---

## ✅ `gui_tasks/task_utils.py`

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

## ✅ `element_description_task.py`

```python
import os
from model_utils import load_elements

class ElementDescriptionTask:
    def __init__(self, image_path):
        self.image_path = image_path
        self.elements = load_elements(image_path)

    def run(self, request_model=None):
        return self.elements
```

---

## ✅ `dense_captioning_task.py`

```python
import os
from model_utils import encode_image, load_elements

class DenseCaptioningTask:
    def __init__(self, image_path):
        self.image_path = image_path
        self.elements = load_elements(image_path)

    def run(self, request_model):
        prompt = "Describe the UI shown in this image in detail."
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encode_image(self.image_path)}"}}
            ]
        }]
        return request_model(messages)
```

---

## ✅ `qa_task.py`

```python
import os
from model_utils import encode_image, load_elements

class QATask:
    def __init__(self, image_path, question):
        self.image_path = image_path
        self.question = question
        self.elements = load_elements(image_path)

    def run(self, request_model):
        prompt = f"Question: {self.question}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encode_image(self.image_path)}"}}
            ]
        }]
        return request_model(messages)
```

---

## ✅ `set_of_mark_task.py`

```python
import os
from model_utils import encode_image, load_elements

class SetOfMarkTask:
    def __init__(self, image_path):
        self.image_path = image_path
        self.elements = load_elements(image_path)

    def run(self, request_model):
        prompt = "Identify the visually emphasized or selected UI elements in this image."
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encode_image(self.image_path)}"}}
            ]
        }]
        return request_model(messages)
```

---

## ✅ `state_transition_captioning_task.py`

```python
from model_utils import encode_image

class StateTransitionCaptioningTask:
    def __init__(self, before_image, after_image):
        self.before_image = before_image
        self.after_image = after_image

    def run(self, request_model):
        prompt = "Describe the UI state change between the before and after images."
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encode_image(self.before_image)}"}},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encode_image(self.after_image)}"}}
            ]
        }]
        return request_model(messages)
```

main
```
from localizer import process_directory_recursively

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

    localize = subparsers.add_parser("localize")
    localize.add_argument("--image_dir", required=True, help="Directory of screenshots (recursive)")
    localize.add_argument("--output_dir", default="som_output", help="Where to save marked images")

    return parser.parse_args()

...

if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == "run":
        run_selected_tasks(args)
    elif args.mode == "localize":
        process_directory_recursively(args.image_dir, args.output_dir)

```
---


