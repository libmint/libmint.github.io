

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
👉 [이미 이 화면에서 전체 코드 제공됨](https://chat.openai.com/share)

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


