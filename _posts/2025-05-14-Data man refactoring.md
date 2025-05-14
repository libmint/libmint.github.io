
---

## ✅ 전체 프로젝트 구조

```
gui_project/
│
├── main.py
├── model_utils.py            ← GPT 호출 및 이미지 인코딩 공통 모듈
├── data_utils.py             ← 기존 데이터 변환 + unknown 채우기
├── gui_tasks/                ← 각 태스크 클래스 정의
│   ├── __init__.py
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
import argparse
from data_utils import convert_existing_data_with_prompts, fill_unknown_answers
from model_utils import request_model
from gui_tasks.element_description_task import ElementDescriptionTask
from gui_tasks.dense_captioning_task import DenseCaptioningTask
from gui_tasks.qa_task import QATask
from gui_tasks.set_of_mark_task import SetOfMarkTask
from gui_tasks.state_transition_captioning_task import StateTransitionCaptioningTask
from datetime import datetime
import os
import json

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
            result["perception_tasks"]["qa"] = {"question": args.question, "answer": qa}

        if args.all or args.task == "mark":
            marks = SetOfMarkTask(img).run(request_model)
            result["perception_tasks"]["set_of_mark"] = marks

        if (args.all or args.task == "state") and args.before and args.after:
            caption = StateTransitionCaptioningTask(args.before, args.after).run(request_model)
            result["perception_tasks"]["state_transition_captioning"] = {
                "before_image": args.before,
                "after_image": args.after,
                "caption": caption
            }

        results.append(result)

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

## ✅ `model_utils.py`

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

## ✅ `data_utils.py`

```python
import os
import json
from datetime import datetime
from model_utils import encode_image, request_model

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
                "dense_captioning": {
                    "prompt": "Describe all visible elements and layout.",
                    "answer": "unknown"
                },
                "qa": {
                    "question": "What is the main purpose of this screen?",
                    "answer": "unknown"
                },
                "set_of_mark": {
                    "prompt": "Identify marked elements.",
                    "answer": "unknown"
                },
                "state_transition_captioning": None
            },
            "created_at": datetime.utcnow().isoformat()
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
            print(f"[!] Image not found: {image_path}")
            continue

        b64img = encode_image(image_path)
        tasks = entry["perception_tasks"]

        def maybe(prompt, key):
            if tasks[key]["answer"] == "unknown":
                tasks[key]["answer"] = request_model(
                    f"You are a GUI {key.replace('_', ' ')} assistant.",
                    tasks[key]["prompt"] if "prompt" in tasks[key] else tasks[key]["question"],
                    [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64img}"}}]
                )

        maybe("captioning", "dense_captioning")
        maybe("question answering", "qa")
        maybe("mark identifier", "set_of_mark")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"[✓] Answers filled and saved: {output_path}")
```

---

## ✅ 예시 실행

```bash
# 기존 데이터 변환
python main.py convert --input old.json --output converted.json

# unknown 채우기
python main.py fill --input converted.json --image_dir ./screens --output filled.json

# GUI 태스크 수행
python main.py run --image_dir ./screens --all --output gui_result.json
```

---

