```
# Final version of main.py with:
# - element_description as JSON array
# - QA changed to screen-level question
# - "set_of_mark_prompting" renamed to "set_of_mark"
# - parse_element_description removed

import os
import json
import base64
import requests
import argparse
from datetime import datetime
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_ENDPOINT = os.getenv("API_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# Encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Send request to model
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

# Abstract task base
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

    def run(self):
        image_data = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.image_data}"}}]
        return request_model(self.system_prompt(), self.user_prompt(), image_data)

# Tasks
class ElementDescriptionTask(BaseGUITask):
    def task_name(self): return "element_description"
    def system_prompt(self):
        return "You are a UI screen analyzer that returns all visible UI elements in JSON format."
    def user_prompt(self):
        return (
            "Return a complete JSON array of all visible UI elements in the screenshot.\n"
            "Each object must include:\n"
            "- 'type': e.g., button, checkbox, label\n"
            "- 'text': visible label or value\n"
            "- 'function': what it does\n"
            "- 'visual': color/style/shape\n"
            "- 'position': top/bottom/left/right\n"
            "- 'box': bounding box [x1, y1, x2, y2]\n\n"
            "Example format:\n"
            "[\n"
            "  {\"type\": \"button\", \"text\": \"Submit\", \"function\": \"submits form\", \"visual\": \"blue rectangle\", \"position\": \"top right\", \"box\": [100, 200, 250, 240]},\n"
            "  {\"type\": \"checkbox\", \"text\": \"I agree\", \"function\": \"accept terms\", \"visual\": \"gray square\", \"position\": \"bottom left\", \"box\": [80, 300, 180, 340]}\n"
            "]"
        )

class DenseCaptioningTask(BaseGUITask):
    def task_name(self): return "dense_captioning"
    def system_prompt(self):
        return "You are a GUI captioning assistant."
    def user_prompt(self):
        return "Describe all visible elements and layout in this UI screenshot in one paragraph."

class QATask(BaseGUITask):
    def __init__(self, image_path, question):
        super().__init__(image_path)
        self.question = question
    def task_name(self): return "qa"
    def system_prompt(self):
        return "You are a screen-level question answering assistant."
    def user_prompt(self):
        return f"Answer the following question about the overall screen: {self.question}"

class SetOfMarkTask(BaseGUITask):
    def task_name(self): return "set_of_mark"
    def system_prompt(self):
        return "You are a GUI element identifier for marked elements."
    def user_prompt(self):
        return "List all visually marked elements in this screenshot with their type, function, and position."

class StateTransitionCaptioningTask:
    def __init__(self, before_image, after_image):
        self.before_image = encode_image(before_image)
        self.after_image = encode_image(after_image)
    def task_name(self): return "state_transition_captioning"
    def run(self):
        image_data = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.before_image}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.after_image}"}}
        ]
        return request_model(
            "You are a GUI transition analyzer.",
            "Describe what changes occurred between the two screenshots.",
            image_data
        )

# CLI argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run GUI perception tasks and output JSON.")
    parser.add_argument("--image", required=True, help="Main screenshot image path")
    parser.add_argument("--before", help="Before image path for state transition")
    parser.add_argument("--after", help="After image path for state transition")
    parser.add_argument("--question", default="What is the main purpose of this screen?", help="Screen-level question for QA")
    parser.add_argument("--output", default="gui_result.json", help="Output JSON filename")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", choices=["element", "caption", "qa", "mark", "state"], help="Run only a specific task")
    task_group.add_argument("--all", action="store_true", help="Run all tasks")
    return parser.parse_args()

# Task runner
def run_selected_tasks(args):
    output = {
        "screenshot": args.image,
        "metadata": {},
        "perception_tasks": {},
        "created_at": datetime.utcnow().isoformat()
    }

    if args.all or args.task == "element":
        print("[+] Running element_description...")
        element_raw = ElementDescriptionTask(args.image).run()
        try:
            output["perception_tasks"]["element_description"] = json.loads(element_raw)
        except json.JSONDecodeError:
            output["perception_tasks"]["element_description"] = {"error": "Invalid JSON returned"}

    if args.all or args.task == "caption":
        print("[+] Running dense_captioning...")
        caption = DenseCaptioningTask(args.image).run()
        output["perception_tasks"]["dense_captioning"] = caption

    if args.all or args.task == "qa":
        print("[+] Running QA...")
        qa = QATask(args.image, args.question).run()
        output["perception_tasks"]["qa"] = {
            "question": args.question,
            "answer": qa
        }

    if args.all or args.task == "mark":
        print("[+] Running set_of_mark...")
        marks = SetOfMarkTask(args.image).run()
        output["perception_tasks"]["set_of_mark"] = marks

    if (args.all or args.task == "state") and args.before and args.after:
        print("[+] Running state_transition_captioning...")
        state_caption = StateTransitionCaptioningTask(args.before, args.after).run()
        output["perception_tasks"]["state_transition_captioning"] = {
            "before_image": args.before,
            "after_image": args.after,
            "caption": state_caption
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[✓] 결과 저장 완료: {args.output}")

# Entry point
if __name__ == "__main__":
    args = parse_arguments()
    run_selected_tasks(args)

```
