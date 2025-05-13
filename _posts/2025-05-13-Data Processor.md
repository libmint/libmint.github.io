```
# Final version of main.py with enhanced element extraction and full integration

import os
import json
import base64
import requests
import argparse
import re
from datetime import datetime
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load .env variables
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

# Parse element description output
def parse_element_description(text):
    elements = []
    current = {}
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if re.match(r"^\d+[\).]", line):
            if current:
                elements.append(current)
                current = {}
        if "type" not in current:
            m = re.search(r"(button|checkbox|textfield|link|image|icon|label)", line, re.I)
            if m:
                current["type"] = m.group(1).lower()
        if "text" not in current:
            m = re.search(r"label(ed)? as ['\"]?([^'\"]+)['\"]?", line, re.I)
            if m:
                current["text"] = m.group(2)
        if "function" not in current:
            m = re.search(r"(submits|toggles|navigates|opens|closes|accepts|sends)[^\.\n]*", line, re.I)
            if m:
                current["function"] = m.group(0).strip().lower()
        if "position" not in current:
            m = re.search(r"(top|bottom|center|left|right)[-\s]?(left|right|center)?", line, re.I)
            if m:
                pos = " ".join(filter(None, m.groups())).lower()
                current["position"] = pos
        if "visual" not in current:
            m = re.search(r"(blue|gray|white|green|red|orange|yellow|black)[^\n\.]*", line, re.I)
            if m:
                current["visual"] = m.group(0).strip().lower()
        if "box" not in current:
            m = re.search(r"\[?\(?\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]?\)?", line)
            if m:
                current["box"] = list(map(int, m.groups()))
    if current:
        elements.append(current)
    return elements

# Base class for tasks
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

# Task implementations
class ElementDescriptionTask(BaseGUITask):
    def task_name(self): return "element_description"
    def system_prompt(self):
        return "You are an expert GUI element extraction agent for screen readers."
    def user_prompt(self):
        return (
            "Analyze the entire UI screenshot and extract a complete list of all visible UI elements.\n"
            "For each element, provide the following fields:\n"
            "- 'type': type of the UI element (e.g., button, checkbox, label, input field)\n"
            "- 'text': the visible label or content\n"
            "- 'function': what it does or triggers\n"
            "- 'visual': color, border, shape, icon, etc.\n"
            "- 'position': top/bottom/left/right/center etc.\n"
            "- 'box': the bounding box in [x1, y1, x2, y2] format\n\n"
            "Return your output as a bullet list, one line per element."
        )

class DenseCaptioningTask(BaseGUITask):
    def task_name(self): return "dense_captioning"
    def system_prompt(self):
        return "You are a captioning assistant for GUI layouts."
    def user_prompt(self):
        return "Generate a dense caption that summarizes the full layout and all visible elements in the UI screenshot."

class QATask(BaseGUITask):
    def __init__(self, image_path, question):
        super().__init__(image_path)
        self.question = question
    def task_name(self): return "qa"
    def system_prompt(self):
        return "You are a GUI question-answering assistant."
    def user_prompt(self):
        return f"Answer the following question about the UI: {self.question}"

class SetOfMarkTask(BaseGUITask):
    def task_name(self): return "set_of_mark_prompting"
    def system_prompt(self):
        return "You are a GUI element identifier using visual markers."
    def user_prompt(self):
        return "Identify and describe the elements visually marked in this screenshot. Include type, function, and position."

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
            "Compare the two screenshots and describe what changed between them.",
            image_data
        )

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run GUI perception tasks and output JSON.")
    parser.add_argument("--image", required=True, help="Main screenshot image path")
    parser.add_argument("--before", help="Before image path for state transition")
    parser.add_argument("--after", help="After image path for state transition")
    parser.add_argument("--question", default="What does the top button do?", help="Question for QA task")
    parser.add_argument("--output", default="gui_result.json", help="Output JSON filename")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", choices=["element", "caption", "qa", "mark", "state"], help="Run only a specific task")
    task_group.add_argument("--all", action="store_true", help="Run all tasks")
    return parser.parse_args()

# Task executor
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
        output["perception_tasks"]["element_description"] = parse_element_description(element_raw)

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
        print("[+] Running set_of_mark_prompting...")
        marks = SetOfMarkTask(args.image).run()
        output["perception_tasks"]["set_of_mark_prompting"] = marks

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

# Entrypoint
if __name__ == "__main__":
    args = parse_arguments()
    run_selected_tasks(args)

```
