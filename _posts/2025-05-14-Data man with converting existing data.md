```
# Unified main.py for full pipeline: run, convert, fill
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

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def request_model(system_prompt, user_prompt, image_b64):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]}
    ]
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2
    }
    response = requests.post(API_ENDPOINT, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

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
        return request_model(self.system_prompt(), self.user_prompt(), self.image_data)

class ElementDescriptionTask(BaseGUITask):
    def task_name(self): return "element_description"
    def system_prompt(self): return "You are a UI screen analyzer that returns all visible UI elements in JSON format."
    def user_prompt(self):
        return (
            "Return a complete JSON array of all visible UI elements in the screenshot.\n"
            "Each object must include:\n"
            "- 'type': e.g., button, checkbox, label\n"
            "- 'text': visible label or value\n"
            "- 'function': what it does\n"
            "- 'visual': color/style/shape\n"
            "- 'position': top/bottom/left/right\n"
            "- 'box': bounding box [x1, y1, x2, y2]\n"
        )

class DenseCaptioningTask(BaseGUITask):
    def task_name(self): return "dense_captioning"
    def system_prompt(self): return "You are a GUI captioning assistant."
    def user_prompt(self): return "Describe all visible elements and layout in this UI screenshot in one paragraph."

class QATask(BaseGUITask):
    def __init__(self, image_path, question):
        super().__init__(image_path)
        self.question = question
    def task_name(self): return "qa"
    def system_prompt(self): return "You are a screen-level question answering assistant."
    def user_prompt(self): return f"Answer the following question about the overall screen: {self.question}"

class SetOfMarkTask(BaseGUITask):
    def task_name(self): return "set_of_mark"
    def system_prompt(self): return "You are a GUI element identifier for marked elements."
    def user_prompt(self): return "List all visually marked elements in this screenshot with their type, function, and position."

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
    print(f"[✓] Results saved: {args.output}")

def convert_existing_data_with_prompts(input_path, output_path, screen_width=1080, screen_height=1920,
                                       default_qa_question="What is the main purpose of this screen?",
                                       default_caption_prompt="Describe all visible elements and layout.",
                                       default_mark_prompt="Identify marked elements in this screenshot."):
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

        perception_tasks = {
            "element_description": elements,
            "dense_captioning": {
                "prompt": default_caption_prompt,
                "answer": "unknown"
            },
            "qa": {
                "question": default_qa_question,
                "answer": "unknown"
            },
            "set_of_mark": {
                "prompt": default_mark_prompt,
                "answer": "unknown"
            },
            "state_transition_captioning": None
        }

        converted = {
            "screenshot": entry["img_filename"],
            "metadata": {},
            "perception_tasks": perception_tasks,
            "created_at": datetime.utcnow().isoformat()
        }
        new_data.append(converted)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"[✓] Converted and saved: {output_path}")
    return output_path

def fill_unknown_answers(data_path, image_dir, output_path):
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for entry in dataset:
        img_path = os.path.join(image_dir, entry["screenshot"])
        if not os.path.exists(img_path):
            print(f"[!] Skipping missing image: {img_path}")
            continue
        b64img = encode_image(img_path)
        tasks = entry["perception_tasks"]

        if isinstance(tasks.get("dense_captioning"), dict) and tasks["dense_captioning"]["answer"] == "unknown":
            print(f"[+] Filling dense_captioning for {entry['screenshot']}")
            result = request_model("You are a GUI captioning assistant.",
                                   tasks["dense_captioning"]["prompt"], b64img)
            tasks["dense_captioning"]["answer"] = result

        if isinstance(tasks.get("qa"), dict) and tasks["qa"]["answer"] == "unknown":
            print(f"[+] Filling QA for {entry['screenshot']}")
            result = request_model("You are a screen-level question answering assistant.",
                                   tasks["qa"]["question"], b64img)
            tasks["qa"]["answer"] = result

        if isinstance(tasks.get("set_of_mark"), dict) and tasks["set_of_mark"]["answer"] == "unknown":
            print(f"[+] Filling set_of_mark for {entry['screenshot']}")
            result = request_model("You are a GUI element identifier for marked elements.",
                                   tasks["set_of_mark"]["prompt"], b64img)
            tasks["set_of_mark"]["answer"] = result

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"[✓] Answers filled and saved: {output_path}")
    return output_path

def parse_arguments():
    parser = argparse.ArgumentParser(description="GUI Task Processor & Transformer")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    run_parser = subparsers.add_parser("run", help="Run GPT-based GUI perception tasks")
    run_parser.add_argument("--image", required=True, help="Main screenshot image path")
    run_parser.add_argument("--before", help="Before image path for state transition")
    run_parser.add_argument("--after", help="After image path for state transition")
    run_parser.add_argument("--question", default="What is the main purpose of this screen?")
    run_parser.add_argument("--output", default="gui_result.json")
    task_group = run_parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", choices=["element", "caption", "qa", "mark", "state"])
    task_group.add_argument("--all", action="store_true")

    convert_parser = subparsers.add_parser("convert", help="Convert old-format GUI dataset")
    convert_parser.add_argument("--input", required=True)
    convert_parser.add_argument("--output", required=True)
    convert_parser.add_argument("--screen_width", type=int, default=1080)
    convert_parser.add_argument("--screen_height", type=int, default=1920)

    fill_parser = subparsers.add_parser("fill", help="Fill unknown answers using GPT")
    fill_parser.add_argument("--input", required=True)
    fill_parser.add_argument("--image_dir", required=True)
    fill_parser.add_argument("--output", required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.mode == "convert":
        convert_existing_data_with_prompts(
            input_path=args.input,
            output_path=args.output,
            screen_width=args.screen_width,
            screen_height=args.screen_height
        )
    elif args.mode == "fill":
        fill_unknown_answers(
            data_path=args.input,
            image_dir=args.image_dir,
            output_path=args.output
        )
    elif args.mode == "run":
        run_selected_tasks(args)

```
