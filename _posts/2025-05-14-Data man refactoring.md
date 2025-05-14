```
from base_task import BaseGUITask

class ElementDescriptionTask(BaseGUITask):
    def task_name(self):
        return "element_description"

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
            "- 'box': bounding box [x1, y1, x2, y2]\n"
        )


from base_task import BaseGUITask

class DenseCaptioningTask(BaseGUITask):
    def task_name(self):
        return "dense_captioning"

    def system_prompt(self):
        return "You are a GUI captioning assistant."

    def user_prompt(self):
        return "Describe all visible elements and layout in this UI screenshot in one paragraph."


from base_task import BaseGUITask

class QATask(BaseGUITask):
    def __init__(self, image_path, question):
        super().__init__(image_path)
        self.question = question

    def task_name(self):
        return "qa"

    def system_prompt(self):
        return "You are a screen-level question answering assistant."

    def user_prompt(self):
        return f"Answer the following question about the overall screen: {self.question}"

from base_task import BaseGUITask

class SetOfMarkTask(BaseGUITask):
    def task_name(self):
        return "set_of_mark"

    def system_prompt(self):
        return "You are a GUI element identifier for marked elements."

    def user_prompt(self):
        return "List all visually marked elements in this screenshot with their type, function, and position."



import base64
import os
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
        {"role": "user", "content": image_data_list + [{"type": "text", "text": user_prompt}]}
    ]
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3
    }
    response = requests.post(API_ENDPOINT, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

class StateTransitionCaptioningTask:
    def __init__(self, before_image, after_image):
        self.before_image = encode_image(before_image)
        self.after_image = encode_image(after_image)

    def task_name(self):
        return "state_transition_captioning"

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


# main
from element_description_task import ElementDescriptionTask
from dense_captioning_task import DenseCaptioningTask
from qa_task import QATask
from set_of_mark_task import SetOfMarkTask
from state_transition_captioning_task import StateTransitionCaptioningTask


```
