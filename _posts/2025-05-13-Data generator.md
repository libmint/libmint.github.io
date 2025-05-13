알겠습니다! 아래는 GPT-4o 기반 GUI 스크린샷 분석기 프로젝트의 최종 완성된 전체 코드입니다. 이 코드는 학습용 JSON 포맷에 맞춰 결과와 메타데이터를 저장합니다.


---

✅ 최종 통합 코드: main.py
```
import os
import json
import base64
from datetime import datetime
from abc import ABC, abstractmethod
import openai

# OpenAI API 키 설정
openai.api_key = "your-api-key-here"  # 여기에 실제 API 키 입력

# 이미지 파일을 base64로 인코딩
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 공통 베이스 클래스
class BaseGUITask(ABC):
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_data = encode_image(image_path)

    @abstractmethod
    def task_prompt(self):
        pass

    def run(self):
        message = [
            {"role": "system", "content": self.task_prompt()["system"]},
            {"role": "user", "content": [
                {"type": "text", "text": self.task_prompt()["user"]},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.image_data}"}}
            ]}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=message,
            temperature=0.3
        )
        return response.choices[0].message["content"]

# 1. 요소 설명
class ElementDescriptionTask(BaseGUITask):
    def task_prompt(self):
        return {
            "system": "You are an assistant that extracts and describes GUI elements from screenshots.",
            "user": "Identify all visible UI elements. For each, provide:\n1. Type\n2. Visual appearance\n3. Function\n4. Position"
        }

# 2. 전체 밀집 캡션
class DenseCaptioningTask(BaseGUITask):
    def task_prompt(self):
        return {
            "system": "You are a captioning assistant for GUI layouts.",
            "user": "Generate a detailed description summarizing all components and layout structure of the UI."
        }

# 3. 상태 전이 캡션
class StateTransitionCaptioningTask:
    def __init__(self, before_image, after_image):
        self.before_image = encode_image(before_image)
        self.after_image = encode_image(after_image)

    def run(self):
        message = [
            {"role": "system", "content": "You are a GUI transition analyzer."},
            {"role": "user", "content": [
                {"type": "text", "text": "Compare these two GUI screenshots and describe what changed (e.g., button pressed, content loaded)."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.before_image}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.after_image}"}}
            ]}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=message,
            temperature=0.3
        )
        return response.choices[0].message["content"]

# 4. 질의응답
class QATask(BaseGUITask):
    def __init__(self, image_path, question):
        super().__init__(image_path)
        self.question = question

    def task_prompt(self):
        return {
            "system": "You are a GUI question-answering assistant.",
            "user": f"Answer this question about the GUI: {self.question}"
        }

# 5. 마커 기반 인식
class SetOfMarkTask(BaseGUITask):
    def task_prompt(self):
        return {
            "system": "You are a GUI element identifier using visual markers.",
            "user": "Describe all elements marked with colored shapes or highlights in this screenshot. Indicate their function and location."
        }

# 모든 태스크 실행 및 JSON 저장
def run_all_tasks(
    image_path,
    before_after_images=None,
    qa_question="What is the purpose of this screen?",
    output_path="gui_analysis.json"
):
    results = {}

    print("Running Element Description...")
    task1 = ElementDescriptionTask(image_path)
    results["element_description"] = task1.run()

    print("Running Dense Captioning...")
    task2 = DenseCaptioningTask(image_path)
    results["dense_captioning"] = task2.run()

    print("Running State Transition Captioning...")
    if before_after_images and len(before_after_images) == 2:
        task3 = StateTransitionCaptioningTask(*before_after_images)
        results["state_transition_captioning"] = task3.run()
        before_path, after_path = before_after_images
    else:
        results["state_transition_captioning"] = "Not provided"
        before_path, after_path = None, None

    print("Running QA Task...")
    task4 = QATask(image_path, qa_question)
    results["qa"] = task4.run()

    print("Running Set of Mark Task...")
    task5 = SetOfMarkTask(image_path)
    results["set_of_mark"] = task5.run()

    # 메타데이터 작성
    metadata = {
        "screenshot_path": image_path,
        "before_image": before_path,
        "after_image": after_path,
        "question": qa_question,
        "created_at": datetime.utcnow().isoformat()
    }

    # 학습용 JSON 구성
    final_output = {
        "metadata": metadata,
        "results": results
    }

    # JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n[완료] 학습용 JSON이 '{output_path}'에 저장되었습니다.")
    return os.path.abspath(output_path)

# 실행 예시
if __name__ == "__main__":
    image_path = "gui_images/screenshot.png"
    before_after = ("gui_images/before.png", "gui_images/after.png")  # 생략 가능
    qa_text = "What does the top-right button do?"
    output_file = "gui_analysis.json"

    run_all_tasks(image_path, before_after_images=before_after, qa_question=qa_text, output_path=output_file)
```

---

✅ 실행 전 체크리스트

1. 라이브러리 설치:



pip install openai

2. 디렉토리 구성 예시:



gui_project/
├── main.py
├── gui_images/
│   ├── screenshot.png
│   ├── before.png
│   └── after.png
└── gui_analysis.json

3. OpenAI API 키 등록:



openai.api_key = "your-api-key-here"


---

✅ 출력 예시 (gui_analysis.json)

{
  "metadata": {
    "screenshot_path": "gui_images/screenshot.png",
    "before_image": "gui_images/before.png",
    "after_image": "gui_images/after.png",
    "question": "What does the top-right button do?",
    "created_at": "2025-05-13T14:00:00"
  },
  "results": {
    "element_description": "...",
    "dense_captioning": "...",
    "state_transition_captioning": "...",
    "qa": "...",
    "set_of_mark": "..."
  }
}


---

이제 이 JSON 파일을 기반으로 학습 데이터셋으로 확장하거나 JSONL로 변환해 fine-tuning에 활용할 수 있습니다.

Improve
```
import os
import json
import base64
from datetime import datetime
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import openai

# .env에서 API 키와 주소 불러오기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")  # 예: http://10.10.10.90:9000

# 이미지 base64 인코딩
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 공통 베이스 클래스
class BaseGUITask(ABC):
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_data = encode_image(image_path)

    @abstractmethod
    def task_prompt(self):
        pass

    def run(self):
        message = [
            {"role": "system", "content": self.task_prompt()["system"]},
            {"role": "user", "content": [
                {"type": "text", "text": self.task_prompt()["user"]},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.image_data}"}}
            ]}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=message,
            temperature=0.3
        )
        return response.choices[0].message["content"]

# 태스크 클래스 정의
class ElementDescriptionTask(BaseGUITask):
    def task_prompt(self):
        return {
            "system": "You are an assistant that extracts and describes GUI elements from screenshots.",
            "user": "Identify all visible UI elements. For each, provide:\n1. Type\n2. Visual appearance\n3. Function\n4. Position"
        }

class DenseCaptioningTask(BaseGUITask):
    def task_prompt(self):
        return {
            "system": "You are a captioning assistant for GUI layouts.",
            "user": "Generate a detailed description summarizing all components and layout structure of the UI."
        }

class StateTransitionCaptioningTask:
    def __init__(self, before_image, after_image):
        self.before_image = encode_image(before_image)
        self.after_image = encode_image(after_image)

    def run(self):
        message = [
            {"role": "system", "content": "You are a GUI transition analyzer."},
            {"role": "user", "content": [
                {"type": "text", "text": "Compare these two GUI screenshots and describe what changed (e.g., button pressed, content loaded)."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.before_image}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.after_image}"}}
            ]}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=message,
            temperature=0.3
        )
        return response.choices[0].message["content"]

class QATask(BaseGUITask):
    def __init__(self, image_path, question):
        super().__init__(image_path)
        self.question = question

    def task_prompt(self):
        return {
            "system": "You are a GUI question-answering assistant.",
            "user": f"Answer this question about the GUI: {self.question}"
        }

class SetOfMarkTask(BaseGUITask):
    def task_prompt(self):
        return {
            "system": "You are a GUI element identifier using visual markers.",
            "user": "Describe all elements marked with colored shapes or highlights in this screenshot. Indicate their function and location."
        }

# 모든 태스크 실행하고 JSON 저장
def run_all_tasks(
    image_path,
    before_after_images=None,
    qa_question="What is the purpose of this screen?",
    output_path="gui_analysis.json"
):
    results = {}

    print("Running Element Description...")
    task1 = ElementDescriptionTask(image_path)
    results["element_description"] = task1.run()

    print("Running Dense Captioning...")
    task2 = DenseCaptioningTask(image_path)
    results["dense_captioning"] = task2.run()

    print("Running State Transition Captioning...")
    if before_after_images and len(before_after_images) == 2:
        task3 = StateTransitionCaptioningTask(*before_after_images)
        results["state_transition_captioning"] = task3.run()
        before_path, after_path = before_after_images
    else:
        results["state_transition_captioning"] = "Not provided"
        before_path, after_path = None, None

    print("Running QA Task...")
    task4 = QATask(image_path, qa_question)
    results["qa"] = task4.run()

    print("Running Set of Mark Task...")
    task5 = SetOfMarkTask(image_path)
    results["set_of_mark"] = task5.run()

    # 메타데이터 저장
    metadata = {
        "screenshot_path": image_path,
        "before_image": before_path,
        "after_image": after_path,
        "question": qa_question,
        "created_at": datetime.utcnow().isoformat()
    }

    final_output = {
        "metadata": metadata,
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n[완료] 학습용 JSON이 '{output_path}'에 저장되었습니다.")
    return os.path.abspath(output_path)

# 실행 예시
if __name__ == "__main__":
    image_path = "gui_images/screenshot.png"
    before_after = ("gui_images/before.png", "gui_images/after.png")  # 선택적
    qa_text = "What does the top-right button do?"
    output_file = "gui_analysis.json"

    run_all_tasks(image_path, before_after_images=before_after, qa_question=qa_text, output_path=output_file)
```
