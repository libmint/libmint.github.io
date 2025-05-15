
---

## 📄 `README.md`

````md
# UI-TARS Task Pipeline

GPT-4V 기반 GUI 이해 및 데이터 생성 파이프라인입니다.  
스크린샷에서 UI 요소를 추출하고 다양한 태스크(설명, 캡션, QA, 변화 감지 등)를 수행할 수 있습니다.

---

## 📦 기능 요약

- 📸 스크린샷 기반 UI 이해 자동화
- 🔍 PaddleOCR로 요소 인식 및 마킹
- 🧠 GPT-4o 기반 GUI 이해 태스크
  - `element_description`: UI 요소 정보
  - `dense_captioning`: 화면 설명 생성
  - `qa`: 사용자 질문 응답
  - `set_of_mark`: 강조된 UI 영역 추론
  - `state_transition_captioning`: 전후 이미지의 변화 설명
- 🧾 결과는 `{image_name}_{task}.json` 형식으로 저장

---

## 🛠️ 설치

```bash
pip install -r requirements.txt
# 또는 직접 설치
pip install paddleocr paddlepaddle requests python-dotenv
````

---

## ⚙️ 환경 설정 (`.env`)

```env
API_KEY=your-key-if-needed
API_BASE=http://10.10.10.90:9000/chat/completions
MODEL_NAME=gpt-4o
```

---

## 🚀 사용법

### 1. UI 요소 Localize (OCR 기반)

```bash
python main.py localize \
  --image_dir ./screenshots \
  --output_dir ./som_output
```

* 각 이미지에 대해:

  * `image_localized.json` 생성
  * `som_output/` 폴더에 마크된 이미지 저장

---

### 2. 태스크 실행

#### 예: QA 태스크

```bash
python main.py run \
  --image_dir ./screenshots \
  --task qa \
  --question "이 버튼은 무슨 기능인가요?" \
  --output dummy/
```

#### 예: 모든 태스크 실행

```bash
python main.py run \
  --image_dir ./screenshots \
  --all \
  --question "이 화면은 어떤 목적인가요?" \
  --output dummy/
```

---

## 📂 디렉토리 구조 예시

```
project/
├── main.py
├── model_utils.py
├── localizer.py
├── .env
├── gui_tasks/
│   ├── element_description_task.py
│   ├── dense_captioning_task.py
│   ├── qa_task.py
│   ├── set_of_mark_task.py
│   ├── state_transition_captioning_task.py
│   └── task_utils.py
└── screenshots/
    ├── home.png
    ├── home_localized.json
    ├── home_qa.json
    ├── home_caption.json
```

---

## 📤 출력 포맷 예시

```json
{
  "screenshot": "screenshots/home.png",
  "perception_tasks": {
    "qa": {
      "question": "이 버튼은 무슨 기능인가요?",
      "answer": "설정을 저장합니다."
    }
  }
}
```

---

## 🧩 개발자 참고

* 모든 task는 단일 `image_path`를 입력으로 받아 처리
* `localized.json`은 반드시 `image_name_localized.json` 형식
* 공통 유틸리티는 `model_utils.py`, `task_utils.py`에 위치

---

## 📬 문의

본 프로젝트에 대한 질문, 개선사항, 확장 제안은 언제든지 환영합니다!

```


