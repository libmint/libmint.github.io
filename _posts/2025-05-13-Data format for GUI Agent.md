UI-TARS에서 사용하는 데이터 포맷은 **스크린샷 기반의 시각-언어 데이터**와 **정형화된 GUI 요소 메타데이터**가 결합된 구조입니다. 최신 논문과 공식 문서에 따르면, UI-TARS의 데이터 포맷은 다음과 같은 주요 특징을 가집니다:

---

## UI-TARS 데이터 포맷의 구조

### **1. 입력 데이터(Perception & Grounding)**
- **스크린샷**: 전체 GUI 화면의 이미지(픽셀 데이터, 보통 base64 인코딩 또는 파일 경로)
- **GUI 요소 정보**: 각 요소별로 아래와 같은 메타데이터가 포함됩니다.
  - **bounding_box**: [x1, y1, x2, y2] 형태의 픽셀 좌표
  - **element_type**: 버튼, 입력창, 체크박스 등 요소의 타입
  - **depth**: GUI 계층 구조상 깊이 정보
  - **text_content**: 해당 요소에 표시된 텍스트
- **메타데이터**: 앱/웹/OS 등 환경 정보, 화면 해상도 등

### **2. 액션 및 상호작용 데이터**
- **task_instruction**: 자연어로 주어진 명령 또는 태스크 설명
- **action_trace**: 에이전트가 수행한 액션 시퀀스(예: 클릭, 입력, 스크롤 등)
  - 각 액션별로 `action_type`, `target_bbox`, `input_text`(필요시) 등 세부 정보 포함
- **thoughts**: 각 액션 전후의 심층적 reasoning(반성적 사고) 로그

---

## **예시 데이터 포맷(JSON)**

```json
{
  "screenshot": "base64_encoded_image",
  "task_instruction": "설정에서 Wi-Fi를 켜세요.",
  "elements": [
    {
      "bounding_box": [120, 300, 400, 360],
      "element_type": "toggle",
      "depth": 3,
      "text_content": "Wi-Fi"
    },
    {
      "bounding_box": [420, 300, 700, 360],
      "element_type": "button",
      "depth": 3,
      "text_content": "켜기"
    }
    // ... 추가 요소
  ],
  "action_trace": [
    {
      "step": 1,
      "action_type": "click",
      "target_bbox": [420, 300, 700, 360],
      "thought": "Wi-Fi를 켜기 위해 '켜기' 버튼을 클릭해야 한다."
    }
    // ... 추가 액션
  ],
  "metadata": {
    "platform": "Android",
    "screen_resolution": "1080x2400"
  }
}
```

---

## **설계 및 활용 포인트**

- **스크린샷과 GUI 요소 메타데이터의 결합**: UI-TARS는 단순히 이미지만 입력받는 것이 아니라, 각 GUI 요소의 위치, 타입, 텍스트 등 정형화된 정보를 함께 사용합니다. 이는 grounding(정확한 요소 식별) 성능을 크게 높입니다[2][5].
- **액션 시퀀스와 reasoning 로그**: 각 액션에 대해 시스템적 reasoning("thoughts")을 기록하여, 에이전트의 의사결정 과정을 명확히 추적할 수 있습니다. 이는 ReAct 프레임워크에서 영감을 받은 구조입니다[2].
- **플랫폼/환경 정보**: 다양한 OS, 앱, 웹 환경을 포괄할 수 있도록 메타데이터를 포함합니다.

---

이 포맷은 UI-TARS가 데스크톱, 모바일, 웹 등 다양한 환경에서 perception, grounding, action, reasoning까지 통합적으로 학습하고 평가할 수 있도록 설계되었습니다. 실제 데이터셋 구축 시, 스크린샷과 함께 각 요소의 위치/속성/텍스트/계층 정보 및 액션 로그를 구조화하여 저장하는 것이 권장됩니다[2][5].

OS-Atlas의 모바일용 데이터 포맷은 **JSON 형식**으로, 각 샘플은 GUI 스크린샷과 해당 화면에서 특정 UI 요소를 참조하는 명령 및 그 요소의 위치(바운딩 박스)를 포함합니다. 데이터는 모바일, 데스크톱, 웹 세 가지 도메인으로 구분되며, 모바일 데이터는 `mobile_domain` 디렉터리 하위에 여러 서브셋(AMEX, UIBert, Widget Captioning, RICOSCA, Android_world_data 등)으로 제공됩니다[1].

### 주요 필드 설명

- **img_filename**: 스크린샷 이미지 파일명
- **instruction**: 사람이 작성한 명령 또는 참조 표현(ally tree, HTML 등에서 추출)
- **bbox**: 명령이 참조하는 타겟 요소의 바운딩 박스  
  - `[left, top, right, bottom]` 순서  
  - 각 값은 0~1 사이의 실수로, 이미지의 너비/높이에 대한 상대적 비율
- **data_type** (선택): 요소의 타입(예: button, text 등), 추출 가능한 경우에만 포함

### 예시

```json
{
  "img_filename": "amex_000123.png",
  "instruction": "Wi-Fi를 켜는 토글을 찾아 클릭하세요.",
  "bbox": [0.12, 0.32, 0.25, 0.38],
  "data_type": "toggle"
}
```

- **img_filename**은 실제 이미지 파일(압축파일 내 포함)과 연결됩니다.
- **bbox**의 각 값은 이미지 크기에 곱해서 실제 픽셀 좌표로 변환할 수 있습니다.
- **data_type**은 있을 때만 포함됩니다.

### 대화형(Conversations) 포맷 예시

OS-Atlas는 대화형 어노테이션도 지원합니다. 예시:
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "\n아래 UI 요소를 찾아주세요.\nWi-Fi\nBluetooth"
    },
    {
      "from": "gpt",
      "value": "Wi-Fi[[120,320,250,380]]\nBluetooth[[120,400,250,460]]"
    }
  ]
}
```
- 이 경우 바운딩 박스는 0~1 정규화 대신 0~1000 스케일로 변환해서 쓰기도 합니다.

### 요약

- **저장 형식**: JSON  
- **필수 필드**: `img_filename`, `instruction`, `bbox`  
- **선택 필드**: `data_type`  
- **좌표**: `[left, top, right, bottom]`, 0~1 사이 실수  
- **이미지**: 별도 압축파일(zip)로 제공, 파일명으로 연결  
- **대화형 포맷**: `conversations` 필드 사용 가능

이 포맷은 OS-Atlas Hugging Face 데이터셋의 [mobile_domain] 디렉터리 내 여러 JSON 파일(예: `amex_raw.json`, `uibert_raw.json` 등)에서 실제로 확인할 수 있습니다[1].


데이터 포맷(스크린샷, 자연어 명령, 바운딩 박스, 요소 속성 등)은 실제로 최신 GUI grounding 및 에이전트 연구 데이터셋에서 사용되는 방식이며, 아래와 같은 공개 자료에서 확인할 수 있습니다.

---

## 관련 데이터셋 및 실제 포맷 위치

### 1. **UGround (Web-Hybrid/Web-Direct)**
- **위치:** Hugging Face Datasets ([osunlp/UGround-V1-Data][1])
- **포맷 예시:**  
  - `screenshot` (이미지)
  - `instruction` (자연어 명령)
  - `referring_expression` (참조 표현)
  - `bbox` (타겟 요소 바운딩 박스)
  - `element_type`, `element_text` 등 부가 정보

### 2. **ScreenSpot**
- **위치:** Hugging Face Datasets ([rootsautomation/ScreenSpot][3])
- **포맷 예시:**  
  - `image`: 스크린샷 이미지
  - `instruction`: 명령/질문
  - `bbox`: 타겟 요소 바운딩 박스 (좌상단 x, y, 우하단 x, y)
  - `data_type`: 요소 타입(텍스트/아이콘)
  - `data_source`: 플랫폼 정보

### 3. **UI-TARS**
- **위치:** 논문 및 공식 GitHub ([arXiv:2501.12326][4], github.com/bytedance/UI-TARS)
- **포맷 설명:**  
  - `screenshot`: 전체 화면 이미지
  - `elements`: 각 GUI 요소의 바운딩 박스, 타입, 텍스트 등 구조화된 메타데이터
  - `task_instruction`: 자연어 명령
  - `action_trace`: 액션 시퀀스(클릭, 입력 등)
  - `metadata`: 플랫폼, 해상도 등

---

## 참고 자료 내 실제 포맷 예시

- **ScreenSpot** 예시:
  ```json
  {
    "image": "screenshot.png",
    "instruction": "Click the 'Submit' button.",
    "bbox": [120, 300, 400, 360],
    "data_type": "text",
    "data_source": "Web"
  }
  ```


- **UI-TARS** 논문 설명:
  > "We curate a large-scale dataset by collecting screenshots using specialized parsing tools to extract metadata such as element types, bounding boxes, and text content... The grounding ability ... is improved by curating a vast dataset that pairs element descriptions with their spatial coordinates."  
  [4]

---

## 요약

- 위에 안내드린 예시 포맷은 실제로 UGround, ScreenSpot, UI-TARS 등 최신 GUI grounding 연구 데이터셋에서 사용되는 구조이며, Hugging Face Datasets 등에서 실제 JSON/CSV 형태로 확인할 수 있습니다.
- 각 데이터셋의 구조와 예시 샘플은 해당 링크에서 직접 확인하거나 다운로드해볼 수 있습니다.
