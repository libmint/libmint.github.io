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
