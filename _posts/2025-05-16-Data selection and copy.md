

```python
import os
import json
import shutil

# 사용자 입력 받기
root_dir = input("덴트리 폴더 경로를 입력하세요: ").strip()
json_path = input("JSON 파일 경로를 입력하세요: ").strip()

# 복사할 원본 파일들이 있는 경로 (필요에 따라 수정)
source_dir = "./source_files"

# JSON 파일 읽기
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# JSON 데이터 순회
for key, files in data.items():
    unique_files = set(files)  # 중복 제거
    for file_name in unique_files:
        dest_folder = os.path.join(root_dir, file_name)
        os.makedirs(dest_folder, exist_ok=True)

        src_file_path = os.path.join(source_dir, file_name)
        dest_file_path = os.path.join(dest_folder, file_name)

        if os.path.exists(src_file_path):
            shutil.copy(src_file_path, dest_file_path)
            print(f"복사 완료: {src_file_path} -> {dest_file_path}")
        else:
            print(f"⚠️ 원본 파일 없음: {src_file_path}")
```

---

### 📌 사용 방법 요약

1. JSON 파일을 예를 들어 다음과 같이 작성하여 `actions.json`이라는 이름으로 저장하세요:

   ```json
   {
     "agoda": ["agoda_action_01", "agoda_action_01"],
     "youtube": ["youtube_action_01", "youtube_action_02"]
   }
   ```

2. 복사할 대상 파일들을 `./source_files/` 폴더에 준비하세요.

3. 프로그램 실행 후, 터미널에서 덴트리 경로와 JSON 경로를 입력하면 자동 복사됩니다.

