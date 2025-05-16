

---

### ✅ 개선 사항 반영 목록

1. `dest_root`로 변수명 변경
2. `source_dir`도 사용자 입력으로 받음
3. 파일만 처리 (`os.path.isfile`)
4. 실패한 복사 목록을 따로 출력
5. 확장자 포함한 파일명도 그대로 처리

---

### 🧠 최종 코드

```python
import os
import json
import shutil

# 사용자 입력
dest_root = input("복사할 목적지 폴더 경로(dest_root)를 입력하세요: ").strip()
json_path = input("JSON 파일 경로를 입력하세요: ").strip()
source_dir = input("원본 파일들이 위치한 폴더 경로(source_dir)를 입력하세요: ").strip()

# 확장자 목록 (우선순위대로 시도)
extensions = ['.png', '.jpg', '.jpeg']

# JSON 읽기
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 파일 목록 추출 (중복 제거)
all_files = set()
for file_list in data.values():
    all_files.update(file_list)

# 실패한 파일 저장 리스트
failed_files = []

# 파일 복사 처리
for file_base in all_files:
    found = False

    for ext in extensions:
        filename = file_base + ext
        src_file_path = os.path.join(source_dir, filename)
        if os.path.isfile(src_file_path):
            dest_folder = os.path.join(dest_root, file_base)
            os.makedirs(dest_folder, exist_ok=True)

            dest_file_path = os.path.join(dest_folder, filename)
            shutil.copy(src_file_path, dest_file_path)

            print(f"✅ 복사 완료: {src_file_path} -> {dest_file_path}")
            found = True
            break

    if not found:
        print(f"⚠️  복사 실패 (파일 없음): {file_base} + (.png/.jpg/.jpeg)")
        failed_files.append(file_base)

# 실패 파일 목록 출력
if failed_files:
    print("\n다음 항목은 확장자 포함된 파일을 찾지 못했습니다:")
    for f in failed_files:
        print(f" - {f}")
else:
    print("\n모든 파일 복사 성공 ✅")

```

---

### 💡 사용 예시

```plaintext
복사할 목적지 폴더 경로(dest_root)를 입력하세요: ./output
JSON 파일 경로를 입력하세요: ./data/actions.json
원본 파일들이 위치한 폴더 경로(source_dir)를 입력하세요: ./source_files
```

---


