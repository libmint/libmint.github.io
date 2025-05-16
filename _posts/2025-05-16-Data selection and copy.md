

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

# JSON 읽기
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 파일 prefix 목록 추출 (중복 제거)
all_prefixes = set()
for file_list in data.values():
    all_prefixes.update(file_list)

# 실패한 항목 기록용
not_found_prefixes = []

# 복사 처리
for prefix in all_prefixes:
    matched_file = None

    # 파일 목록 중 prefix로 시작하는 첫 번째 파일 찾기
    for filename in os.listdir(source_dir):
        if filename.startswith(prefix) and os.path.isfile(os.path.join(source_dir, filename)):
            matched_file = filename
            break

    if matched_file:
        dest_folder = os.path.join(dest_root, prefix)
        os.makedirs(dest_folder, exist_ok=True)

        src_path = os.path.join(source_dir, matched_file)
        dest_path = os.path.join(dest_folder, matched_file)
        shutil.copy(src_path, dest_path)

        print(f"✅ 복사 완료: {src_path} -> {dest_path}")
    else:
        print(f"⚠️  복사 실패: '{prefix}'로 시작하는 파일을 찾을 수 없음")
        not_found_prefixes.append(prefix)

# 요약 출력
if not_found_prefixes:
    print("\n다음 prefix는 해당하는 파일이 source_dir에 존재하지 않습니다:")
    for p in not_found_prefixes:
        print(f" - {p}")
else:
    print("\n모든 prefix에 해당하는 파일이 성공적으로 복사되었습니다 ✅")


```

---

### 💡 사용 예시

```plaintext
복사할 목적지 폴더 경로(dest_root)를 입력하세요: ./output
JSON 파일 경로를 입력하세요: ./data/actions.json
원본 파일들이 위치한 폴더 경로(source_dir)를 입력하세요: ./source_files
```

---


