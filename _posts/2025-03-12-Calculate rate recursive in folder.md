다음은 하위 폴더에서 eval.json 파일을 찾아 success 항목이 1인 경우와 0인 경우의 개수와 해당 폴더 이름을 출력하는 파이썬 프로그램입니다:

```python
import os
import json
from collections import defaultdict

def find_eval_json_files():
    # 결과를 저장할 딕셔너리 초기화
    success_count = 0
    failure_count = 0
    success_folders = []
    failure_folders = []
    
    # 현재 디렉토리부터 시작
    root_dir = '.'
    
    # 모든 디렉토리와 하위 디렉토리 순회
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 현재 디렉토리에 eval.json 파일이 있는지 확인
        if 'eval.json' in filenames:
            file_path = os.path.join(dirpath, 'eval.json')
            
            try:
                # JSON 파일 읽기
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # success 항목이 있는지 확인
                if 'success' in data:
                    # success가 1인 경우
                    if data['success'] == 1:
                        success_count += 1
                        success_folders.append(dirpath)
                    # success가 0인 경우
                    elif data['success'] == 0:
                        failure_count += 1
                        failure_folders.append(dirpath)
            except json.JSONDecodeError:
                print(f"JSON 파싱 오류: {file_path}")
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {file_path}, 오류: {str(e)}")
    
    # 결과 출력
    print(f"Success 항목이 1인 경우: {success_count}개")
    for folder in success_folders:
        print(f"  - {folder}")
    
    print(f"\nSuccess 항목이 0인 경우: {failure_count}개")
    for folder in failure_folders:
        print(f"  - {folder}")

if __name__ == "__main__":
    find_eval_json_files()
```

```
import os
import json
from collections import defaultdict

def find_eval_json_files():
    # 결과를 저장할 변수 초기화
    success_count = 0
    failure_count = 0
    success_folders = []
    failure_folders = []
    
    # 현재 디렉토리부터 시작
    root_dir = '.'
    
    # 모든 디렉토리와 하위 디렉토리 순회
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 현재 디렉토리에 eval.json 파일이 있는지 확인
        if 'eval.json' in filenames:
            file_path = os.path.join(dirpath, 'eval.json')
            
            try:
                # JSON 파일 읽기
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # 데이터가 리스트 형태인지 확인
                if isinstance(data, list):
                    for item in data:
                        # 각 항목에 success 키가 있는지 확인
                        if 'succes' in item:  # 'succes'로 오타가 있는 경우 처리
                            # success가 1인 경우
                            if item['succes'] == 1:
                                success_count += 1
                                if dirpath not in success_folders:
                                    success_folders.append(dirpath)
                            # success가 0인 경우
                            elif item['succes'] == 0:
                                failure_count += 1
                                if dirpath not in failure_folders:
                                    failure_folders.append(dirpath)
                        # 'success'로 올바르게 쓰여진 경우도 확인
                        elif 'success' in item:
                            # success가 1인 경우
                            if item['success'] == 1:
                                success_count += 1
                                if dirpath not in success_folders:
                                    success_folders.append(dirpath)
                            # success가 0인 경우
                            elif item['success'] == 0:
                                failure_count += 1
                                if dirpath not in failure_folders:
                                    failure_folders.append(dirpath)
            except json.JSONDecodeError:
                print(f"JSON 파싱 오류: {file_path}")
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {file_path}, 오류: {str(e)}")
    
    # 결과 출력
    print(f"Success 항목이 1인 경우: {success_count}개")
    for folder in success_folders:
        print(f"  - {folder}")
    
    print(f"\nSuccess 항목이 0인 경우: {failure_count}개")
    for folder in failure_folders:
        print(f"  - {folder}")

if __name__ == "__main__":
    find_eval_json_files()
```

```
import os
import json
import re
from collections import defaultdict

def find_eval_json_files():
    # 결과를 저장할 변수 초기화
    success_count = 0
    failure_count = 0
    success_folders = []
    failure_folders = []
    total_folders_checked = 0
    
    # 카테고리별 통계를 저장할 딕셔너리
    categories = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0, "success_folders": [], "failure_folders": []})
    
    # 현재 디렉토리부터 시작
    root_dir = '.'
    
    # 모든 디렉토리와 하위 디렉토리 순회
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 현재 디렉토리에 eval.json 파일이 있는지 확인
        if 'eval.json' in filenames:
            total_folders_checked += 1
            file_path = os.path.join(dirpath, 'eval.json')
            
            # 폴더명에서 카테고리 추출
            folder_name = os.path.basename(dirpath)
            category = None
            
            # Task로 시작하는 폴더명에서 카테고리 추출 (예: TaskABC--23--run0 -> ABC)
            match = re.search(r'Task([A-Za-z0-9]+)', folder_name)
            if match:
                category = match.group(1)
            else:
                category = "기타"  # 카테고리를 찾을 수 없는 경우
            
            # 카테고리 총 개수 증가
            categories[category]["total"] += 1
            
            try:
                # JSON 파일 읽기
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # 데이터가 리스트 형태인지 확인
                if isinstance(data, list):
                    for item in data:
                        # 각 항목에 success 키가 있는지 확인
                        if 'succes' in item:  # 'succes'로 오타가 있는 경우 처리
                            # success가 1인 경우
                            if item['succes'] == 1:
                                success_count += 1
                                if dirpath not in success_folders:
                                    success_folders.append(dirpath)
                                    categories[category]["success"] += 1
                                    categories[category]["success_folders"].append(dirpath)
                            # success가 0인 경우
                            elif item['succes'] == 0:
                                failure_count += 1
                                if dirpath not in failure_folders:
                                    failure_folders.append(dirpath)
                                    categories[category]["failure"] += 1
                                    categories[category]["failure_folders"].append(dirpath)
                        # 'success'로 올바르게 쓰여진 경우도 확인
                        elif 'success' in item:
                            # success가 1인 경우
                            if item['success'] == 1:
                                success_count += 1
                                if dirpath not in success_folders:
                                    success_folders.append(dirpath)
                                    categories[category]["success"] += 1
                                    categories[category]["success_folders"].append(dirpath)
                            # success가 0인 경우
                            elif item['success'] == 0:
                                failure_count += 1
                                if dirpath not in failure_folders:
                                    failure_folders.append(dirpath)
                                    categories[category]["failure"] += 1
                                    categories[category]["failure_folders"].append(dirpath)
            except json.JSONDecodeError:
                print(f"JSON 파싱 오류: {file_path}")
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {file_path}, 오류: {str(e)}")
    
    # 전체 결과 출력
    print(f"총 체크한 폴더 개수: {total_folders_checked}개")
    print(f"Success 항목이 1인 경우: {success_count}개")
    print(f"Success 항목이 0인 경우: {failure_count}개")
    
    # 카테고리별 결과 출력
    print("\n## 카테고리별 결과")
    print(f"총 카테고리 개수: {len(categories)}개")
    
    for category, stats in categories.items():
        print(f"\n### 카테고리: {category}")
        print(f"총 폴더 개수: {stats['total']}개")
        print(f"Success 항목이 1인 경우: {stats['success']}개")
        if stats['success'] > 0:
            print("  성공한 폴더:")
            for folder in stats['success_folders']:
                print(f"  - {folder}")
        
        print(f"Success 항목이 0인 경우: {stats['failure']}개")
        if stats['failure'] > 0:
            print("  실패한 폴더:")
            for folder in stats['failure_folders']:
                print(f"  - {folder}")

if __name__ == "__main__":
    find_eval_json_files()
```

이 프로그램은 다음과 같은 작업을 수행합니다:

1. 현재 디렉토리에서 시작하여 모든 하위 디렉토리를 재귀적으로 탐색합니다[3][6].
2. 각 디렉토리에서 'eval.json' 파일을 찾습니다[7][8].
3. 파일을 찾으면 JSON 형식으로 파싱하고 'success' 항목을 확인합니다[2][5].
4. success 값이 1이면 성공 카운터를 증가시키고 해당 폴더 이름을 저장합니다.
5. success 값이 0이면 실패 카운터를 증가시키고 해당 폴더 이름을 저장합니다.
6. 최종적으로 성공 및 실패 케이스의 개수와 해당 폴더 이름을 출력합니다.

이 프로그램은 JSON 파싱 오류나 기타 예외 상황도 처리하여 안정적으로 동작합니다.
