# 이미지 URL을 제외하고 파일에 쓰는 방식으로 코드를 수정하겠습니다.

```python
# Stage 1. Query
print(">> Stage 1. Query")
# 이미지 URL을 제외한 메시지만 파일에 쓰기
filtered_messages = self.filter_image_urls(messages)
with open("monitor.log", "w") as f:
    f.write(json.dumps(filtered_messages, indent=4))
description = self.agent.act(messages)
print(description, end="\n\n")
with open("monitor.log", "w") as f:
    f.write(description)
messages.append({"role": "assistant", "content": description})
messages.append({"role": "user", "content": referring_user_prompt})

# Stage 2. Referring
print(">> Stage 2. Referring")
# 이미지 URL을 제외한 메시지만 파일에 쓰기
filtered_messages = self.filter_image_urls(messages)
with open("monitor.log", "w") as f:
    f.write(json.dumps(filtered_messages, indent=4))

referring = self.agent.act(messages)
print(referring, end="\n\n")
with open("monitor.log", "w") as f:
    f.write(referring)
```

그리고 이미지 URL을 필터링하는 함수를 추가해야 합니다:

```python
def filter_image_urls(self, messages):
    """이미지 URL을 제외한 메시지 복사본을 반환합니다."""
    filtered_messages = copy.deepcopy(messages)
    
    for message in filtered_messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            # content가 리스트인 경우 (멀티모달 메시지)
            filtered_content = []
            for item in message["content"]:
                if "image_url" not in item:
                    filtered_content.append(item)
                else:
                    # 이미지 URL이 있는 항목은 제외하거나 다른 정보로 대체할 수 있음
                    # 예: filtered_content.append({"image_placeholder": "이미지 URL 생략"})
                    pass
            message["content"] = filtered_content
    
    return filtered_messages
```

이 코드는 메시지 목록에서 이미지 URL을 포함하는 항목을 필터링하여 로그 파일에 쓸 때는 이미지 URL을 제외합니다. 원본 메시지는 그대로 유지하면서 로그 파일에 쓰는 용도로만 필터링된 복사본을 사용합니다.

---
