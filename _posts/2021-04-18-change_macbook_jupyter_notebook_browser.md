---
layout: post
title: 맥북 주피터 노트북 브라우저 변경
---

맥북에서 아나콘다를 설치하면 주피터노트북이 기본으로 설치가 됩니다.

주피터노트북은 브라우저를 이용하여 동작이 되는데

맥북에서는 기본적으로 사파리에서 주피터노트북이 구동이 됩니다.

근데 아마도 크롬 부라우저에서 주피터노트북을 사용하기 원하시는 분들도 있을겁니다.

(예를들어 chromedriver를 사용하고자 하는 경우)

주피터노트북의 설정을 변경하는 방법입니다.

다음과 같이 config 파일을 수정하여 브라우저를 변경할 수 있습니다.

```bash
$ cd ~/.jupyter/
$ vi jupyter_notebook_config.py

# 만약 jupyter_notebook_config.py 파일이 없으면 아래 명령으로 생성
$ jupyter notebook --generate-config

# jupyter_notebook_config.py 파일에 아래 내용 추가
c.NotebookApp.browser = 'open -a /Applications/Google\ Chrome.app %s'
```