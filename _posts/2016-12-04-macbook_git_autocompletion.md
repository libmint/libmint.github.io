---
layout: post
title: 맥북 Git 자동완성
---

1. 아래 링크에서 git-completion.bash 파일을 다운로드 한다.
    - https://github.com/git/git/blob/master/contrib/completion/git-completion.bash
2. 다운로드 받은 파일을 자신의 계정의 홈디렉토리에 복사한다.
    - $ cp git-completion.bash ~/
3. ~/.bash_profile 파일에 아래의 명령을 추가한다.
    - source ~/git-completion.bash