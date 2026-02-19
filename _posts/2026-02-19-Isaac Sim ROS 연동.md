오케이, (2) **설치 폴더에서 `isaac-sim.bat` 직접 실행**이면 해결이 제일 깔끔합니다.
지금 에러(`could not find assets root folder`)는 **샘플 씬이 참조하는 “Assets Root(기본은 인터넷 S3)”를 못 찾아서** 나는 거라, **로컬 Assets 팩을 받고 Isaac Sim에 “로컬 루트”를 지정**해주면 바로 해결됩니다. ([NVIDIA Developer Forums][1])

아래 순서대로 하세요.

---

## 1) Isaac Sim 5.1 “Local Assets Complete Pack” 다운로드

NVIDIA 문서에서 **Local Assets Packs(air-gapped/로컬용)**를 안내합니다. “Complete Pack(zip)”을 받아 로컬에 풀어 쓰는 방식이에요. ([Isaac Sim Documentation][2])

* 추천 위치(예시): `C:\isaacsim_assets\Assets\Isaac\5.1\`

압축을 풀고 나면 **중요 조건**이 있습니다:

✅ **루트 폴더 아래에 `Isaac` 폴더와 `NVIDIA` 폴더가 둘 다 있어야** 합니다.
(이게 없으면 루트를 잡아도 계속 “assets root” 못 찾습니다.) ([Isaac Sim Documentation][2])

예:

```
C:\isaacsim_assets\Assets\Isaac\5.1\
  ├─ Isaac\
  └─ NVIDIA\
```

---

## 2) `isaac-sim.bat` 실행 시 “로컬 asset_root”를 플래그로 지정 (가장 확실)

설치 폴더에서 PowerShell/CMD로 Isaac Sim을 이렇게 실행하세요:

```powershell
cd <IsaacSim_설치폴더>
.\isaac-sim.bat --/persistent/isaac/asset_root/default="C:/isaacsim_assets/Assets/Isaac/5.1"
```

이 플래그 방식이 문서에 있는 정석 루트입니다. ([Isaac Sim Documentation][2])

> 경로는 **슬래시(`/`)로 써도** Windows에서 잘 먹습니다(위 예시처럼).

---

## 3) (선택) 영구 적용: user.config.json에 asset_root 박기

매번 플래그 주기 싫으면 config 파일에 저장할 수 있어요.

Windows에서 standalone(직접 설치 실행)일 때 user config가 보통 여기로 잡힙니다:
`C:\Users\<사용자>\AppData\Local\ov\data\Kit\Isaac-Sim Full\user.config.json` ([GitHub][3])

`user.config.json`에 아래를 넣거나 수정:

```json
{
  "persistent": {
    "isaac": {
      "asset_root": {
        "default": "C:/isaacsim_assets/Assets/Isaac/5.1"
      }
    }
  }
}
```

---

## 4) 다시 샘플 씬 로드

이제 Isaac Sim 재실행(2번 플래그 방식이면 그걸로 실행) →
`Robotics Examples → ROS2 → Isaac ROS → Sample Scene` 다시 Load 하면 에러가 없어야 합니다.

---

# ROS2 테스트까지 한 번에 (추천 실행 커맨드)

어차피 목표가 ROS2 연동이니, 같은 실행에서 환경변수까지 같이 맞추면 좋아요.

### Windows(PowerShell)에서 Isaac Sim 실행 전에:

```powershell
$env:ROS_DOMAIN_ID="0"
$env:RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
cd <IsaacSim_설치폴더>
.\isaac-sim.bat --/persistent/isaac/asset_root/default="C:/isaacsim_assets/Assets/Isaac/5.1"
```

### WSL(Ubuntu)에서:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

샘플 씬 로드 + **Play(시뮬 시작)** 후, WSL에서:

```bash
ros2 topic list
ros2 topic echo /clock
```

`/clock` 찍히면 **Isaac Sim → ROS2 통신 OK**입니다.

---

## 자주 막히는 포인트 2개만 체크

1. 로컬 Assets 루트에 **`Isaac/`와 `NVIDIA/`가 둘 다 있는지** ([Isaac Sim Documentation][2])
2. asset_root가 아직 S3로 잡혀 있으면(사내 프록시/차단) 계속 실패합니다 → 로컬로 바꿔야 함 ([NVIDIA Developer Forums][1])

---

### 암기팁

**“assets root 에러 = 인터넷(S3) 막힘 → 로컬 assets 팩 + asset_root를 C:로 지정”**

---

원하시면, 지금

* 로컬에 푼 폴더의 최상위(= `C:\isaacsim_assets\Assets\Isaac\5.1\`) 스크린샷(폴더 목록만)
* 또는 그 폴더에서 `dir` 결과

딱 하나만 주시면, **루트 경로가 정확히 어디여야 하는지** 제가 바로 찍어서 커맨드를 완성형으로 맞춰드릴게요.

[1]: https://forums.developer.nvidia.com/t/could-not-find-assets-root-folder/353359?utm_source=chatgpt.com "Could not find assets root folder - Isaac Sim"
[2]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_faq.html?utm_source=chatgpt.com "Setup Tips — Isaac Sim Documentation"
[3]: https://github.com/isaac-sim/IsaacSim/issues/376?utm_source=chatgpt.com "Isaac Sim (standalone 5.1) fails to add extension search ..."
