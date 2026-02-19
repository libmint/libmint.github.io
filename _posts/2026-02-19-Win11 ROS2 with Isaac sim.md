목표가 **Isaac Sim ↔ ROS 2 Humble 연동**이면, Windows 11에서 “ROS 2를 윈도우에 직접 깔기”보다 **Isaac Sim은 Windows에서 돌리고, ROS 2는 WSL2(Ubuntu 22.04)에서 돌리는 방식**이 문서/커뮤니티 기준으로 제일 흔하고 안정적입니다. (Windows 11은 Humble 지원) ([Isaac Sim Documentation][1])

아래는 **Windows 11 + Isaac Sim(Windows) + ROS2 Humble(WSL2 Ubuntu 22.04)** 기준의 “바로 되는” 절차입니다.

---

## 1) WSL2 + Ubuntu 22.04 설치 (Windows)

PowerShell(관리자):

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

재부팅 후 Ubuntu(WSL) 실행.

---

## 2) WSL(Ubuntu 22.04) 안에 ROS 2 Humble 설치

WSL Ubuntu에서:

```bash
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

(이 단계는 ROS 공식 설치 흐름 그대로입니다.)

---

## 3) Isaac Sim 쪽: ROS2 Bridge 확장 활성화

Isaac Sim 실행 후:

* **Window → Extensions**
* 검색창에 `ros2` 입력
* **`omni.isaac.ros2_bridge` (ROS2 Bridge)** 활성화 ([MathWorks][2])

> 참고: Isaac Sim 문서에서 Windows의 ROS2 브리지는 “WSL2(Ubuntu 22.04)를 ROS 라이브러리/ROS 실행에 사용”하는 구성을 안내합니다. ([Isaac Sim Documentation][3])

---

## 4) DDS/RMW를 “서로 맞추기” (중요)

연동이 안 되는 대부분의 이유가 **DDS 미들웨어 불일치 / 네트워크(멀티캐스트) / 방화벽**입니다.

가장 무난한 조합은 **Fast DDS**(기본값인 경우가 많음)로 통일하는 겁니다.
WSL 터미널에서 다음을 먼저 맞춰주세요:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
```

그리고 Isaac Sim도 같은 `ROS_DOMAIN_ID`를 쓰도록 맞추세요(환경변수 또는 Isaac Sim 실행 환경에서 동일하게).

---

## 5) 토픽 송수신 테스트 (최소 예제)

### (A) WSL에서 listener

```bash
ros2 run demo_nodes_cpp listener
```

### (B) Isaac Sim에서 talker 역할(예: 카메라/클럭/TF 등)을 퍼블리시하는 샘플이나, ROS2 브리지 예제 액션 실행

* Isaac Sim 예제(ROS2) 실행 후
* WSL에서:

```bash
ros2 topic list
ros2 topic echo /clock
```

토픽이 보이면 성공.

---

## 6) “토픽이 안 보임” 흔한 해결책 3개

### 1) Windows 방화벽 예외

Windows 방화벽이 DDS(UDP 멀티캐스트/유니캐스트)를 막으면 WSL↔Windows 간 discovery가 안 됩니다.
Isaac Sim, Omniverse 관련 exe에 대해 **Private 네트워크 허용**을 먼저 확인하세요. (증상: Isaac Sim은 뜨는데 WSL에서 토픽 0개) ([NVIDIA Developer Forums][4])

### 2) WSL 네트워크(미러 모드/멀티캐스트)

Windows 11의 WSL 네트워킹 모드에 따라 DDS discovery가 꼬일 수 있습니다. “WSL에서 ROS2 토픽이 안 뜬다”류 이슈가 포럼에 꾸준히 있습니다. ([NVIDIA Developer Forums][4])
이 경우는 `wsl --shutdown` 후 재시작, 또는 WSL 네트워크 설정(미러링/브리지)을 바꾸면 해결되는 케이스가 많습니다.

### 3) CycloneDDS로 바꿨는데 문제

CycloneDDS는 조합에 따라 Isaac Sim 쪽에서 기능/액션 그래프가 비정상이라는 최근 이슈도 보고됩니다. 우선은 Fast DDS로 먼저 성공시키는 걸 추천합니다. ([NVIDIA Developer Forums][5])

---

## 추천 결론

* **가장 추천:** `Windows 11 (Isaac Sim)` + `WSL2 Ubuntu 22.04 (ROS2 Humble)` ([Isaac Sim Documentation][3])
* “윈도우에 ROS2 Humble 네이티브 설치”도 가능은 하지만, Isaac Sim 연동만 보면 굳이 이 길로 갈 이유가 적습니다(문서도 WSL 구성을 중심으로 안내). ([Isaac Sim Documentation][3])

---


[1]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html?utm_source=chatgpt.com "ROS 2 Installation - Isaac Sim Documentation"
[2]: https://la.mathworks.com/help/ros/ug/set-up-and-connect-to-nvidia-isaac-sim.html?utm_source=chatgpt.com "Set Up and Connect to NVIDIA Isaac Sim"
[3]: https://docs.isaacsim.omniverse.nvidia.com/4.2.0/installation/install_ros.html?utm_source=chatgpt.com "ROS and ROS 2 Installation - Isaac Sim Documentation"
[4]: https://forums.developer.nvidia.com/t/troubleshooting-missing-ros2-topics-in-isaac-sim-on-windows-11-with-wsl/315941?utm_source=chatgpt.com "Missing ROS2 Topics in Isaac Sim on Windows 11 with WSL"
[5]: https://forums.developer.nvidia.com/t/ros2-bridge-does-not-work-with-cyclone-dds/356766?utm_source=chatgpt.com "ROS2 bridge does not work with Cyclone DDS - Isaac Sim"
