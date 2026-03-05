이번에는 **Isaac Sim 5.1.0 (Windows)** 기준으로 **실제로 존재하는 메뉴와 NVIDIA 공식 API 흐름 기준**으로,
**Isaac Sim을 막 켠 상태부터 Deformable(Soft Body) 객체를 만드는 과정**을 처음부터 끝까지 설명하겠습니다.
(불필요한 옛 버전 UI는 제외합니다)

---

# 1️⃣ Isaac Sim 실행 후 새 씬 만들기

Isaac Sim 실행 후 첫 화면에서

```
File → New
```

새 Stage를 만듭니다.

Stage 구조

```
/World
```

만 남아있는 상태가 됩니다.

---

# 2️⃣ Physics Scene 확인

상단 메뉴

```
Create → Physics → Physics Scene
```

생성되면 Stage

```
/World
   PhysicsScene
```

Isaac Sim은 physics scene이 있어야 시뮬레이션이 동작합니다.

---

# 3️⃣ Ground Plane 생성

바닥이 필요합니다.

```
Create → Physics → Ground Plane
```

Stage

```
/World
   GroundPlane
```

---

# 4️⃣ Script Editor 열기

Isaac Sim 5.x에서 deformable은 **GUI가 아니라 Python API로 생성하는 것이 공식 방식**입니다.

상단 메뉴

```
Window → Script Editor
```

아래쪽에 Python 콘솔이 열립니다.

---

# 5️⃣ Deformable 생성 코드 실행

Script Editor에 아래 코드를 **전체 복사 후 Run**합니다.

```python
from pxr import UsdGeom, Gf
import omni.usd
from omni.physx.scripts import deformableUtils

stage = omni.usd.get_context().get_stage()

# root Xform
root = UsdGeom.Xform.Define(stage, "/World/SoftBody")

# render mesh (cube)
cube = UsdGeom.Cube.Define(stage, "/World/SoftBody/RenderMesh")
cube.AddTranslateOp().Set(Gf.Vec3f(0,0,100))

# deformable 생성
deformableUtils.create_auto_volume_deformable_hierarchy(
    stage=stage,
    root_prim_path="/World/SoftBody",
    simulation_tetmesh_path="/World/SoftBody/SimulationMesh",
    collision_tetmesh_path="/World/SoftBody/CollisionMesh",
    cooking_src_mesh_path="/World/SoftBody/RenderMesh"
)
```

---

# 6️⃣ Stage 구조 확인

코드를 실행하면 Stage가 이렇게 바뀝니다.

```
/World
   GroundPlane
   PhysicsScene
   SoftBody
      RenderMesh
      SimulationMesh
      CollisionMesh
```

각 의미

| 항목             | 설명                      |
| -------------- | ----------------------- |
| RenderMesh     | 화면에 보이는 메쉬              |
| SimulationMesh | 물리 계산용 tetrahedral mesh |
| CollisionMesh  | 충돌 계산                   |

---

# 7️⃣ Simulation 실행

상단 버튼

```
PLAY
```

실행하면

```
cube가 떨어지면서 변형
```

됩니다.

즉 **soft body deformable simulation**이 동작합니다.

---

# 8️⃣ Deformable 내부 mesh 확인

내부 tetrahedral mesh를 보려면

상단 메뉴

```
Window → Physics Debug
```

체크

```
Tetrahedral Mesh
```

그러면 내부 구조가 보입니다.

---

# 9️⃣ Deformable 주요 파라미터

Stage에서

```
SoftBody
```

선택

Property 창에서 다음을 조정할 수 있습니다.

### Density

```
Density
```

밀도

예

```
1000
```

---

### Young's Modulus

재질 강성

예

```
젤리 = 1000
고무 = 10000
플라스틱 = 100000
```

---

### Poisson Ratio

변형시 옆 방향 수축

```
0.3 ~ 0.45
```

---

### Damping

진동 감쇠

```
0.01
```

---

# 🔎 가장 흔한 문제

### deformable이 안 움직임

대부분 이유

1️⃣ PhysicsScene 없음
2️⃣ GroundPlane 없음
3️⃣ 코드 실행 안됨

---

# 📌 Isaac Sim 5.x deformable 구조 (중요)

Soft body는 이렇게 구성됩니다.

```
Render Mesh
     ↓
Tetrahedral Mesh
     ↓
PhysX FEM Simulation
```

PhysX는 **FEM (Finite Element Method)** 기반입니다.

---


# 🚀 추가로 매우 중요한 내용

Isaac Sim에서 deformable을 사용할 때 **연구나 로봇 시뮬레이션에서는 아래 3개가 핵심입니다.**

### 1️⃣ Deformable + 로봇 그리퍼

soft grasping 연구

### 2️⃣ Cloth simulation

천 조작 연구

### 3️⃣ Food / Soft material manipulation

음식 조작 연구

---


특히 **Isaac Sim에서 deformable을 제대로 쓰려면 반드시 알아야 하는 설정 7개**가 있습니다.
