Isaac Sim에서 순수 물리 시뮬레이션 연구를 위한 **1D Deformable (Particle Rope / PBD 기반 로프)** 생성 방법을 바로 설명해 드립니다.

PhysX 5의 1D Deformable은 내부적으로 **PBD(Position Based Dynamics)** 입자 시스템을 사용합니다. 얇은 선 위에 수십 개의 파티클(점)을 일렬로 배치하고, 그 사이를 거리(Distance) 및 굽힘(Bending) 스프링으로 강하게 연결하여 **Cosserat Rod(코세라 로드)**나 탄성 빔(Beam)과 같은 물리적 비틀림과 텐션을 모사합니다.

빈 스테이지(`File > New`)에서 아래 코드를 실행해 주십시오.

---

### 1D Deformable 전선(Cable) 생성 스크립트

```python
import omni.usd
import omni.kit.commands
from pxr import Gf, Sdf, UsdPhysics, PhysxSchema
import omni.physx.scripts.physicsUtils as physicsUtils
import omni.physx.scripts.particleUtils as particleUtils

stage = omni.usd.get_context().get_stage()

# 1. Physics Scene 및 바닥 세팅 (PBD 입자 연산은 GPU가 필수입니다)
scene_path = "/physicsScene"
if not stage.GetPrimAtPath(scene_path):
    physicsUtils.add_physics_scene(stage, scene_path)

physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
physx_scene.CreateEnableGPUDynamicsAttr(True)
physx_scene.CreateBroadphaseTypeAttr("GPU")
physx_scene.CreateSolverTypeAttr("TGS")

ground_path = "/World/GroundPlane"
if not stage.GetPrimAtPath(ground_path):
    physicsUtils.add_ground_plane(stage, ground_path, "Z", 1500.0, Gf.Vec3f(0.0), Gf.Vec3f(0.5))

# 2. PBD 파티클 시스템 생성 (1D Deformable의 두뇌 역할)
particle_system_path = "/World/WireParticleSystem"
if stage.GetPrimAtPath(particle_system_path):
    omni.kit.commands.execute("DeletePrims", paths=[particle_system_path])

# 파티클 간의 충돌 거리와 연산 반복 횟수(정밀도)를 설정합니다.
particleUtils.add_physx_particle_system(
    stage,
    particle_system_path,
    contact_offset=0.02,
    rest_offset=0.02,
    particle_contact_offset=0.02,
    solid_rest_offset=0.02,
    fluid_rest_offset=0.0,
    solver_position_iterations=32 # 전선이 늘어나는 것을 막기 위해 반복 연산을 높입니다.
)

# 3. 전선 경로(Points) 데이터 생성
num_particles = 40
wire_length = 2.0 # 2미터
spacing = wire_length / num_particles
# Z축 3.0 높이에서 X축 방향으로 일렬로 배열된 좌표 리스트
positions = [Gf.Vec3f(i * spacing, 0.0, 3.0) for i in range(num_particles)]

# 4. 1D Deformable 파티클 로프 생성
wire_path = "/World/DeformableWire"
if stage.GetPrimAtPath(wire_path):
    omni.kit.commands.execute("DeletePrims", paths=[wire_path])

# [핵심 파라미터] 여기서 전선의 '빳빳함'이 결정됩니다.
particleUtils.add_physx_particle_rope(
    stage,
    wire_path,
    particle_system_path,
    positions=positions,
    radius=0.02,               # 전선의 두께 (2cm)
    drop_endpoints=False,
    rope_stiffness=10000.0,    # 길이 방향으로 늘어나지 않게 버티는 장력
    rope_damping=10.0,         # 출렁거림을 줄이는 감쇠력
    bending_stiffness=500.0,   # ★ 굽힘 강성 (이 값이 높으면 빳빳한 파워 케이블, 낮으면 부드러운 실이 됩니다)
    bending_damping=5.0
)

# 5. 시각적 표현을 위한 렌더링 설정
wire_prim = stage.GetPrimAtPath(wire_path)
# 파티클 자체를 화면에 구(Sphere) 형태로 렌더링하도록 강제 활성화
wire_prim.CreateAttribute("primvars:points", Sdf.ValueTypeNames.Point3fArray).Set(positions)
wire_prim.CreateAttribute("widths", Sdf.ValueTypeNames.FloatArray).Set([0.04] * num_particles)

print("1D Deformable 전선 세팅 완료. Play를 눌러주세요.")

```

### 1D Deformable (Particle Rope) 제어 팩트

* **Bending Stiffness (굽힘 강성):** 코드를 실행해 보시면 전선이 바닥에 닿을 때 일반 실처럼 완전히 뭉치지 않고, 두꺼운 케이블처럼 둥글게 곡선을 유지하며 똬리를 틉니다. `bending_stiffness` 값을 0.0으로 낮추면 신발끈처럼 힘없이 무너집니다.
* **Stretching 방지:** 물리 엔진이 1D 선을 계산할 때 무거운 물체를 매달면 고무줄처럼 늘어나는 고질적인 문제가 있습니다. 이를 잡기 위해 코드 상단의 `solver_position_iterations=32`를 높게 주어(기본값 4) 연산 정밀도를 강제로 끌어올렸습니다.

Play(▶)를 눌러 바닥에 떨어지는 형태를 관찰해 보십시오.
확인이 끝나시면, 앞서 언급했던 **강화학습/로봇 제어용 '다관절 체인(Articulation)' 전선 스크립트**를 바로 제공해 드릴까요? 아니면 이 1D 전선의 한쪽 끝을 벽에 고정(Anchor)하는 방법을 먼저 해보시겠습니까?
