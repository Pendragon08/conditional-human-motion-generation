"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python scripts/replay_npz_clean.py --motion_file /ssd/humanoid/dataset/mocap_0128_add/npz/mocap_mocap_sofa_1_1.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch
import time
import os

# 在启动前设置环境变量禁用默认天空
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, required=False, help="The name of the wand registry (when not using --motion_file).")
parser.add_argument("--motion_file", type=str, required=False, help="Path to a local motion .npz file.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # 不使用地面
    # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 使用白色 dome light 提供均匀白色光照
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=800.0,  # 降低强度
            color=(1.0, 1.0, 1.0),  # 纯白色光
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Extract scene entities
    robot = scene["robot"]
    joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    print(f"robot_joint_indexes: {robot_joint_indexes}")
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    import pathlib
    motion_file: str

    if args_cli.motion_file:
        # Use provided local file
        local_path = pathlib.Path(args_cli.motion_file)
        if not local_path.exists():
            raise FileNotFoundError(f"Motion file not found: {local_path}")
        motion_file = str(local_path)
    else:
        # Fallback to W&B artifact
        if not args_cli.registry_name:
            raise ValueError("Either --motion_file must be provided, or --registry_name must be set.")
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"

        from wandb.apis.public import Api

        api = Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    print(f"motion.body_pos_w.shape: {motion.body_pos_w.shape}")
    print(f"motion.body_quat_w.shape: {motion.body_quat_w.shape}")
    print(f"motion.body_lin_vel_w.shape: {motion.body_lin_vel_w.shape}")
    print(f"motion.body_ang_vel_w.shape: {motion.body_ang_vel_w.shape}")
    print(f"motion.joint_pos.shape: {motion.joint_pos.shape}")
    print(f"motion.joint_vel.shape: {motion.joint_vel.shape}")
    print(f"body_quat_w: {motion.body_quat_w[0]}")
    print(f"body_quat_w: {motion.body_quat_w[-1]}")
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # 根据机器人朝向创建背景板
    import omni.usd
    from pxr import Gf, UsdGeom, UsdShade, Sdf
    from scipy.spatial.transform import Rotation as R
    
    stage = omni.usd.get_context().get_stage()
    
    # 获取机器人初始位置和朝向
    initial_pos = motion.body_pos_w[0][0].cpu().numpy() + scene.env_origins[0].cpu().numpy()
    initial_quat = motion.body_quat_w[0][0].cpu().numpy()  # (w, x, y, z) 或 (x, y, z, w)
    
    # 将四元数转换为旋转矩阵，获取前向方向
    # Isaac Sim 通常使用 (w, x, y, z) 格式
    rot = R.from_quat([initial_quat[1], initial_quat[2], initial_quat[3], initial_quat[0]])  # scipy 用 (x,y,z,w)
    forward = rot.apply([1, 0, 0])  # 机器人前向方向（假设 x 轴为前向）
    
    print(f"Robot forward direction: {forward}")
    
    # 背景板放在机器人后方（前向的反方向）
    bg_distance = 15  # 背景板距离机器人的距离
    bg_center = initial_pos - forward * bg_distance  # 在机器人后方
    
    # 计算背景板的法线方向（朝向机器人/相机）
    # 背景板需要垂直于 forward 方向
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # 创建黑色材质
    black_mat_path = "/World/BlackScreenMaterial"
    black_mat = UsdShade.Material.Define(stage, black_mat_path)
    shader = UsdShade.Shader.Define(stage, f"{black_mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    black_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    
    # 创建背景板，以 bg_center 为中心，宽 100 米，高 80 米
    half_width = 50
    half_height = 40
    p1 = bg_center - right * half_width + up * (-half_height + 1)  # 左下（稍微抬高避免穿地）
    p2 = bg_center + right * half_width + up * (-half_height + 1)  # 右下
    p3 = bg_center + right * half_width + up * half_height         # 右上
    p4 = bg_center - right * half_width + up * half_height         # 左上
    
    bg_plane = UsdGeom.Mesh.Define(stage, "/World/BlackScreenPlane")
    bg_plane.GetPointsAttr().Set([
        Gf.Vec3f(*p1), Gf.Vec3f(*p2), Gf.Vec3f(*p3), Gf.Vec3f(*p4)
    ])
    bg_plane.GetFaceVertexCountsAttr().Set([4])
    bg_plane.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])  # 法线朝向机器人
    bg_plane.GetPrim().CreateAttribute("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
    UsdShade.MaterialBindingAPI(bg_plane.GetPrim()).Bind(black_mat)
    
    print(f"Background plane center: {bg_center}")

    # 只在开始时设置相机位置，之后可以自由拖拽
    sim.set_camera_view(initial_pos + np.array([2.0, 2.0, 0.5]), initial_pos)

    # Simulation loop
    while simulation_app.is_running():
        loop_start = time.perf_counter()
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        # 节拍限制到 50Hz（0.02s）。若渲染/写入耗时已超过 dt，则不再额外 sleep。
        elapsed = time.perf_counter() - loop_start
        if elapsed < sim_dt:
            time.sleep(sim_dt - elapsed)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # 禁用网格
    import carb.settings
    settings = carb.settings.get_settings()
    settings.set("/app/viewport/grid/enabled", False)
    
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
