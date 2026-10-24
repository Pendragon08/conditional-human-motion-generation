"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._add_standing_windows()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _add_standing_windows(self):
        """Add start/end standing posture windows with transition interpolation.

        This post-process extends motion at both ends:
        - A fixed window (duration FIXED_SEC) with standing pose and fixed xy/rot, z=0.78
        - A transition window (duration WINDOW_SEC) blending standing pose and the first/last original frame
        """
        root_pos = self.motion_base_poss
        root_rot = self.motion_base_rots
        dof_pos = self.motion_dof_poss

        print(f"root_pos: {root_pos.shape}, root_rot: {root_rot.shape}, dof_pos: {dof_pos.shape}")

        num_frames = root_pos.shape[0]
        if num_frames <= 1:
            return

        aligned_fps = self.output_fps
        WINDOW_SEC = 0.5
        FIXED_SEC = 1.0
        W = max(1, min(int(WINDOW_SEC * aligned_fps), num_frames // 2))
        F = max(1, min(int(FIXED_SEC * aligned_fps), max(0, num_frames - 2 * W)))

        D = dof_pos.shape[1]
        target = torch.zeros((D,), dtype=dof_pos.dtype, device=self.device)

        if D == 29:
            idx_vals = [
                (0, -0.1), (1, 0.0), (2, 0.0),
                (3, 0.3), (4, -0.2), (5, 0.0),
                (6, -0.1), (7, 0.0), (8, 0.0),
                (9, 0.3), (10, -0.2), (11, 0.0),
                (12, 0.0), (13, 0.0), (14, 0.0),
                (15, 0.2), (16, 0.2), (17, 0.0), (18, 0.9),
                (19, 0.0), (20, 0.0), (21, 0.0),
                (22, 0.2), (23, -0.2), (24, 0.0), (25, 0.9),
                (26, 0.0), (27, 0.0), (28, 0.0),
            ]
            for idx, val in idx_vals:
                if idx < D:
                    target[idx] = val
        if D == 23:
            idx_vals = [
                (0, -0.1), (1, 0.0), (2, 0.0),
                (3, 0.3), (4, -0.2), (5, 0.0),
                (6, -0.1), (7, 0.0), (8, 0.0),
                (9, 0.3), (10, -0.2), (11, 0.0),
                (12, 0.0),
                (13, 0.2), (14, 0.2), (15, 0.0), (16, 0.9),
                (17, 0.0),
                (18, 0.2), (19, -0.2), (20, 0.0), (21, 0.9),
                (22, 0.0),
            ]
            for idx, val in idx_vals:
                if idx < D:
                    target[idx] = val

        head = F + W
        tail = F + W
        new_n = head + num_frames + tail

        inter_root_pos = torch.zeros((new_n, 3), dtype=root_pos.dtype, device=self.device)
        inter_root_rot = torch.zeros((new_n, 4), dtype=root_rot.dtype, device=self.device)
        inter_dof_pos = torch.zeros((new_n, D), dtype=dof_pos.dtype, device=self.device)

        inter_root_pos[head : head + num_frames] = root_pos
        inter_root_rot[head : head + num_frames] = root_rot
        inter_dof_pos[head : head + num_frames] = dof_pos

        if F > 0:
            inter_root_pos[:F, 2] = 0.78
            inter_root_pos[:F, :2] = root_pos[0, :2]
            # 固定窗口保持与 z 轴平行（yaw-only）
            inter_root_rot[:F] = self._yaw_only(root_rot[0])
            inter_dof_pos[:F] = target

            inter_root_pos[-F:, 2] = 0.78
            inter_root_pos[-F:, :2] = root_pos[-1, :2]
            inter_root_rot[-F:] = self._yaw_only(root_rot[-1])
            inter_dof_pos[-F:] = target

        if W > 0 and F > 0:
            s0 = F
            s1 = F + W
            alphas = torch.linspace(0.0, 1.0, s1 - s0, dtype=dof_pos.dtype, device=self.device)
            inter_root_pos[s0:s1, :2] = root_pos[0, :2]
            for t in range(alphas.shape[0]):
                a = alphas[t]
                # 过渡窗口旋转直接插值：从固定窗口末尾（yaw-only）到首帧原始姿态
                q_from = inter_root_rot[s0 - 1]
                q_to = root_rot[0]
                inter_root_rot[s0 + t] = quat_slerp(q_from, q_to, a)
                inter_root_pos[s0 + t, 2] = a * root_pos[0, 2] + (1 - a) * inter_root_pos[0, 2]
                inter_dof_pos[s0 + t, :] = a * dof_pos[0, :] + (1 - a) * target

        if W > 0 and F > 0:
            e1 = new_n - F
            e0 = e1 - W
            alphas = torch.linspace(0.0, 1.0, e1 - e0, dtype=dof_pos.dtype, device=self.device)
            inter_root_pos[e0:e1, :2] = root_pos[-1, :2]
            for k in range(alphas.shape[0]):
                a = alphas[k]
                # 过渡窗口（尾部）旋转直接插值：从末帧原始姿态到固定窗口首个（yaw-only）
                q_from = root_rot[-1]
                q_to = inter_root_rot[e1]
                inter_root_rot[e0 + k] = quat_slerp(q_from, q_to, a)
                inter_root_pos[e0 + k, 2] = (1 - a) * root_pos[-1, 2] + a * inter_root_pos[-1, 2]
                inter_dof_pos[e0 + k, :] = (1 - a) * dof_pos[-1, :] + a * target

        self.motion_base_poss = inter_root_pos
        self.motion_base_rots = inter_root_rot
        self.motion_dof_poss = inter_dof_pos
        self.output_frames = new_n
        # Informative print
        print(
            f"Motion post-processed with standing windows: head={head}, tail={tail}, total_frames={self.output_frames}"
        )
        print(f"motion_base_poss: {self.motion_base_poss.shape}, motion_base_rots: {self.motion_base_rots.shape}, motion_dof_poss: {self.motion_dof_poss.shape}")
        print(f"output_frames: {self.output_frames}")

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _yaw_only(self, quats_wxyz: torch.Tensor) -> torch.Tensor:
        """Project a batch of quaternions (wxyz) to yaw-only (rotation about z-axis).

        Args:
            quats_wxyz: shape (, 4) in wxyz ordering.
        Returns:
            shape (, 4) yaw-only quaternions in wxyz ordering.
        """
        w = quats_wxyz[0]
        x = quats_wxyz[1]
        y = quats_wxyz[2]
        z = quats_wxyz[3]

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        half_yaw = 0.5 * yaw
        cy = torch.cos(half_yaw)
        sy = torch.sin(half_yaw)

        out = torch.zeros_like(quats_wxyz)
        out[0] = cy
        out[1] = 0.0
        out[2] = 0.0
        out[3] = sy
        return out

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        bool,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                # print(f"{k}: {log[k].__class__}, len={len(log[k])}, first shape={log[k][0].shape if len(log[k]) > 0 else None}")
                log[k] = np.stack(log[k], axis=0)

            # Motion interpolated, input frames: 3945, input fps: 30, output frames: 6574, output fps: 50
            # joint_pos: <class 'list'>, len=6574, first shape=(29,)
            # joint_vel: <class 'list'>, len=6574, first shape=(29,)
            # body_pos_w: <class 'list'>, len=6574, first shape=(30, 3)
            # body_quat_w: <class 'list'>, len=6574, first shape=(30, 4)
            # body_lin_vel_w: <class 'list'>, len=6574, first shape=(30, 3)
            # body_ang_vel_w: <class 'list'>, len=6574, first shape=(30, 3)

            # import ipdb; ipdb.set_trace()
            np.savez("/tmp/motion.npz", **log)

            import wandb

            COLLECTION = args_cli.output_name
            run = wandb.init(project="csv_to_npz", name=COLLECTION)  # type: ignore[attr-defined]
            print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
            REGISTRY = "motions"
            logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY)
            run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
            print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
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
        ],
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
