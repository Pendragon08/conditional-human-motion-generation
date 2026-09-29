# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.assets import Articulation, RigidObject

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--baseline", type=str, default="a", help="Name of the baseline.")
parser.add_argument("--id", type=str, default="0", help="Name of the id.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--path", type=str, default=None, help="Path to the task.")
parser.add_argument("--metrics-out", type=str, required=True,
                    help="Write fixed-policy physics rollout diagnostics to this JSON file.")
parser.add_argument("--eval-seed", type=int, default=42,
                    help="Fixed simulator and motion-sampling seed for paired evaluations.")
parser.add_argument("--freeze-task-goal-from-dir", type=str, default=None,
                    help="Custom diagnostic: use this raw-reference directory to hold goal, scene offset and event time constant across arms.")
parser.add_argument("--external-goals-json", type=Path, required=True,
                    help="Frozen pre-generation goals mapped into G1 scene coordinates before either arm is evaluated.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# exit(0)

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import random
import numpy as np

# from omni.isaac.core.utils.prims import set_prim_pose
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.motion_lib.motion_lib_base import MotionLibBase
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
# cfg = UrdfConverterCfg(
#     asset_path="/home/azureuser/IsaacLab/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf",     # Absolute path to your URDF file
#     usd_dir="/home/azureuser/IsaacLab/HumanoidVerse/humanoidverse/data/robots/g1",       # Directory to store the generated USD
#     usd_file_name="g1_27dof.usd",                # Optional: name for the USD file
#     force_usd_conversion=True,                # Optional: force re-generation
#     make_instanceable=True,                   # Optional: for memory efficiency
#     fix_base=True                             # Optional: fix the base link
# )

# converter = UrdfConverter(cfg)
# print("USD file generated at:", converter.usd_path)
# exit(0)
# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    # exit(0)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    random.seed(args_cli.eval_seed)
    np.random.seed(args_cli.eval_seed)
    torch.manual_seed(args_cli.eval_seed)
    env_cfg.seed = args_cli.eval_seed
    agent_cfg.seed = args_cli.eval_seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        print(resume_path)
        import pdb; pdb.set_trace()  # noqa: E702
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        # import pdb; pdb.set_trace()  # noqa: E702
    else:
        # print(log_root_path)
        if args_cli.path:
            # if path is provided, use it to get the checkpoint
            resume_path = args_cli.path
        else :
            task_names_ = args_cli.task.split("-")[3:-1]
            task_name = ""
            for task_name_ in task_names_:
                task_name += task_name_ + "-"
            task_name = task_name[:-1]
            if args_cli.baseline is not None:
                task_name += "-" + args_cli.baseline
            # import pdb; pdb.set_trace()  # noqa: E702
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, task_name=task_name)

    log_dir = os.path.dirname(resume_path)
    env_cfg.viewer.eye = (2.5,-5.,5.)
    env_cfg.viewer.lookat = (2., 0., 0.)
    env_cfg.observations.policy.enable_corruption = False
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    goal_rows = json.loads(args_cli.external_goals_json.read_text())["rows"]
    if not all(row.get("mapped_fixed_goal_xyz_m") is not None for row in goal_rows):
        raise ValueError("Goal file must use the fixed-scene target definition")
    if not json.loads(args_cli.external_goals_json.read_text()).get("target_is_fixed_per_motion"):
        raise ValueError("Rejecting a time-varying goal file")
    by_name = {f'{int(row["sample"])}.pkl': row for row in goal_rows}
    actual_order = [Path(path).name for path in env.unwrapped.motion_lib._motion_data_load]
    if set(actual_order) != set(by_name) or len(actual_order) != len(by_name):
        raise ValueError("External goal names must match the reference library exactly")
    fixed_goal = torch.as_tensor(
        [by_name[name]["mapped_fixed_goal_xyz_m"] for name in actual_order],
        device=env.unwrapped.device, dtype=torch.float32,
    )
    motion_lib = env.unwrapped.motion_lib
    motion_lib.grab_pos = fixed_goal.clone()
    # Official Kick places the motion so its target lies at scene x=2.1,y=0.
    # Freeze that translation from the *external* goal for both arms.
    scene_offsets = motion_lib.offsets.clone()
    scene_offsets[:, 0] = 2.1 - fixed_goal[:, 0]
    scene_offsets[:, 1] = -fixed_goal[:, 1]
    motion_lib.offsets = scene_offsets
    motion_lib.switch_idxs = torch.as_tensor(
        [by_name[name]["goal_anchor_frame"] for name in actual_order],
        device=env.unwrapped.device, dtype=motion_lib.switch_idxs.dtype,
    )
    print("[I1] Fixed external mapped goal, scene offset and event time for both arms", flush=True)
    if args_cli.freeze_task_goal_from_dir:
        baseline_motion_lib = MotionLibBase(
            env.unwrapped.num_envs, env.unwrapped.device,
            args_cli.freeze_task_goal_from_dir,
        )
        baseline_motion_lib.load_motions()
        actual_order = [Path(path).name for path in env.unwrapped.motion_lib._motion_data_load]
        baseline_order = [Path(path).name for path in baseline_motion_lib._motion_data_load]
        if actual_order != baseline_order:
            raise ValueError("Reference motion file order differs from frozen raw directory")
        for field in ("grab_pos", "offsets", "switch_idxs"):
            setattr(env.unwrapped.motion_lib, field, getattr(baseline_motion_lib, field).clone())
        print("[I1] Frozen raw task goal, offset and event time", flush=True)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    print(ISAACLAB_NUCLEUS_DIR)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": "eval_"+args_cli.baseline+"_"+args_cli.id,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Keep the official checkpoint directory untouched; the fixed policy is
    # only used for evaluation, so no ONNX/JIT export is needed here.

    dt = env.unwrapped.step_dt
    print("dt:", dt)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # print("Here???")
    diagnostic = {
        "steps": 0, "joint_abs_error_sum": 0.0, "root_error_sum": 0.0,
        "observations": 0, "early_resets": 0, "root_below_0_3_count": 0,
        "motion_id_counts": {}, "contact_sensor_available": False,
        "contact_body_names": [], "left_ankle_force_gt_10n": 0,
        "right_ankle_force_gt_10n": 0, "contact_observations": 0,
    }
    robot_joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
    reference_joint_names = list(env.unwrapped.motion_lib.joint_names)
    reference_joint_indices = [robot_joint_names.index(name) for name in reference_joint_names]
    diagnostic["tracked_joint_names"] = reference_joint_names
    diagnostic["initial_motion_ids"] = env.unwrapped.motion_ids.detach().cpu().tolist()
    diagnostic["motion_library_size"] = len(env.unwrapped.motion_lib._motion_lengths)
    try:
        contact_sensor = env.unwrapped.scene["contact_forces"]
        contact_names = list(contact_sensor.body_names)
        diagnostic["contact_sensor_available"] = True
        diagnostic["contact_body_names"] = contact_names
        left_contact_index = contact_names.index("left_ankle_roll_link")
        right_contact_index = contact_names.index("right_ankle_roll_link")
    except (KeyError, ValueError, AttributeError):
        contact_sensor = None
    # print("Whats happening???")
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            unwrapped = env.unwrapped
            active_ids = unwrapped.motion_ids.clone()
            motion_times = unwrapped.episode_length_buf * unwrapped.step_dt + unwrapped.start_motion_times
            reference = unwrapped.motion_lib.get_motion_state(active_ids, motion_times)
            observed_joints = unwrapped.scene["robot"].data.joint_pos
            observed_root = unwrapped.scene["robot"].data.root_pos_w - unwrapped.scene.env_origins
            diagnostic["joint_abs_error_sum"] += float(
                torch.abs(observed_joints[:, reference_joint_indices] - reference["dof_pos"]).sum().item()
            )
            diagnostic["root_error_sum"] += float(
                torch.linalg.norm(observed_root - reference["root_pos"], dim=1).sum().item()
            )
            diagnostic["root_below_0_3_count"] += int((observed_root[:, 2] < 0.3).sum().item())
            diagnostic["observations"] += int(observed_root.shape[0])
            for key, count in zip(*torch.unique(active_ids, return_counts=True)):
                label = str(int(key.item()))
                diagnostic["motion_id_counts"][label] = diagnostic["motion_id_counts"].get(label, 0) + int(count.item())
            actions = policy(obs).clone()
            obs, _, dones, _ = env.step(actions)
            diagnostic["steps"] += 1
            if timestep < args_cli.video_length - 1:
                diagnostic["early_resets"] += int(dones.sum().item())
            if contact_sensor is not None:
                forces = torch.linalg.norm(contact_sensor.data.net_forces_w, dim=-1)
                diagnostic["left_ankle_force_gt_10n"] += int((forces[:, left_contact_index] > 10).sum().item())
                diagnostic["right_ankle_force_gt_10n"] += int((forces[:, right_contact_index] > 10).sum().item())
                diagnostic["contact_observations"] += int(forces.shape[0])

        if args_cli.video:
            timestep += 1
            print(timestep)

            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # This diagnostic does not overwrite the official motion dump.
    success_rate = 100*(torch.sum(env.unwrapped.n_successes > 0)/int(args_cli.num_envs)).detach().cpu().item()
    print("Success rate: ", success_rate, "%")
    diagnostic.update({
        "status": "simulation_only_fixed_tracker_external_goal_diagnostic",
        "reference_dir": os.environ.get("DREAMCONTROL_KICK_REF_PATH"),
        "external_goal_definition": "one fixed pre-generation hint, mapped by input/URDF leg length and anchored to original robot root at the hint median frame",
        "external_goals_json": str(args_cli.external_goals_json),
        "external_goals_sha256": hashlib.sha256(args_cli.external_goals_json.read_bytes()).hexdigest(),
        "checkpoint": str(resume_path), "seed": args_cli.eval_seed,
        "num_envs": args_cli.num_envs, "video_length": args_cli.video_length,
        "custom_external_goal_proximity_success_percent": success_rate,
        "joint_mae_rad": diagnostic["joint_abs_error_sum"] / (diagnostic["observations"] * len(reference_joint_indices)),
        "root_mean_error_m": diagnostic["root_error_sum"] / diagnostic["observations"],
        "root_below_0_3_fraction": diagnostic["root_below_0_3_count"] / diagnostic["observations"],
        "early_resets_per_environment": diagnostic["early_resets"] / args_cli.num_envs,
        "left_ankle_contact_frame_fraction": diagnostic["left_ankle_force_gt_10n"] / diagnostic["contact_observations"] if diagnostic["contact_observations"] else None,
        "right_ankle_contact_frame_fraction": diagnostic["right_ankle_force_gt_10n"] / diagnostic["contact_observations"] if diagnostic["contact_observations"] else None,
        "target_impact_contact_measured": False,
        "notes": ["Custom Kick evaluation replaces the reference-derived goal with a fixed mapped pre-generation hint. It is not the paper's official task score.",
                  "The task counts ankle-to-goal proximity, not a force-confirmed impact.",
                  "Ankle contact force is not separated into floor versus target object contacts.",
                  "This evaluates a fixed pretrained tracker; failure does not by itself prove an impossible reference."],
    })
    out = Path(args_cli.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {out}")
    out.write_text(json.dumps(diagnostic, indent=2) + "\n")

    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
