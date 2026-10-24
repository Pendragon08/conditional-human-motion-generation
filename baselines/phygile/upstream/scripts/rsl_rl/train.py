# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Path to a local motion .npz file. If provided, overrides --registry_name.",
)
parser.add_argument(
    "--motion_dir",
    type=str,
    nargs="+",
    default=None,
    help="One or more directories of motion .npz files. Overrides --motion_file and --registry_name.",
)
parser.add_argument(
    "--urdf", type=str, default=None,
    help="URDF filename or path (e.g. main_no_hand_collision.urdf). Default: main.urdf",
)
parser.add_argument(
    "--g1_training_cfg",
    type=str,
    default=None,
    help="Path to a YAML file overriding G1 training parameters (sets G1_TRAINING_CFG_PATH).",
)
parser.add_argument(
    "--no_wandb_model_upload",
    action="store_true",
    default=False,
    help="Do not upload model/checkpoint artifacts to wandb (metrics still logged).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# allow overriding G1 training config via CLI
if args_cli.g1_training_cfg:
    g1_cfg_path = os.path.abspath(args_cli.g1_training_cfg)
    os.environ["G1_TRAINING_CFG_PATH"] = g1_cfg_path
    print(f"[INFO] Using G1 training config: {g1_cfg_path}")

# disable wandb uploads if requested
upload_models_to_wandb = not getattr(args_cli, "no_wandb_model_upload", False)
if not upload_models_to_wandb:
    print("[INFO] Wandb model uploads disabled via --no_wandb_model_upload")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import shutil

import gymnasium as gym
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve motion source priority: motion_dir > motion_file > registry artifact > config default
    registry_name = args_cli.registry_name
    motion_file = args_cli.motion_file
    motion_dir = args_cli.motion_dir
    resolved_registry_name = None

    if motion_dir:
        if len(motion_dir) == 1:
            env_cfg.commands.motion.motion_dir = os.path.abspath(motion_dir[0])
        else:
            env_cfg.commands.motion.motion_dir = [os.path.abspath(d) for d in motion_dir]
        print(f"[INFO] Using local motion directory: {env_cfg.commands.motion.motion_dir}")
    elif motion_file:
        motion_file = os.path.abspath(motion_file)
        env_cfg.commands.motion.motion_file = motion_file
        print(f"[INFO] Using local motion file: {motion_file}")
    elif registry_name:
        resolved_registry_name = registry_name if ":" in registry_name else f"{registry_name}:latest"
        import pathlib

        import wandb

        api = wandb.Api()  # type: ignore[attr-defined]
        artifact = api.artifact(resolved_registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO] Downloaded motion from wandb registry to: {env_cfg.commands.motion.motion_file}")
    else:
        print("[INFO] Using motion source from task config (no CLI overrides provided).")

    # override URDF if specified
    if args_cli.urdf is not None:
        urdf_arg = args_cli.urdf
        if not os.path.isabs(urdf_arg) and not os.path.isfile(urdf_arg):
            from whole_body_tracking.assets import ASSET_DIR
            urdf_arg = os.path.join(ASSET_DIR, "unitree_description", "urdf", "g1", urdf_arg)
        urdf_arg = os.path.abspath(urdf_arg)
        if not os.path.isfile(urdf_arg):
            raise FileNotFoundError(f"URDF file does not exist: {urdf_arg}")
        env_cfg.scene.robot.spawn.asset_path = urdf_arg
        print(f"[INFO] Overriding URDF: {urdf_arg}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Reproduction guard: reject silent DOF/body-order or frame-rate mismatches.
    import numpy as np
    import json
    import hashlib
    from pathlib import Path
    motion_cfg = env_cfg.commands.motion
    sources = []
    if motion_cfg.motion_dir:
        folders = motion_cfg.motion_dir if isinstance(motion_cfg.motion_dir, (list, tuple)) else [motion_cfg.motion_dir]
        sources = [p for folder in folders for p in sorted(Path(folder).glob("*.npz"))]
    else:
        sources = [Path(motion_cfg.motion_file)]
    records = []
    robot = env.unwrapped.scene["robot"]
    for source in sources:
        with np.load(source, allow_pickle=False) as reference:
            assert reference["joint_names"].tolist() == robot.joint_names, f"Joint order mismatch: {source}"
            assert reference["body_names"].tolist() == robot.body_names, f"Body order mismatch: {source}"
            assert abs(float(reference["fps"].reshape(-1)[0]) * env.unwrapped.step_dt - 1.0) < 1e-6
            records.append({"file": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "frames": int(reference["joint_pos"].shape[0])})
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    (Path(log_dir) / "reference_validation.json").write_text(json.dumps(records, indent=2))
    print(f"[Reproduction] Verified joint/body order and 50 Hz timing for {len(records)} references.")
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
        registry_name=resolved_registry_name,
        upload_models_to_wandb=upload_models_to_wandb,
        task_name=args_cli.task,
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    resume_path = None
    if agent_cfg.resume:
        # get path to previous checkpoint
        # Support cross-experiment loading: if load_run is not found under the
        # current experiment, search all sibling experiment directories (e.g.
        # Phase B loading a Phase A checkpoint).
        try:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        except ValueError:
            rsl_rl_root = os.path.dirname(log_root_path)
            resume_path = None
            for sibling in sorted(os.listdir(rsl_rl_root)):
                sibling_path = os.path.join(rsl_rl_root, sibling)
                if not os.path.isdir(sibling_path) or sibling_path == log_root_path:
                    continue
                try:
                    resume_path = get_checkpoint_path(sibling_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
                    print(f"[INFO] Found checkpoint in sibling experiment: {sibling}")
                    break
                except ValueError:
                    continue
            if resume_path is None:
                raise ValueError(
                    f"Cannot find load_run '{agent_cfg.load_run}' in '{log_root_path}' "
                    f"or any sibling experiment under '{rsl_rl_root}'"
                )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        # 恢复 curriculum level（如果 checkpoint 中包含）
        try:
            # 直接加载 checkpoint 文件来获取 curriculum level
            checkpoint = torch.load(resume_path, map_location=runner.device)
            motion_term = env.unwrapped.command_manager.get_term("motion")
            if motion_term is not None and hasattr(motion_term, "_curriculum") and motion_term._curriculum is not None:
                saved_level = checkpoint.get("curriculum_level")  # 兼容旧 checkpoint 可能使用 curriculum_max_level
                if saved_level is None:
                    saved_level = checkpoint.get("curriculum_max_level")  # 向后兼容
                # 可选：resume 时强制覆盖 level（用于“本来只到level3，但希望直接从level4开始训”）
                override_level = int(getattr(getattr(motion_term, "cfg", None), "curriculum_resume_level_override", 0) or 0)
                target_level = int(saved_level) if (saved_level is not None and saved_level > 0) else 1
                if override_level > 0:
                    target_level = override_level

                if target_level > 0:
                    # 恢复/覆盖 level
                    motion_term._curriculum.max_level = int(target_level)
                    motion_term._curriculum_current_level = int(target_level)
                    # 关键：把该level的“解锁迭代”设为当前迭代，避免 resume 后渐进引入/自动升级立刻失真
                    try:
                        # Prefer the iteration stored in checkpoint; runner.current_learning_iteration
                        # may not be updated consistently across rsl-rl versions at this point.
                        cur_it = int(checkpoint.get("iter", getattr(runner, "current_learning_iteration", 0)) or 0)
                        if hasattr(motion_term._curriculum, "_level_unlocked_iter"):
                            motion_term._curriculum._level_unlocked_iter[int(target_level)] = cur_it
                        if hasattr(motion_term._curriculum, "_current_iter"):
                            motion_term._curriculum._current_iter = cur_it
                        # Also sync MotionCommand's internal iteration tracker used for evaluation scheduling.
                        if hasattr(motion_term, "set_training_iteration"):
                            motion_term.set_training_iteration(cur_it)
                    except Exception:
                        pass
                    # 重建文件列表以匹配恢复的 level
                    if hasattr(motion_term, "_rebuild_file_list_for_curriculum"):
                        motion_term._rebuild_file_list_for_curriculum()
                    # 同步更新 MoE policy 的 max_active_experts（关键！否则会导致 aux loss 为 inf）
                    try:
                        policy = runner.alg.policy
                        if hasattr(policy, "set_max_active_experts"):
                            policy.set_max_active_experts(int(target_level))
                        if hasattr(policy, "set_current_max_level"):
                            policy.set_current_max_level(int(target_level))
                        # 同步更新 runner 的 _last_curriculum_level 避免重复触发 _reset_params_on_level_change
                        if hasattr(runner, "_last_curriculum_level"):
                            runner._last_curriculum_level = int(target_level)
                        # 清除 file-level 缓存
                        if hasattr(motion_term, "invalidate_file_level_cache"):
                            motion_term.invalidate_file_level_cache()
                    except Exception:
                        pass
                    if override_level > 0:
                        print(f"[Curriculum] ✓ Overrode resume level to {target_level} (checkpoint had {saved_level})")
                    else:
                        print(f"[Curriculum] ✓ Restored level {target_level} from checkpoint")
                else:
                    print(f"[Curriculum] No curriculum level found in checkpoint, starting from level 1")
            # 跳过 resume 后的首次 curriculum 评估
            if motion_term is not None and hasattr(motion_term, "skip_next_curriculum_evaluation"):
                motion_term.skip_next_curriculum_evaluation()
        except Exception as e:
            print(f"[Curriculum] Warning: Failed to restore curriculum level: {e}")
            import traceback
            traceback.print_exc()

        # Phase B: unlock experts trained in Phase A, switch to adaptive routing
        if getattr(runner, "_phase_b_pending_unlock", False):
            try:
                saved_level = checkpoint.get("curriculum_level") or checkpoint.get("curriculum_max_level")
                active = int(saved_level) if saved_level and int(saved_level) > 0 else runner.alg.policy._num_experts
                policy = runner.alg.policy
                policy.set_max_active_experts(active)
                policy.set_current_max_level(active)
                policy.set_routing_mode("adaptive")
                print(
                    f"[Phase B] Unlocked {active} experts from Phase A checkpoint "
                    f"(total={policy._num_experts}), routing mode -> adaptive"
                )

                # Reset critic last layer for new data distribution
                critic_last = policy.critic[-1]
                if isinstance(critic_last, torch.nn.Linear):
                    torch.nn.init.orthogonal_(critic_last.weight, gain=0.01)
                    if critic_last.bias is not None:
                        torch.nn.init.zeros_(critic_last.bias)
                    print("[Phase B] Reset critic last layer (orthogonal gain=0.01)")

            except Exception as e:
                print(f"[Phase B] Warning: Failed to unlock experts: {e}")
            runner._phase_b_pending_unlock = False

            # Phase B: configure stability measures (critic warmup, value clamp)
            if hasattr(runner.alg, "configure_phase_b_stability"):
                moe_cfg_b = {}
                _g1_cfg_path = os.environ.get("G1_TRAINING_CFG_PATH")
                if _g1_cfg_path and os.path.isfile(_g1_cfg_path):
                    import yaml as _yaml
                    with open(_g1_cfg_path, "r", encoding="utf-8") as _f:
                        _g1_full = _yaml.safe_load(_f) or {}
                    moe_cfg_b = _g1_full.get("policy", {}).get("moe_config") or {}
                runner.alg.configure_phase_b_stability(
                    max_value_err=float(moe_cfg_b.get("max_value_err", 10.0)),
                    critic_warmup_iters=int(moe_cfg_b.get("critic_warmup_iters", 3000)),
                    surrogate_ramp_iters=int(moe_cfg_b.get("surrogate_ramp_iters", 5000)),
                    start_iter=int(checkpoint.get("iter", 0) or 0),
                )

            # Phase B: set up pseudo-levels for old data routing supervision
            try:
                motion_term = env.unwrapped.command_manager.get_term("motion")
                if motion_term is not None:
                    # Try to get CSV path from config or Phase A config
                    csv_path = getattr(getattr(motion_term, "cfg", None), "phase_b_pseudo_level_csv", None)
                    if csv_path is None:
                        csv_path = getattr(getattr(motion_term, "cfg", None), "curriculum_csv_path", None)
                    if csv_path and os.path.isfile(csv_path):
                        motion_term.compute_phase_b_pseudo_levels(csv_path)

                    # Configure data mixing if multi-dir
                    cfg = getattr(motion_term, "cfg", None)
                    old_ratio = float(getattr(cfg, "phase_b_old_data_ratio", 0) or 0)
                    if old_ratio > 0 and hasattr(motion_term, "_num_source_dirs") and motion_term._num_source_dirs > 1:
                        motion_term.configure_phase_b_data_mixing(
                            old_data_ratio=old_ratio,
                            old_data_ratio_min=float(getattr(cfg, "phase_b_old_data_ratio_min", 0.3)),
                            anneal_iters=int(getattr(cfg, "phase_b_ratio_anneal_iters", 30000)),
                            start_iter=int(checkpoint.get("iter", 0) or 0),
                        )
            except Exception as e:
                print(f"[Phase B] Warning: pseudo-level/data-mixing setup failed: {e}")
                import traceback
                traceback.print_exc()

    # auto-import drop.txt from the resumed run's directory
    if agent_cfg.resume and resume_path is not None:
        prev_run_dir = os.path.dirname(resume_path)
        prev_drop = os.path.join(prev_run_dir, "drop.txt")
        if os.path.isfile(prev_drop):
            try:
                os.makedirs(log_dir, exist_ok=True)
                dropped_names: set[str] = set()
                with open(prev_drop, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3 and parts[1] == "drop":
                            dropped_names.add(parts[2])
                if dropped_names:
                    base_dir = ""
                    if motion_dir:
                        base_dir = os.path.abspath(motion_dir[0])
                    freeze_in_path = os.path.join(log_dir, "freeze_in.txt")
                    with open(freeze_in_path, "w", encoding="utf-8") as f:
                        for name in sorted(dropped_names):
                            full = os.path.join(base_dir, name) if base_dir else name
                            f.write(full + "\n")
                    shutil.copy2(prev_drop, os.path.join(log_dir, "drop.txt"))
                    print(f"[Drop Import] Imported {len(dropped_names)} dropped files from {prev_drop}")
                    motion_term = env.unwrapped.command_manager.get_term("motion")
                    if motion_term is not None and hasattr(motion_term, "set_log_dir"):
                        motion_term.set_log_dir(log_dir)
            except Exception as e:
                print(f"[Drop Import] Warning: Failed to import drop.txt: {e}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore[call-arg]
    # close sim app
    simulation_app.close()
