"""
Standalone curriculum evaluation script.

This script evaluates motion tracking performance for curriculum learning.
It can be run:
1. As a subprocess during training (via MotionOnPolicyRunner)
2. Independently after training for detailed evaluation

Usage (subprocess mode - called by MotionOnPolicyRunner):
    python curriculum_eval.py --task Tracking-Flat-G1-v0 \
        --motion_folder /path/to/motions \
        --curriculum_level 1 \
        --eval_files_json /path/to/files.json \
        --checkpoint_path /path/to/model.pt \
        --output_file eval_results.json

Usage (standalone mode):
    python curriculum_eval.py --task Tracking-Flat-G1-v0 \
        --motion_folder /path/to/motions \
        --load_run run_name \
        --curriculum_level 1 \
        --output_file eval_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Force unbuffered output for subprocess logging
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
from pathlib import Path

# Parse arguments before importing Isaac modules
parser = argparse.ArgumentParser(
    description="Curriculum evaluation for motion tracking."
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--curriculum_level", type=int, required=True, help="Curriculum level to evaluate.")
parser.add_argument("--output_file", type=str, required=True, help="Output file to save evaluation results.")
parser.add_argument("--motion_folder", type=str, help="Path to motion files directory.")
parser.add_argument("--num_envs", type=int, default=1000, help="Number of environments.")
parser.add_argument("--eval_runs", type=int, default=5, help="Number of evaluation runs per file.")
parser.add_argument("--disable_fabric", action="store_true", default=False)

# New arguments for subprocess mode
parser.add_argument("--eval_files_json", type=str, default=None,
                    help="JSON file containing list of files to evaluate (subprocess mode).")
parser.add_argument("--checkpoint_path", type=str, default=None,
                    help="Direct path to model checkpoint (subprocess mode).")

# Import cli_args for RSL-RL arguments
import cli_args  # isort: skip

cli_args.add_rsl_rl_args(parser)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Force headless mode
if hasattr(args_cli, "headless"):
    args_cli.headless = True
if hasattr(args_cli, "video"):
    args_cli.video = False

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import heavy modules
import numpy as np
import torch
import gymnasium as gym

from collections import deque
from dataclasses import dataclass

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import register_task_to_hydra

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner


@dataclass
class ClipMeta:
    name: str
    path: str
    index: int
    frames: int


class EvalJobScheduler:
    """Manages evaluation jobs across parallel environments."""

    def __init__(self, motion_term, eval_files: list[str], eval_runs: int):
        self.motion_term = motion_term
        self.device = motion_term.device
        self.eval_runs = max(eval_runs, 1)

        self.clip_meta: list[ClipMeta] = []
        for file_path in eval_files:
            try:
                idx = motion_term.motion_file_paths.index(file_path)
                frames = int(motion_term.motion_file_lengths[idx].item())
                self.clip_meta.append(ClipMeta(
                    name=os.path.basename(file_path),
                    path=file_path,
                    index=idx,
                    frames=frames,
                ))
            except ValueError:
                print(f"[WARN] File {file_path} not found in motion term, skipping.")
                continue

        self.job_queue: deque[tuple[int, int]] = deque()
        for clip_idx, _ in enumerate(self.clip_meta):
            for run_idx in range(self.eval_runs):
                self.job_queue.append((clip_idx, run_idx))

        self.total_jobs = len(self.job_queue)
        self.completed_jobs = 0
        self.active_jobs: dict[int, dict] = {}

        self.clip_mpjpe_sums = [0.0] * len(self.clip_meta)
        self.clip_mpjae_sums = [0.0] * len(self.clip_meta)
        self.clip_step_counts = [0] * len(self.clip_meta)

    def assign_jobs(self, env_ids: torch.Tensor):
        assigned_envs, assigned_files, assigned_lengths = [], [], []
        for env_id in env_ids.tolist():
            if self.job_queue:
                clip_idx, run_idx = self.job_queue.popleft()
                job = {
                    "clip_idx": clip_idx,
                    "run_idx": run_idx,
                    "file_idx": self.clip_meta[clip_idx].index,
                    "frames": self.clip_meta[clip_idx].frames,
                    "padding": False,
                }
            else:
                clip_idx = 0 if self.clip_meta else -1
                job = {
                    "clip_idx": clip_idx,
                    "run_idx": -1,
                    "file_idx": self.clip_meta[clip_idx].index if clip_idx >= 0 else 0,
                    "frames": self.clip_meta[clip_idx].frames if clip_idx >= 0 else 1,
                    "padding": True,
                }
            self.active_jobs[env_id] = job
            assigned_envs.append(env_id)
            assigned_files.append(job["file_idx"])
            assigned_lengths.append(job["frames"])

        return (
            torch.tensor(assigned_envs, dtype=torch.long, device=self.device),
            torch.tensor(assigned_files, dtype=torch.long, device=self.device),
            torch.tensor(assigned_lengths, dtype=torch.long, device=self.device),
        )

    def record_metrics(self, env_ids: torch.Tensor, mpjpe: torch.Tensor, mpjae: torch.Tensor):
        for i, env_id in enumerate(env_ids.tolist()):
            job = self.active_jobs.get(env_id)
            if job is None or job["padding"]:
                continue
            clip_idx = job["clip_idx"]
            mpjpe_val = float(mpjpe[i].item())
            mpjae_val = float(mpjae[i].item())
            # 过滤 inf/nan 值，避免污染统计
            # 只有有效值才计入 sum 和 count
            if math.isfinite(mpjpe_val) and math.isfinite(mpjae_val):
                self.clip_mpjpe_sums[clip_idx] += mpjpe_val
                self.clip_mpjae_sums[clip_idx] += mpjae_val
                self.clip_step_counts[clip_idx] += 1

    def release_jobs(self, env_ids: torch.Tensor):
        for env_id in env_ids.tolist():
            job = self.active_jobs.pop(env_id, None)
            if job and not job["padding"]:
                self.completed_jobs += 1

    def get_active_eval_envs(self) -> tuple[torch.Tensor, list[int]]:
        envs, clips = [], []
        for env_id, job in self.active_jobs.items():
            if not job["padding"]:
                envs.append(env_id)
                clips.append(job["clip_idx"])
        return torch.tensor(envs, dtype=torch.long, device=self.device), clips

    def compute_final_metrics(self) -> tuple[float, float, list[str]]:
        """计算最终指标，排除无有效数据的问题文件。
        
        只有当文件**完全没有有效步数**（所有步都是 inf/nan）时才标记为 bad file。
        如果有一些有效步数，就正常计算平均值。
        
        Returns:
            (mpjpe, mpjae, bad_files): 平均指标和无有效数据的问题文件列表
        """
        total_mpjpe, total_mpjae, count = 0.0, 0.0, 0
        bad_files: list[str] = []  # 完全无有效数据的问题文件（永久 freeze）
        
        for i in range(len(self.clip_meta)):
            if self.clip_step_counts[i] > 0:
                # 有有效步数，计算平均值并计入总体统计
                clip_mpjpe = self.clip_mpjpe_sums[i] / self.clip_step_counts[i]
                clip_mpjae = self.clip_mpjae_sums[i] / self.clip_step_counts[i]
                total_mpjpe += clip_mpjpe
                total_mpjae += clip_mpjae
                count += 1
            else:
                # 完全没有有效步数（所有值都是 inf/nan），标记为问题文件并永久 freeze
                bad_files.append(self.clip_meta[i].path)
                print(f"[CurriculumEval] Bad file (no valid steps): {self.clip_meta[i].name}")
        
        if count == 0:
            return float("inf"), float("inf"), bad_files
        return total_mpjpe / count, total_mpjae / count, bad_files

    @property
    def finished(self) -> bool:
        return self.completed_jobs >= self.total_jobs and not self.job_queue


def _install_eval_scheduler(motion_term, scheduler: EvalJobScheduler) -> None:
    """Monkey patch motion_term._multi_file_sampling to use EvalJobScheduler.
    
    This ensures environment resets use the scheduler for file assignment,
    keeping indices consistent and tracking metrics correctly.
    """
    import types as _types
    
    def _scheduled_multi_file_sampling(self, env_ids, initialize: bool = False, episode_failed=None):
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if env_ids_tensor.numel() == 0:
            return
        
        # Release completed jobs (unless initializing)
        if not initialize:
            scheduler.release_jobs(env_ids_tensor)
        
        # Assign new jobs
        assigned_envs, assigned_files, assigned_lengths = scheduler.assign_jobs(env_ids_tensor)
        if assigned_envs.numel() == 0:
            return
        
        # Update motion term state
        self.file_indices[assigned_envs] = assigned_files
        self.clip_start_steps[assigned_envs] = 0
        self.clip_end_steps[assigned_envs] = assigned_lengths
        self.time_steps[assigned_envs] = 0
        
        if hasattr(self, "clip_step_count"):
            self.clip_step_count[assigned_envs] = 0.0
        if hasattr(self, "clip_error_accum"):
            self.clip_error_accum[assigned_envs] = 0.0
        if hasattr(self, "static_mask"):
            self.static_mask[assigned_envs] = False
        if hasattr(self, "static_ref_steps"):
            self.static_ref_steps[assigned_envs] = 0
        
        self._apply_command_to_robot(assigned_envs)
    
    motion_term._multi_file_sampling = _types.MethodType(_scheduled_multi_file_sampling, motion_term)
    motion_term._eval_scheduler = scheduler


def main():
    """Run curriculum evaluation."""
    print("[CurriculumEval] Starting evaluation...")
    
    # Check if running in subprocess mode
    subprocess_mode = args_cli.checkpoint_path is not None and args_cli.eval_files_json is not None
    if subprocess_mode:
        print("[CurriculumEval] Running in subprocess mode (called from training).")
    else:
        print("[CurriculumEval] Running in standalone mode.")

    # Setup task
    task_name = args_cli.task
    if not task_name.endswith("-v0"):
        task_name = task_name + "-v0"

    env_cfg, agent_cfg = register_task_to_hydra(task_name, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # Apply CLI overrides
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.motion_folder:
        env_cfg.commands.motion.motion_dir = os.path.abspath(args_cli.motion_folder)
        env_cfg.commands.motion.motion_file = None
        env_cfg.commands.motion.multi_file_mode = True

    # Disable visualizations and optimize memory for subprocess mode
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.debug_vis = False
        if subprocess_mode:
            # Use CPU for preloading to save GPU memory (data transferred on-demand)
            env_cfg.commands.motion.preload_device = "cpu"
            # Disable global_cat to reduce memory footprint
            env_cfg.commands.motion.use_global_cat = False
            env_cfg.commands.motion.preload_all_motion = False
            # 关键：禁用 curriculum 以加载所有文件
            # 原因：curriculum 模式下 motion term 只加载当前 level 的文件
            # 但评估可能需要访问任意 level 的文件（由 eval_files_json 指定）
            # 禁用后会加载 motion_dir 下所有文件，由 scheduler 控制只评估指定的文件
            env_cfg.commands.motion.curriculum_enabled = False
            # 注意：preload_all_motion = False，所以文件是按需加载的，不会占用太多内存
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "contact_forces"):
        env_cfg.scene.contact_forces.debug_vis = False

    # Set long episode length to allow full clip playback
    # Clips will be managed by scheduler, but we need enough time per episode
    env_cfg.episode_length_s = 60.0  # 60 seconds should cover most motion clips

    # Get checkpoint path
    if subprocess_mode:
        resume_path = args_cli.checkpoint_path
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[CurriculumEval] Loading model from: {resume_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = getattr(env, "unwrapped", env)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    # Load model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    if subprocess_mode:
        # In subprocess mode, load checkpoint directly (may not have all keys)
        checkpoint = torch.load(resume_path, map_location=agent_cfg.device)
        ppo_runner.alg.policy.load_state_dict(checkpoint["model_state_dict"])
        if ppo_runner.empirical_normalization and "obs_norm_state_dict" in checkpoint:
            ppo_runner.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
        if ppo_runner.empirical_normalization and "privileged_obs_norm_state_dict" in checkpoint:
            ppo_runner.privileged_obs_normalizer.load_state_dict(checkpoint["privileged_obs_norm_state_dict"])
        print(f"[CurriculumEval] Loaded checkpoint directly (subprocess mode).")
    else:
        # Standalone mode: use standard load
        ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Get motion term
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None:
        raise RuntimeError("Command manager not available.")
    motion_term = command_manager.get_term("motion")
    if motion_term is None:
        raise RuntimeError("Motion term not found.")

    # Get files for evaluation
    level = args_cli.curriculum_level
    
    if subprocess_mode and args_cli.eval_files_json:
        # Load files from JSON (subprocess mode)
        try:
            with open(args_cli.eval_files_json, "r") as f:
                files_data = json.load(f)
            eval_files = files_data.get("files", [])
            print(f"[CurriculumEval] Loaded {len(eval_files)} files from JSON.")
        except Exception as e:
            print(f"[CurriculumEval] Failed to load eval files JSON: {e}")
            eval_files = []
    else:
        # Get files from curriculum manager (standalone mode)
        if not hasattr(motion_term, "_curriculum") or motion_term._curriculum is None:
            raise RuntimeError("Curriculum learning is not enabled.")
        eval_files = motion_term.get_curriculum_files_for_evaluation(set(), level)

    if not eval_files:
        print(f"[CurriculumEval] No files for level {level}.")
        result = {"level": level, "mpjpe": float("inf"), "mpjae": float("inf"), "num_files": 0}
        Path(args_cli.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args_cli.output_file, "w") as f:
            json.dump(result, f, indent=2)
        env.close()
        return

    print(f"[CurriculumEval] Evaluating {len(eval_files)} files at level {level}...", flush=True)

    # Create scheduler and install it via monkey patch
    scheduler = EvalJobScheduler(motion_term, eval_files, args_cli.eval_runs)
    _install_eval_scheduler(motion_term, scheduler)
    print(f"[CurriculumEval] Total jobs: {scheduler.total_jobs}", flush=True)

    # Initial assignment via the patched _multi_file_sampling
    all_env_ids = torch.arange(motion_term.num_envs, device=motion_term.device, dtype=torch.long)
    motion_term._multi_file_sampling(all_env_ids, initialize=True)

    obs, _ = env.get_observations()
    last_progress = -1

    while not scheduler.finished and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        # env.step may trigger resets which call _multi_file_sampling (now patched)
        obs, _, _, _ = env.step(actions)

        # Record metrics for active (non-padding) jobs
        eval_envs, _ = scheduler.get_active_eval_envs()
        if eval_envs.numel() > 0:
            # Compute metrics - these tensors have shape (num_envs,)
            body_err = torch.norm(motion_term.body_pos_w - motion_term.robot_body_pos_w, dim=-1).mean(dim=1)
            joint_err = torch.abs(motion_term.joint_pos - motion_term.robot_joint_pos).mean(dim=1)
            # Index only valid envs
            valid_mask = eval_envs < body_err.shape[0]
            valid_envs = eval_envs[valid_mask]
            if valid_envs.numel() > 0:
                scheduler.record_metrics(valid_envs, body_err[valid_envs], joint_err[valid_envs])

        # Progress reporting (avoid spam)
        progress_step = max(1, scheduler.total_jobs // 20)
        if scheduler.completed_jobs // progress_step > last_progress:
            last_progress = scheduler.completed_jobs // progress_step
            print(f"[CurriculumEval] Progress: {scheduler.completed_jobs}/{scheduler.total_jobs}", flush=True)

    mpjpe, mpjae, bad_files = scheduler.compute_final_metrics()
    print(f"[CurriculumEval] Results: mpjpe={mpjpe:.4f}, mpjae={mpjae:.4f}")
    if bad_files:
        print(f"[CurriculumEval] Found {len(bad_files)} problematic files (inf/nan metrics)")

    result = {
        "level": level,
        "mpjpe": mpjpe,
        "mpjae": mpjae,
        "num_files": len(eval_files),
        "eval_runs": args_cli.eval_runs,
        "bad_files": bad_files,  # 问题文件列表（用于永久freeze）
    }

    Path(args_cli.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args_cli.output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[CurriculumEval] Saved results to {args_cli.output_file}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
