"""Evaluate trained RSL-RL policy metrics over every motion inside a folder."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import types
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from typing import Optional

import numpy as np
import yaml
from yaml.constructor import ConstructorError
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments (match play.py inputs)
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL over a motion folder.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--motion_folder",
    type=str,
    required=True,
    help="Path to a directory of motion files to evaluate jointly.",
)
parser.add_argument(
    "--respect_training_terminations",
    action="store_true",
    help="Keep training termination logic instead of forcing full clip playback.",
)
parser.add_argument(
    "--eval_runs",
    type=int,
    default=1,
    help="Number of evaluation episodes per motion clip (default: 1).",
)
parser.add_argument(
    "--mpjpe_print_threshold",
    type=float,
    default=None,
    help="Only list clips with E_mpjpe above this threshold in the text report.",
)
parser.add_argument(
    "--phase_b",
    action="store_true",
    help="MoE Phase B mode: activate all experts regardless of curriculum_level.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.math import quat_error_magnitude, quat_rotate_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401,E402
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner  # noqa: E402

METRIC_NAMES = ["E_mpjpe", "E_mpjae", "E_mpjve", "E_root_p", "E_root_r", "E_root_v"]


@dataclass
class ClipMeta:
    """Metadata about a single motion file."""

    name: str
    path: str
    frames: int
    duration: Optional[float]


@dataclass
class AssignmentBatch:
    """Represents a batch of env assignments returned by the scheduler."""

    envs: torch.Tensor
    files: torch.Tensor
    lengths: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device) -> "AssignmentBatch":
        zero = torch.zeros(0, dtype=torch.long, device=device)
        return cls(envs=zero, files=zero, lengths=zero)


class MotionJobScheduler:
    """Assign deterministic motion jobs to envs and keep track of progress."""

    def __init__(self, motion_term, clips: list[ClipMeta], eval_runs: int):
        if not getattr(motion_term, "multi_file_mode", False):
            raise RuntimeError("MotionJobScheduler requires the motion term to run in multi-file mode.")

        self.motion_term = motion_term
        self.device = motion_term.device
        self.eval_runs = max(int(eval_runs), 1)

        available_names = [os.path.basename(name) for name in motion_term.motion_file_names]
        clip_lookup = {clip.name: clip for clip in clips}
        missing = [name for name in available_names if name not in clip_lookup]
        if missing:
            raise ValueError(
                f"motion_folder is missing clips referenced by the motion term: {missing[:3]}{'...' if len(missing) > 3 else ''}"
            )

        self.clip_meta_by_index: list[ClipMeta] = [clip_lookup[name] for name in available_names]
        self.clip_lengths: list[int] = [max(1, meta.frames) for meta in self.clip_meta_by_index]
        self.clip_count = len(self.clip_meta_by_index)

        self.job_queue: deque[tuple[int, int]] = deque()
        for run_idx in range(self.eval_runs):
            for clip_idx in range(self.clip_count):
                self.job_queue.append((clip_idx, run_idx))

        self.total_jobs = len(self.job_queue)
        self.completed_jobs = 0
        self.active_jobs: dict[int, dict] = {}
        self._fallback_cursor = 0
        self._last_reported = -1
        self._progress_dirty = False

        # Success rate tracking: per-env "ever failed" flag and counters
        self.ever_failed = torch.zeros(motion_term.num_envs, dtype=torch.bool, device=self.device)
        self.success_count = 0
        self.failure_count = 0
        self.clip_success = [0] * self.clip_count
        self.clip_failure = [0] * self.clip_count

    def assign(self, env_ids_tensor: torch.Tensor) -> AssignmentBatch:
        env_ids = env_ids_tensor.tolist()
        assigned_envs: list[int] = []
        assigned_files: list[int] = []
        assigned_lengths: list[int] = []

        for env_id in env_ids:
            job = self._pop_job()
            if job is None:
                job = self._make_padding_job()
            self.active_jobs[env_id] = job
            assigned_envs.append(env_id)
            assigned_files.append(job["clip_index"])
            assigned_lengths.append(job["clip_length"])

        if not assigned_envs:
            return AssignmentBatch.empty(self.device)

        return AssignmentBatch(
            envs=torch.tensor(assigned_envs, dtype=torch.long, device=self.device),
            files=torch.tensor(assigned_files, dtype=torch.long, device=self.device),
            lengths=torch.tensor(assigned_lengths, dtype=torch.long, device=self.device),
        )

    def release(self, env_ids_tensor: torch.Tensor) -> None:
        env_ids = env_ids_tensor.tolist()
        for env_id in env_ids:
            job = self.active_jobs.pop(env_id, None)
            if job is None:
                continue
            if not job["padding"]:
                self.completed_jobs += 1
                clip_idx = job["clip_index"]
                if self.ever_failed[env_id].item():
                    self.failure_count += 1
                    self.clip_failure[clip_idx] += 1
                else:
                    self.success_count += 1
                    self.clip_success[clip_idx] += 1
                self._report_progress()
        # Reset ever_failed for released envs
        self.ever_failed[env_ids_tensor] = False

    def mark_failed(self, env_ids: torch.Tensor) -> None:
        """Mark envs as having triggered termination conditions during current episode."""
        if env_ids.numel() > 0:
            self.ever_failed[env_ids] = True

    def eval_env_assignments(self) -> tuple[torch.Tensor, torch.Tensor]:
        active_envs: list[int] = []
        active_files: list[int] = []
        for env_id, job in self.active_jobs.items():
            if job["padding"]:
                continue
            active_envs.append(env_id)
            active_files.append(job["clip_index"])
        if not active_envs:
            return (
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
            )
        return (
            torch.tensor(active_envs, dtype=torch.long, device=self.device),
            torch.tensor(active_files, dtype=torch.long, device=self.device),
        )

    @property
    def finished(self) -> bool:
        if self.completed_jobs < self.total_jobs:
            return False
        # ensure no outstanding real jobs
        return all(job["padding"] for job in self.active_jobs.values())

    def clip_name(self, index: int) -> str:
        return self.clip_meta_by_index[index].name
    
    def progress(self) -> tuple[int, int]:
        return self.completed_jobs, self.total_jobs
    
    def finalize_progress(self) -> None:
        if self._progress_dirty:
            print()
            self._progress_dirty = False

    def _pop_job(self) -> Optional[dict]:
        if not self.job_queue:
            return None
        clip_idx, run_idx = self.job_queue.popleft()
        return {
            "clip_index": clip_idx,
            "run_index": run_idx,
            "clip_length": self.clip_lengths[clip_idx],
            "padding": False,
        }

    def _make_padding_job(self) -> dict:
        clip_idx = self._fallback_cursor
        self._fallback_cursor = (self._fallback_cursor + 1) % max(1, self.clip_count)
        return {
            "clip_index": clip_idx,
            "run_index": -1,
            "clip_length": self.clip_lengths[clip_idx],
            "padding": True,
        }

    def _report_progress(self) -> None:
        if self.completed_jobs == self._last_reported:
            return
        self._last_reported = self.completed_jobs
        total = max(1, self.total_jobs)
        percent = (self.completed_jobs / total) * 100.0
        print(
            f"\r[PROGRESS] Completed {self.completed_jobs}/{self.total_jobs} scheduled runs "
            f"({percent:.1f}%).",
            end="",
            flush=True,
        )
        self._progress_dirty = True


class MetricsAccumulator:
    """Track global and per-clip metric statistics."""

    def __init__(self, clip_meta: list[ClipMeta]):
        self.clip_meta = clip_meta
        self.global_sums = {name: 0.0 for name in METRIC_NAMES}
        self.global_steps = 0
        self.clip_stats = [
            {"name": clip.name, "metrics": {name: 0.0 for name in METRIC_NAMES}, "steps": 0}
            for clip in clip_meta
        ]

    def record(self, metrics: dict[str, torch.Tensor], env_ids: torch.Tensor, clip_indices: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        clip_list = clip_indices.tolist()
        for metric_name in METRIC_NAMES:
            values = metrics[metric_name][env_ids]
            self.global_sums[metric_name] += float(values.sum().item())
            for offset, clip_idx in enumerate(clip_list):
                self.clip_stats[clip_idx]["metrics"][metric_name] += float(values[offset].item())
        for clip_idx in clip_list:
            self.clip_stats[clip_idx]["steps"] += 1
        self.global_steps += int(env_ids.numel())

    def global_average(self) -> dict[str, float]:
        if self.global_steps == 0:
            raise RuntimeError("No samples collected; unable to compute averages.")
        return {
            name: self.global_sums[name] / float(self.global_steps)
            for name in METRIC_NAMES
        }

    def clip_averages(self) -> list[dict]:
        reports: list[dict] = []
        for stats in self.clip_stats:
            steps = stats["steps"]
            if steps == 0:
                continue
            per_metric = {
                name: stats["metrics"][name] / float(steps)
                for name in METRIC_NAMES
            }
            reports.append(
                {
                    "name": stats["name"],
                    "steps": steps,
                    "metrics": per_metric,
                }
            )
        return reports

    @property
    def total_steps(self) -> int:
        return self.global_steps


def _load_run_configs(log_dir: str, env_cfg, agent_cfg) -> None:
    """Load env/agent configs from a run directory if available."""
    params_dir = os.path.join(log_dir, "params")
    env_path = os.path.join(params_dir, "env.yaml")
    agent_path = os.path.join(params_dir, "agent.yaml")
    env_pkl_path = os.path.join(params_dir, "env.pkl")
    agent_pkl_path = os.path.join(params_dir, "agent.pkl")

    def _load_yaml(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        if not content.strip():
            return {}

        def _load(loader_name: str, loader_func):
            try:
                return loader_func(content) or {}
            except ConstructorError as exc:
                print(f"[WARN] yaml.{loader_name} failed for {path}: {exc}")
                return None

        data = _load("safe_load", lambda txt: yaml.safe_load(txt))
        if data is not None:
            return data

        data = _load("load (FullLoader)", lambda txt: yaml.load(txt, Loader=yaml.FullLoader))
        if data is not None:
            return data

        data = _load("load (UnsafeLoader)", lambda txt: yaml.load(txt, Loader=yaml.UnsafeLoader))
        if data is not None:
            print(f"[WARN] yaml.safe_load and FullLoader failed for {path}; loaded via UnsafeLoader.")
            return data

        raise RuntimeError(f"Unable to parse YAML file: {path}")

    def _prime_cfg_types(cfg_obj, data: dict) -> None:
        """Ensure cfg_obj fields have compatible runtime types before from_dict assignments."""
        for key, value in data.items():
            if value is None:
                continue
            target = None
            if isinstance(cfg_obj, dict):
                if key not in cfg_obj:
                    continue
                target = cfg_obj.get(key)
            else:
                if not hasattr(cfg_obj, key):
                    continue
                target = getattr(cfg_obj, key)

            if isinstance(value, dict):
                if target is not None:
                    _prime_cfg_types(target, value)
                continue

            if target is None:
                replacement = copy.deepcopy(value)
                if isinstance(cfg_obj, dict):
                    cfg_obj[key] = replacement
                else:
                    setattr(cfg_obj, key, replacement)

    def _coerce_numeric_types(cfg_obj, data: dict) -> None:
        """Convert ints -> floats if target expects float."""
        for key, value in list(data.items()):
            target = None
            if isinstance(cfg_obj, dict):
                target = cfg_obj.get(key)
            else:
                target = getattr(cfg_obj, key, None)

            if isinstance(value, dict) and target is not None:
                _coerce_numeric_types(target, value)
            elif isinstance(value, int) and isinstance(target, float):
                data[key] = float(value)

    def _apply_cfg_from_dict(cfg_obj, data: dict, cfg_label: str, src_path: str):
        """Apply dict to cfg, handling fields that need manual assignment."""
        if not data:
            return

        special_fields = {}
        data_local = dict(data)
        if isinstance(cfg_obj, ManagerBasedRLEnvCfg) and "seed" in data_local:
            special_fields["seed"] = data_local.pop("seed")

        try:
            _coerce_numeric_types(cfg_obj, data_local)
            _prime_cfg_types(cfg_obj, data_local)
            cfg_obj.from_dict(data_local)
        except Exception:
            data_local.update(special_fields)
            raise
        else:
            for key, value in special_fields.items():
                setattr(cfg_obj, key, value)
            print(f"[INFO] Loaded {cfg_label} config from {src_path}")

    def _load_cfg_from_pickle(pkl_path: str, cfg_obj, cfg_label: str) -> bool:
        """Fallback to loading pickled configuration objects."""
        if not os.path.isfile(pkl_path):
            print(f"[WARN] {cfg_label.capitalize()} pickle not found at {pkl_path}.")
            return False
        try:
            with open(pkl_path, "rb") as handle:
                loaded_cfg = pickle.load(handle)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load {cfg_label} config from {pkl_path}: {exc}")
            return False

        try:
            if hasattr(loaded_cfg, "to_dict"):
                cfg_dict = loaded_cfg.to_dict()
            else:
                cfg_dict = {
                    key: copy.deepcopy(value) for key, value in vars(loaded_cfg).items() if not key.startswith("__")
                }
            _apply_cfg_from_dict(cfg_obj, copy.deepcopy(cfg_dict), cfg_label, pkl_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to copy {cfg_label} config from pickle {pkl_path}: {exc}")
            return False

        print(f"[INFO] Loaded {cfg_label} config from {pkl_path}")
        return True

    if os.path.isfile(env_path):
        try:
            _apply_cfg_from_dict(env_cfg, _load_yaml(env_path), "environment", env_path)
        except Exception as exc:
            print(f"[WARN] Failed to load environment config from {env_path}: {exc}")
            if not _load_cfg_from_pickle(env_pkl_path, env_cfg, "environment"):
                print("[WARN] Falling back to default environment config.")
    else:
        print(f"[WARN] env.yaml not found under {params_dir}, using default env config.")

    if os.path.isfile(agent_path):
        try:
            if hasattr(agent_cfg, "from_dict"):
                _apply_cfg_from_dict(agent_cfg, _load_yaml(agent_path), "agent", agent_path)
        except Exception as exc:
            print(f"[WARN] Failed to load agent config from {agent_path}: {exc}")
            if not _load_cfg_from_pickle(agent_pkl_path, agent_cfg, "agent"):
                print("[WARN] Falling back to default agent config.")
    else:
        print(f"[WARN] agent.yaml not found under {params_dir}, using default agent config.")


def _apply_future_command_encoder_from_params(log_dir: str, agent_cfg) -> None:
    """强制让 policy.future_command_encoder 与训练时保存的配置一致。"""
    # 如果当前 policy 缺失，先创建一个空容器，确保后续赋值不会失败
    if not hasattr(agent_cfg, "policy") or agent_cfg.policy is None:
        try:
            agent_cfg.policy = types.SimpleNamespace()
        except Exception:
            return

    params_dir = os.path.join(log_dir, "params")
    agent_path = os.path.join(params_dir, "agent.yaml")
    if not os.path.isfile(agent_path):
        return

    try:
        with open(agent_path, "r", encoding="utf-8") as handle:
            agent_data = yaml.safe_load(handle) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to parse {agent_path}: {exc}")
        return

    policy_cfg = agent_data.get("policy", {}) if isinstance(agent_data, dict) else {}
    future_cfg = policy_cfg.get("future_command_encoder")
    if future_cfg:
        try:
            agent_cfg.policy.future_command_encoder = copy.deepcopy(future_cfg)
            print("[INFO] Applied future_command_encoder from saved agent.yaml (overriding current value).")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to apply future_command_encoder from agent.yaml: {exc}")


def _sync_motion_command_with_policy(env_cfg, agent_cfg) -> None:
    """Keep env command frame counts aligned with policy future_command_encoder."""
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    policy_cfg = getattr(getattr(agent_cfg, "policy", None), "future_command_encoder", None)
    if motion_cfg is None or policy_cfg is None:
        return

    policy_immediate = policy_cfg.get("immediate_steps")
    policy_conv_steps = policy_cfg.get("conv_steps") or policy_cfg.get("future_steps")

    if getattr(motion_cfg, "future_steps", None) is None and policy_immediate is not None:
        motion_cfg.future_steps = int(policy_immediate)
        print(f"[INFO] env.commands.motion.future_steps set to {motion_cfg.future_steps} from policy.")
    if getattr(motion_cfg, "encoder_future_steps", None) is None and policy_conv_steps is not None:
        motion_cfg.encoder_future_steps = int(policy_conv_steps)
        print(f"[INFO] env.commands.motion.encoder_future_steps set to {motion_cfg.encoder_future_steps} from policy.")

    if policy_immediate is not None and getattr(motion_cfg, "future_steps", None) not in (None, policy_immediate):
        print(
            f"[WARN] policy.immediate_steps={policy_immediate} but env.commands.motion.future_steps="
            f"{motion_cfg.future_steps}; observations may misalign."
        )
    if policy_conv_steps is not None and getattr(motion_cfg, "encoder_future_steps", None) not in (None, policy_conv_steps):
        print(
            f"[WARN] policy.conv_steps={policy_conv_steps} but env.commands.motion.encoder_future_steps="
            f"{motion_cfg.encoder_future_steps}; observations may misalign."
        )


def _resize_obs_normalizer(normalizer, target_dim: int) -> None:
    """Pad or truncate obs normalizer stats to match the current observation dim."""
    if normalizer is None or not hasattr(normalizer, "_mean") or not hasattr(normalizer, "_std"):
        return

    try:
        current_dim = int(normalizer._mean.shape[-1])
    except Exception:
        return

    if current_dim == target_dim:
        return

    device = normalizer._mean.device
    if current_dim < target_dim:
        pad = target_dim - current_dim
        normalizer._mean = torch.cat(
            [normalizer._mean, torch.zeros(pad, device=device, dtype=normalizer._mean.dtype)],
            dim=-1,
        )
        normalizer._std = torch.cat(
            [normalizer._std, torch.ones(pad, device=device, dtype=normalizer._std.dtype)],
            dim=-1,
        )
        print(f"[INFO] Obs normalizer padded from {current_dim} -> {target_dim}.")
    else:
        normalizer._mean = normalizer._mean[..., :target_dim]
        normalizer._std = normalizer._std[..., :target_dim]
        print(f"[INFO] Obs normalizer truncated from {current_dim} -> {target_dim}.")


def _disable_training_terminations(term_cfg) -> dict[str, object]:
    """Remove non timeout termination terms from the config."""
    disabled_terms: dict[str, object] = {}
    if term_cfg is None:
        return disabled_terms

    def _iter_fields():
        if is_dataclass(term_cfg):
            for field in fields(term_cfg):
                yield field.name
        else:
            for key in term_cfg.__dict__.keys():
                if not key.startswith("_"):
                    yield key

    for name in _iter_fields():
        term = getattr(term_cfg, name, None)
        if term is None or getattr(term, "time_out", False):
            continue
        disabled_terms[name] = term
        setattr(term_cfg, name, None)
    return disabled_terms


def _configure_play_reset_strategy(env_cfg, respect_training_terminations: bool, clip_duration: Optional[float]) -> None:
    """Adjust env configuration so play episodes only reset at clip end/time."""
    # Always try to set episode length to motion duration
    if clip_duration is not None:
        try:
            env_cfg.episode_length_s = float(clip_duration)
            print(f"[INFO] Episode length set to longest motion duration: {clip_duration:.2f}s.")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to update episode length for eval: {exc}")

    if respect_training_terminations:
        print("[INFO] Keeping training termination conditions active.")
        return

    term_cfg = getattr(env_cfg, "terminations", None)
    disabled_terms = _disable_training_terminations(term_cfg)
    if disabled_terms:
        print(f"[INFO] Disabled training termination terms for evaluation: {list(disabled_terms.keys())}")


def _scan_motion_folder(env_cfg, folder_path: str) -> list[ClipMeta]:
    """Return sorted metadata for all .npz files in the folder."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Motion folder does not exist: {folder_path}")
    candidates = sorted(
        [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith(".npz")
        ]
    )
    if not candidates:
        raise ValueError(f"No .npz files found under motion folder: {folder_path}")

    default_period = None
    sim_cfg = getattr(env_cfg, "sim", None)
    sim_dt = getattr(sim_cfg, "dt", None) if sim_cfg else None
    decimation = getattr(env_cfg, "decimation", None)
    if sim_dt and decimation:
        default_period = float(sim_dt) * float(decimation)

    meta_list: list[ClipMeta] = []
    for path in candidates:
        with np.load(path) as data:
            if "joint_pos" not in data:
                raise ValueError(f"Motion file missing 'joint_pos': {path}")
            frames = int(data["joint_pos"].shape[0])
            fps = float(data["fps"]) if "fps" in data else None
        duration = None
        if frames > 0:
            if fps and fps > 0.0:
                duration = frames / fps
            elif default_period:
                duration = frames * default_period
        meta_list.append(
            ClipMeta(
                name=os.path.basename(path),
                path=path,
                frames=frames,
                duration=duration,
            )
        )
    return meta_list


def _install_folder_scheduler(motion_term, clips: list[ClipMeta], eval_runs: int) -> MotionJobScheduler:
    """Monkey patch the motion term to rely on the deterministic scheduler."""
    scheduler = MotionJobScheduler(motion_term, clips, eval_runs)

    def _scheduled_multi_file_sampling(self, env_ids, initialize: bool = False, episode_failed=None):
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if env_ids_tensor.numel() == 0:
            return
        if not initialize:
            if hasattr(self, "_finalize_clip_errors"):
                self._finalize_clip_errors(env_ids_tensor)
            if episode_failed is not None and hasattr(self, "_update_file_success_stats"):
                self._update_file_success_stats(env_ids_tensor, episode_failed)
            scheduler.release(env_ids_tensor)

        assignment = scheduler.assign(env_ids_tensor)
        if assignment.envs.numel() == 0:
            env_ids_tensor.resize_(0)
            return

        env_ids_tensor.resize_(assignment.envs.numel())
        env_ids_tensor.copy_(assignment.envs)
        self.file_indices[env_ids_tensor] = assignment.files
        self.clip_start_steps[env_ids_tensor] = 0
        self.clip_end_steps[env_ids_tensor] = assignment.lengths
        self.time_steps[env_ids_tensor] = 0
        self.clip_step_count[env_ids_tensor] = 0.0
        self.clip_error_accum[env_ids_tensor] = 0.0

        if hasattr(self, "static_mask"):
            self.static_mask[env_ids_tensor] = False
        if hasattr(self, "static_ref_steps"):
            self.static_ref_steps[env_ids_tensor] = 0

        self._apply_command_to_robot(env_ids_tensor)

    motion_term._multi_file_sampling = types.MethodType(_scheduled_multi_file_sampling, motion_term)
    motion_term._folder_scheduler = scheduler
    return scheduler


def _compute_step_metrics(motion_term) -> dict[str, torch.Tensor]:
    """Compute per-env metric tensors for the current timestep."""
    metrics: dict[str, torch.Tensor] = {}

    body_pos_err = torch.norm(motion_term.body_pos_w - motion_term.robot_body_pos_w, dim=-1).mean(dim=1)
    metrics["E_mpjpe"] = body_pos_err

    joint_pos_err = torch.abs(motion_term.joint_pos - motion_term.robot_joint_pos).mean(dim=1)
    metrics["E_mpjae"] = joint_pos_err

    joint_vel_err = torch.abs(motion_term.joint_vel - motion_term.robot_joint_vel).mean(dim=1)
    metrics["E_mpjve"] = joint_vel_err

    root_pos_err = torch.norm(motion_term.anchor_pos_w - motion_term.robot_anchor_pos_w, dim=-1)
    metrics["E_root_p"] = root_pos_err

    root_rot_err = quat_error_magnitude(motion_term.anchor_quat_w, motion_term.robot_anchor_quat_w)
    metrics["E_root_r"] = root_rot_err

    root_vel_err = torch.norm(motion_term.anchor_lin_vel_w - motion_term.robot_anchor_lin_vel_w, dim=-1)
    metrics["E_root_v"] = root_vel_err

    return metrics


def _check_termination_conditions(motion_term, base_env) -> torch.Tensor:
    """Check if termination conditions are triggered (without actually terminating).

    Returns a boolean tensor indicating which envs would have failed.
    This replicates the logic from terminations.py but only for detection.
    """
    device = motion_term.device
    num_envs = motion_term.num_envs
    failed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # 1. anchor_pos_z: pelvis Z deviation > 0.3m
    anchor_z_err = torch.abs(motion_term.anchor_pos_w[:, -1] - motion_term.robot_anchor_pos_w[:, -1])
    failed |= (anchor_z_err > 0.3)

    # 2. anchor_ori: gravity projection difference > 0.8
    robot = getattr(base_env.scene, "robot", None)
    if robot is not None and hasattr(robot.data, "GRAVITY_VEC_W"):
        gravity_vec = robot.data.GRAVITY_VEC_W
        motion_proj_grav = quat_rotate_inverse(motion_term.anchor_quat_w, gravity_vec)
        robot_proj_grav = quat_rotate_inverse(motion_term.robot_anchor_quat_w, gravity_vec)
        ori_err = torch.abs(motion_proj_grav[:, 2] - robot_proj_grav[:, 2])
        failed |= (ori_err > 0.8)

    # 3. ee_body_pos_z: end-effector Z deviation > 0.3m
    ee_names = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    ee_indices = [i for i, name in enumerate(motion_term.cfg.body_names) if name in ee_names]
    if ee_indices:
        ee_z_err = torch.abs(
            motion_term.body_pos_relative_w[:, ee_indices, -1] - motion_term.robot_body_pos_w[:, ee_indices, -1]
        )
        failed |= torch.any(ee_z_err > 0.3, dim=-1)

    return failed


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate policy metrics for every motion inside a folder."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import pathlib

        import wandb
        from wandb.apis.public import Api

        run_path = args_cli.wandb_path

        api = Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is not None:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    log_dir = os.path.dirname(resume_path)

    # load saved run configs if present
    _load_run_configs(log_dir, env_cfg, agent_cfg)
    # ensure model arch matches saved run (for conv command encoder checkpoints)
    _apply_future_command_encoder_from_params(log_dir, agent_cfg)
    # align env motion command settings with policy encoder
    _sync_motion_command_with_policy(env_cfg, agent_cfg)
    # re-apply CLI overrides after loading saved configs
    cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # apply CLI overrides to env config
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # disable debug visualizations (joint/frame arrows and contact forces)
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.debug_vis = False
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "contact_forces"):
        env_cfg.scene.contact_forces.debug_vis = False

    folder_path = os.path.abspath(args_cli.motion_folder)
    clip_meta = _scan_motion_folder(env_cfg, folder_path)
    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "motion"):
        raise ValueError("Environment configuration has no motion command to override.")
    env_cfg.commands.motion.motion_dir = folder_path
    env_cfg.commands.motion.motion_file = None
    env_cfg.commands.motion.multi_file_mode = True
    env_cfg.commands.motion.start_from_zero_step = True

    total_jobs = len(clip_meta) * max(1, args_cli.eval_runs)
    if total_jobs == 0:
        raise RuntimeError("Motion folder evaluation requires at least one motion clip.")
    desired_envs = min(env_cfg.scene.num_envs, total_jobs)
    if desired_envs <= 0:
        desired_envs = 1
    if desired_envs != env_cfg.scene.num_envs:
        print(f"[INFO] Adjusting num_envs from {env_cfg.scene.num_envs} to {desired_envs} to match job count.")
        env_cfg.scene.num_envs = desired_envs

    clip_count = len(clip_meta)
    longest_duration = max((meta.duration or 0.0) for meta in clip_meta) or None
    runs_per_clip = max(1, args_cli.eval_runs)
    print(
        f"[INFO] Total scheduled runs: {total_jobs} = {clip_count} clips * {runs_per_clip} eval_runs"
    )
    if longest_duration:
        print(
            f"[INFO] Evaluating {clip_count} clips; longest duration {longest_duration:.2f}s."
        )
    else:
        print(f"[INFO] Evaluating {clip_count} clips.")

    _configure_play_reset_strategy(env_cfg, args_cli.respect_training_terminations, clip_duration=longest_duration)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = getattr(env, "unwrapped", env)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # Restore curriculum level for MoE models (sets _max_active_experts)
    _is_moe_policy = hasattr(ppo_runner.alg.policy, "experts") and hasattr(ppo_runner.alg.policy, "gate")
    if _is_moe_policy:
        policy = ppo_runner.alg.policy
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        curriculum_level = ckpt.get("curriculum_level", 1)
        num_experts = getattr(policy, "_num_experts", curriculum_level)
        # Phase B 检测：命令行 > agent.yaml > policy 属性
        is_phase_b = args_cli.phase_b
        if not is_phase_b:
            # 从 agent.yaml 检测
            agent_yaml_path = os.path.join(log_dir, "params", "agent.yaml")
            if os.path.isfile(agent_yaml_path):
                try:
                    with open(agent_yaml_path, "r", encoding="utf-8") as f:
                        agent_data = yaml.safe_load(f) or {}
                    moe_cfg = agent_data.get("policy", {}).get("moe_config", {})
                    is_phase_b = bool(moe_cfg.get("phase_b_enabled", False))
                except Exception:
                    pass
        if not is_phase_b:
            is_phase_b = getattr(policy, "_phase_b_enabled", False)
        if is_phase_b:
            active_experts = num_experts
            print(f"[INFO] MoE Phase B: all {num_experts} experts active")
        else:
            active_experts = curriculum_level
            print(f"[INFO] MoE: restored curriculum_level={curriculum_level}, max_active_experts={active_experts}")
        policy.set_max_active_experts(active_experts)
        policy.set_current_max_level(active_experts)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()
    try:
        _resize_obs_normalizer(ppo_runner.obs_normalizer, obs.shape[-1])
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to adjust obs normalizer shape: {exc}")
    else:
        print(f"[INFO] Eval obs dim={obs.shape[-1]}, normalizer dim={ppo_runner.obs_normalizer._mean.shape[-1]}")

    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_term"):
        raise RuntimeError("Command manager unavailable; cannot access motion term for metric computation.")
    motion_term = command_manager.get_term("motion")
    if motion_term is None:
        raise RuntimeError("Motion command term not found; cannot compute metrics.")

    scheduler = _install_folder_scheduler(motion_term, clip_meta, args_cli.eval_runs)
    metrics_acc = MetricsAccumulator(scheduler.clip_meta_by_index)

    while simulation_app.is_running():
        if scheduler.finished:
            break

        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        # Check termination conditions (without actually terminating) for success rate
        step_failed = _check_termination_conditions(motion_term, base_env)
        if step_failed.any():
            failed_env_ids = step_failed.nonzero(as_tuple=True)[0]
            scheduler.mark_failed(failed_env_ids)

        eval_envs, clip_indices = scheduler.eval_env_assignments()
        if eval_envs.numel() == 0:
            continue
        step_metrics = _compute_step_metrics(motion_term)
        # 过滤掉包含非有限值的环境，避免全局统计被 NaN/Inf 污染
        # 过滤掉包含非有限值或越界 env 的样本，避免全局统计被 NaN/Inf/越界污染
        all_finite = torch.ones_like(eval_envs, dtype=torch.bool, device=eval_envs.device)
        for metric_name in METRIC_NAMES:
            vals = step_metrics[metric_name]
            valid_idx = (eval_envs >= 0) & (eval_envs < vals.shape[0])
            per_env_finite = torch.zeros_like(eval_envs, dtype=torch.bool, device=eval_envs.device)
            if torch.any(valid_idx):
                per_env_finite[valid_idx] = torch.isfinite(vals[eval_envs[valid_idx]])
            all_finite &= (per_env_finite & valid_idx)
        if not all_finite.all():
            bad_envs = eval_envs[~all_finite]
            bad_clips = clip_indices[~all_finite]
            bad_names = [scheduler.clip_meta_by_index[int(idx)].path for idx in bad_clips.tolist()]
            print(f"[WARN] Detected non-finite metrics in envs {bad_envs.tolist()} for clips: {bad_names}")
        # 仅对有效环境累积指标（保持 step_metrics 原始形状，依赖 env 索引）
        eval_envs = eval_envs[all_finite]
        clip_indices = clip_indices[all_finite]
        if eval_envs.numel() == 0:
            continue
        metrics_acc.record(step_metrics, eval_envs, clip_indices)

    scheduler.finalize_progress()
    runs_requested = scheduler.total_jobs
    runs_completed = scheduler.completed_jobs
    if metrics_acc.total_steps == 0:
        raise RuntimeError("No simulation steps executed; unable to compute metrics.")

    if runs_completed < runs_requested:
        print(f"[WARN] Completed {runs_completed}/{runs_requested} scheduled motion runs before exit.")

    averaged_metrics = metrics_acc.global_average()
    per_clip_metrics = metrics_acc.clip_averages()

    # Add success rate info to per-clip metrics
    for clip_info in per_clip_metrics:
        clip_name = clip_info["name"]
        clip_idx = next(
            (i for i, meta in enumerate(scheduler.clip_meta_by_index) if meta.name == clip_name),
            None,
        )
        if clip_idx is not None:
            successes = scheduler.clip_success[clip_idx]
            failures = scheduler.clip_failure[clip_idx]
            total_runs = successes + failures
            clip_info["runs"] = total_runs
            clip_info["successes"] = successes
            clip_info["failures"] = failures
            clip_info["success_rate"] = successes / max(1, total_runs)

    # Compute global success rate
    global_success_rate = scheduler.success_count / max(1, scheduler.completed_jobs)

    threshold = args_cli.mpjpe_print_threshold
    filtered_clips = [
        clip
        for clip in per_clip_metrics
        if threshold is None or clip["metrics"]["E_mpjpe"] >= threshold
    ]
    result = {
        "runs_requested": runs_requested,
        "runs_completed": runs_completed,
        "steps_evaluated": metrics_acc.total_steps,
        "clips_evaluated": len(clip_meta),
        "motion_folder": args_cli.motion_folder,
        "success_rate": global_success_rate,
        "successes": scheduler.success_count,
        "failures": scheduler.failure_count,
        "metrics": averaged_metrics,
        "clips": filtered_clips,
    }

    run_name = getattr(agent_cfg, "load_run", None) or getattr(agent_cfg, "experiment_name", None) or "default"
    output_dir = os.path.join("logs", "rsl_rl", "eval", f"{run_name}_metrics")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics_folder.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("[RESULT] Evaluation metrics:")
    for name, value in averaged_metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"  success_rate: {global_success_rate:.4f} ({scheduler.success_count}/{scheduler.completed_jobs})")
    listed = len(filtered_clips)
    total = len(per_clip_metrics)
    threshold_desc = threshold if threshold is not None else "ALL"
    print(f"[RESULT] Clips listed in JSON: {listed}/{total} (threshold={threshold_desc})")
    print(f"[RESULT] Saved metrics to: {output_path}")

    # close the simulator and env
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore[misc]
    # close sim app
    simulation_app.close()

