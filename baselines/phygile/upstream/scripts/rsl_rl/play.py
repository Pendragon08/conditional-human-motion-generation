"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import os
import pickle
import sys
from dataclasses import fields, is_dataclass
from typing import Optional

import numpy as np
import yaml
from yaml.constructor import ConstructorError
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--urdf", type=str, default=None, help="URDF filename or path (e.g. main_no_hand_collision.urdf). Default: main.urdf")
parser.add_argument(
    "--play_full_motion",
    action="store_true",
    default=False,
    help="Replay the entire motion sequence (no random clip sampling, single-file mode).",
)
parser.add_argument(
    "--no_start_from_zero",
    dest="start_from_zero",
    action="store_false",
    help="Allow motion clips to start from random offsets during playback.",
)
parser.add_argument(
    "--respect_training_terminations",
    action="store_true",
    help="Keep training termination logic during play instead of running the full clip.",
)
parser.add_argument(
    "--phase_b",
    action="store_true",
    help="MoE Phase B mode: activate all experts regardless of curriculum_level.",
)
parser.set_defaults(start_from_zero=True)
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

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner


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
            # ManagerBasedRLEnvCfg currently expects seed to be NoneType when validated.
            special_fields["seed"] = data_local.pop("seed")

        try:
            _coerce_numeric_types(cfg_obj, data_local)
            _prime_cfg_types(cfg_obj, data_local)
            cfg_obj.from_dict(data_local)
        except Exception:
            # restore original dictionary before re-raising so callers can debug contents
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
                cfg_dict = {key: copy.deepcopy(value) for key, value in vars(loaded_cfg).items() if not key.startswith("__")}

            # convert any slice-strings back? assume already proper types
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
    """Ensure policy.future_command_encoder matches the saved run config."""
    current_cfg = getattr(getattr(agent_cfg, "policy", None), "future_command_encoder", None)
    if current_cfg:
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
            print("[INFO] Applied future_command_encoder from saved agent.yaml.")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to apply future_command_encoder from agent.yaml: {exc}")


def _sync_motion_command_with_policy(env_cfg, agent_cfg) -> None:
    """Keep env command frame counts aligned with policy future_command_encoder."""
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    policy_cfg = getattr(getattr(agent_cfg, "policy", None), "future_command_encoder", None)
    if motion_cfg is None or policy_cfg is None:
        return

    # expected counts from policy side
    policy_immediate = policy_cfg.get("immediate_steps")
    policy_conv_steps = policy_cfg.get("conv_steps") or policy_cfg.get("future_steps")

    # fill missing fields on env side
    if getattr(motion_cfg, "future_steps", None) is None and policy_immediate is not None:
        motion_cfg.future_steps = int(policy_immediate)
        print(f"[INFO] env.commands.motion.future_steps set to {motion_cfg.future_steps} from policy.")
    if getattr(motion_cfg, "encoder_future_steps", None) is None and policy_conv_steps is not None:
        motion_cfg.encoder_future_steps = int(policy_conv_steps)
        print(f"[INFO] env.commands.motion.encoder_future_steps set to {motion_cfg.encoder_future_steps} from policy.")

    # warn on mismatch (will keep env cfg as loaded to avoid silent bugs)
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
            [normalizer._mean, torch.zeros(pad, device=device, dtype=normalizer._mean.dtype)], dim=-1
        )
        normalizer._std = torch.cat(
            [normalizer._std, torch.ones(pad, device=device, dtype=normalizer._std.dtype)], dim=-1
        )
        print(f"[INFO] Obs normalizer padded from {current_dim} -> {target_dim}.")
    else:
        normalizer._mean = normalizer._mean[..., :target_dim]
        normalizer._std = normalizer._std[..., :target_dim]
        print(f"[INFO] Obs normalizer truncated from {current_dim} -> {target_dim}.")


def _infer_motion_clip_stats(env_cfg) -> tuple[Optional[float], Optional[int]]:
    """Return (duration_seconds, frame_count) for the configured single-file motion."""
    commands = getattr(env_cfg, "commands", None)
    if commands is None or not hasattr(commands, "motion"):
        return None, None
    motion_cfg = commands.motion
    motion_file = getattr(motion_cfg, "motion_file", None)
    if motion_file is None or not os.path.isfile(motion_file):
        return None, None
    try:
        with np.load(motion_file) as npz:
            joint_pos = npz["joint_pos"]
            fps = float(npz["fps"]) if "fps" in npz else None
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to read motion clip stats from {motion_file}: {exc}")
        return None, None

    frame_count = int(joint_pos.shape[0])
    if frame_count <= 0:
        return None, None

    if fps is None or fps <= 0.0:
        sim_dt = getattr(getattr(env_cfg, "sim", None), "dt", None)
        decimation = getattr(env_cfg, "decimation", None)
        if sim_dt and decimation:
            fps = 1.0 / (float(sim_dt) * float(decimation))
    duration_s = frame_count / fps if fps and fps > 0.0 else None
    return duration_s, frame_count


def _disable_training_terminations(term_cfg) -> dict[str, object]:
    """Remove non timeout termination terms from the config and return them for logging."""
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
        if term is None:
            continue
        if getattr(term, "time_out", False):
            continue
        disabled_terms[name] = term
        setattr(term_cfg, name, None)
    return disabled_terms


def _configure_play_reset_strategy(env_cfg, respect_training_terminations: bool) -> None:
    """Adjust env configuration so play episodes only reset at clip end/time."""
    # Always try to set episode length to motion duration
    duration_s, frame_count = _infer_motion_clip_stats(env_cfg)
    if duration_s is not None:
        try:
            env_cfg.episode_length_s = float(duration_s)
            print(f"[INFO] Play episode length set to motion duration: {duration_s:.2f}s, total frames: {frame_count}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to update episode length for play: {exc}")

    if respect_training_terminations:
        print("[INFO] Keeping training termination conditions active.")
        return

    term_cfg = getattr(env_cfg, "terminations", None)
    disabled_terms = _disable_training_terminations(term_cfg)
    if disabled_terms:
        print(f"[INFO] Disabled training termination terms for play: {list(disabled_terms.keys())}")


def _lock_motion_sampling_to_start(motion_term, context: str) -> bool:
    """Force a single-file motion command to always restart from timestep zero."""
    if getattr(motion_term, "multi_file_mode", False):
        print(f"[WARN] {context}: Only single motion files are supported for start-from-zero playback.")
        return False
    if getattr(motion_term, "_start_from_zero_sampling_locked", False):
        return True

    def _reset_to_start(env_ids):
        env_ids_tensor = torch.as_tensor(
            env_ids, device=motion_term.time_steps.device, dtype=torch.long
        ).reshape(-1)
        if env_ids_tensor.numel() == 0:
            return
        motion_term.time_steps[env_ids_tensor] = 0

    motion_term._adaptive_sampling = _reset_to_start  # type: ignore[attr-defined]
    motion_term.time_steps.zero_()
    setattr(motion_term, "_start_from_zero_sampling_locked", True)
    return True


def _enable_full_motion_playback(base_env) -> None:
    """Force motion command to replay entire sequence from the beginning."""
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None:
        print("[WARN] Command manager not found; cannot enable full-motion playback.")
        return
    try:
        motion_term = command_manager.get_term("motion")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to access motion term: {exc}")
        return
    if motion_term is None:
        print("[WARN] Motion term not available; cannot enable full-motion playback.")
        return
    if _lock_motion_sampling_to_start(motion_term, "Full-motion playback"):
        print("[INFO] Full-motion playback enabled; each episode runs the entire clip from the start.")


def _force_start_of_motion(base_env, log_resets: bool = True) -> None:
    """Ensure motion clips reset at t=0 and optionally log reset events."""
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None:
        print("[WARN] Command manager not found; cannot force start-of-motion resets.")
        return
    try:
        motion_term = command_manager.get_term("motion")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to access motion term: {exc}")
        return
    if motion_term is None:
        print("[WARN] Motion term not available; cannot force start-of-motion resets.")
        return

    if hasattr(motion_term, "cfg") and hasattr(motion_term.cfg, "start_from_zero_step"):
        motion_term.cfg.start_from_zero_step = True
    if hasattr(motion_term, "start_from_zero_step"):
        setattr(motion_term, "start_from_zero_step", True)
    if hasattr(motion_term, "time_steps"):
        try:
            motion_term.time_steps.zero_()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to reset motion time steps: {exc}")

    if log_resets and hasattr(motion_term, "_resample_command") and not getattr(
        motion_term, "_start_from_zero_log_hooked", False
    ):
        original_resample = motion_term._resample_command

        def _resample_with_log(self, env_ids):  # type: ignore[override]
            if env_ids is not None:
                env_tensor = torch.as_tensor(env_ids, device=self.device).reshape(-1)
                if env_tensor.numel() > 0:
                    print(f"[PLAY] Resetting env ids {env_tensor.tolist()} to motion start (t=0).")
            return original_resample(env_ids)

        import types

        motion_term._resample_command = types.MethodType(_resample_with_log, motion_term)
        setattr(motion_term, "_start_from_zero_log_hooked", True)

    if _lock_motion_sampling_to_start(motion_term, "Start-from-zero"):
        print("[INFO] Motion command configured to start each episode from t=0.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb
        from wandb.apis.public import Api

        run_path = args_cli.wandb_path

        api = Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    log_dir = os.path.dirname(resume_path)

    # load saved run configs if present
    _load_run_configs(log_dir, env_cfg, agent_cfg)
    # ensure future command encoder is restored from saved agent config (for conv checkpoints)
    _apply_future_command_encoder_from_params(log_dir, agent_cfg)
    # align motion command frame counts with policy conv encoder settings
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

    # always disable static command sampling during play runs
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is not None and hasattr(motion_cfg, "random_static_prob"):
        if getattr(motion_cfg, "random_static_prob", 0.0) != 0.0:
            print("[INFO] Disabling random static sampling for play mode.")
        motion_cfg.random_static_prob = 0.0

    # override motion source via CLI for single-file playback
    if args_cli.motion_file is not None:
        motion_path = os.path.abspath(args_cli.motion_file)
        if not os.path.isfile(motion_path):
            raise FileNotFoundError(f"Motion file does not exist: {motion_path}")
        if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "motion"):
            raise ValueError("Environment configuration has no motion command to override.")
        env_cfg.commands.motion.motion_dir = None
        env_cfg.commands.motion.motion_file = motion_path
        print(f"[INFO] Overriding motion source for play: {motion_path}")

    _configure_play_reset_strategy(env_cfg, args_cli.respect_training_terminations)

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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = getattr(env, "unwrapped", env)
    if args_cli.start_from_zero:
        _force_start_of_motion(base_env, log_resets=True)
    if args_cli.play_full_motion:
        _enable_full_motion_playback(base_env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # grab initial observations to align normalizer with current obs shape
    obs, _ = env.get_observations()
    try:
        _resize_obs_normalizer(ppo_runner.obs_normalizer, obs.shape[-1])
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to adjust obs normalizer shape: {exc}")
    else:
        print(f"[INFO] Play obs dim={obs.shape[-1]}, normalizer dim={ppo_runner.obs_normalizer._mean.shape[-1]}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    # Extract iteration number from checkpoint filename (e.g., model_120000.pt -> 120000)
    import re
    checkpoint_basename = os.path.basename(resume_path)
    iteration_match = re.search(r"model_(\d+)\.pt", checkpoint_basename)
    
    # Get motion file name prefix (e.g., 005265 from 005265.npz)
    motion_file_for_name = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_file_for_name is not None:
        motion_file_for_name = getattr(motion_file_for_name, "motion_file", None)
    motion_prefix = ""
    if motion_file_for_name is not None:
        motion_basename = os.path.basename(motion_file_for_name)
        motion_prefix = os.path.splitext(motion_basename)[0] + "_"
    
    if iteration_match:
        iteration_num = iteration_match.group(1)
        onnx_filename = f"{motion_prefix}policy_{iteration_num}.onnx"
    else:
        onnx_filename = f"{motion_prefix}policy.onnx"

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename=onnx_filename,
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir, onnx_filename)

    # copy motion file to export directory
    motion_file_path = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_file_path is not None:
        motion_file_path = getattr(motion_file_path, "motion_file", None)
    if motion_file_path is not None and os.path.isfile(motion_file_path):
        import shutil
        motion_dst = os.path.join(export_model_dir, os.path.basename(motion_file_path))
        shutil.copy2(motion_file_path, motion_dst)
        print(f"[INFO] Copied motion file to: {motion_dst}")

    # reset environment (reuse initial obs if available)
    # obs already captured above; re-fetch to ensure up-to-date state
    obs, _ = env.get_observations()
    timestep = 0

    # Setup termination diagnostics
    motion_term = None
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is not None:
        try:
            motion_term = command_manager.get_term("motion")
        except Exception:
            pass

    # Track errors from previous step (before reset happens)
    prev_errors = {"anchor_pos": 0.0, "anchor_z": 0.0, "anchor_ori": 0.0, "ee_z": 0.0, "dof_mean": 0.0, "dof_max": 0.0}
    episode_step = 0

    def _compute_current_errors():
        """Compute current tracking errors."""
        if motion_term is None:
            return {"anchor_pos": 0.0, "anchor_z": 0.0, "anchor_ori": 0.0, "ee_z": 0.0, "dof_mean": 0.0, "dof_max": 0.0}
        try:
            robot = base_env.scene["robot"]
            # Anchor position error (3D norm)
            anchor_pos_err = torch.norm(
                motion_term.anchor_pos_w - motion_term.robot_anchor_pos_w, dim=-1
            ).item()
            # Anchor Z error only
            anchor_z_err = torch.abs(
                motion_term.anchor_pos_w[:, -1] - motion_term.robot_anchor_pos_w[:, -1]
            ).item()
            # Anchor orientation error
            import isaaclab.utils.math as math_utils
            motion_grav = math_utils.quat_rotate_inverse(motion_term.anchor_quat_w, robot.data.GRAVITY_VEC_W)
            robot_grav = math_utils.quat_rotate_inverse(motion_term.robot_anchor_quat_w, robot.data.GRAVITY_VEC_W)
            anchor_ori_err = (motion_grav[:, 2] - robot_grav[:, 2]).abs().item()
            # EE body pos Z error
            body_names = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
            body_indices = [i for i, name in enumerate(motion_term.cfg.body_names) if name in body_names]
            ee_z_err = torch.abs(
                motion_term.body_pos_relative_w[:, body_indices, -1]
                - motion_term.robot_body_pos_w[:, body_indices, -1]
            )
            ee_max_err = ee_z_err.max().item() if ee_z_err.numel() > 0 else 0.0
            # DOF (joint position) error
            ref_joint_pos = motion_term.joint_pos  # Reference joint positions
            robot_joint_pos = robot.data.joint_pos  # Current robot joint positions
            dof_err = torch.abs(ref_joint_pos - robot_joint_pos)
            dof_mean_err = dof_err.mean().item()
            dof_max_err = dof_err.max().item()
            return {
                "anchor_pos": anchor_pos_err,
                "anchor_z": anchor_z_err,
                "anchor_ori": anchor_ori_err,
                "ee_z": ee_max_err,
                "dof_mean": dof_mean_err,
                "dof_max": dof_max_err,
            }
        except Exception as e:
            return {"anchor_pos": 0.0, "anchor_z": 0.0, "anchor_ori": 0.0, "ee_z": 0.0, "dof_mean": 0.0, "dof_max": 0.0}

    # simulate environment
    debug_print_steps = 0  # 只打印前几个 step 的 obs
    while simulation_app.is_running():
        # Compute errors BEFORE stepping (so we capture pre-reset values)
        prev_errors = _compute_current_errors()

        # === DEBUG: 打印 obs 值用于对齐检查 ===
        if timestep < debug_print_steps:
            obs_np = obs[0].cpu().numpy() if isinstance(obs, torch.Tensor) else obs[0]
            print(f"\n{'='*60}")
            print(f"[DEBUG OBS] Step {timestep}, obs shape: {obs_np.shape}")
            
            # 解析观察空间结构 (616 = 520 + 6 + 3 + 29 + 29 + 29)
            cmd_dim = 520
            ori_dim = 6
            ang_dim = 3
            jpos_dim = 29
            jvel_dim = 29
            act_dim = 29
            
            idx = 0
            command = obs_np[idx:idx+cmd_dim]; idx += cmd_dim
            motion_anchor_ori_b = obs_np[idx:idx+ori_dim]; idx += ori_dim
            base_ang_vel = obs_np[idx:idx+ang_dim]; idx += ang_dim
            joint_pos = obs_np[idx:idx+jpos_dim]; idx += jpos_dim
            joint_vel = obs_np[idx:idx+jvel_dim]; idx += jvel_dim
            last_action = obs_np[idx:idx+act_dim]; idx += act_dim
            
            # command 结构: [current_frame(65) + future_2frames(130) + encoder_5frames(325)]
            # 每帧: joint_pos(29) + joint_vel(29) + anchor_pos(3) + anchor_quat(4) = 65
            frame0 = command[:65]
            print(f"  command[0:65] (frame0):")
            print(f"    joint_pos[0:5]:   {frame0[0:5]}")
            print(f"    joint_vel[0:5]:   {frame0[29:34]}")
            print(f"    anchor_pos:       {frame0[58:61]}")
            print(f"    anchor_quat(wxyz):{frame0[61:65]}")
            
            print(f"  motion_anchor_ori_b (6): {motion_anchor_ori_b}")
            print(f"  base_ang_vel (3):        {base_ang_vel}")
            print(f"  joint_pos[0:10]:         {joint_pos[:10]}")
            print(f"  joint_vel[0:10]:         {joint_vel[:10]}")
            print(f"  last_action[0:10]:       {last_action[:10]}")
            print(f"{'='*60}\n")

        # agent inference
        with torch.inference_mode():
            actions = policy(obs)
        # env stepping
        obs, _, dones, _ = env.step(actions)
        timestep += 1
        episode_step += 1

        # Log errors every 50 steps
        if episode_step % 50 == 0:
            print(
                f"[STEP] ep_step={episode_step} | "
                f"anchor_pos={prev_errors['anchor_pos']:.4f} | "
                f"anchor_z={prev_errors['anchor_z']:.4f} | "
                f"anchor_ori={prev_errors['anchor_ori']:.4f} | "
                f"ee_z={prev_errors['ee_z']:.4f} | "
                f"dof_mean={prev_errors['dof_mean']:.4f} | "
                f"dof_max={prev_errors['dof_max']:.4f}"
            )

        # Log diagnostics when reset is triggered
        if dones is not None:
            done_triggered = False
            is_terminated = False
            if isinstance(dones, torch.Tensor) and dones.any():
                done_triggered = True
                term_mgr = getattr(base_env, "termination_manager", None)
                if term_mgr is not None and hasattr(term_mgr, "terminated"):
                    is_terminated = term_mgr.terminated.any().item()
            elif isinstance(dones, bool) and dones:
                done_triggered = True
            if done_triggered:
                reason = "TERMINATED" if is_terminated else "TIMEOUT"
                print(
                    f"[RESET] ep_step={episode_step} reason={reason} | "
                    f"anchor_pos={prev_errors['anchor_pos']:.4f} | "
                    f"anchor_z={prev_errors['anchor_z']:.4f} | "
                    f"anchor_ori={prev_errors['anchor_ori']:.4f} | "
                    f"ee_z={prev_errors['ee_z']:.4f} | "
                    f"dof_mean={prev_errors['dof_mean']:.4f} | "
                    f"dof_max={prev_errors['dof_max']:.4f}"
                )
                episode_step = 0  # Reset episode step counter

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore[misc]
    # close sim app
    simulation_app.close()
