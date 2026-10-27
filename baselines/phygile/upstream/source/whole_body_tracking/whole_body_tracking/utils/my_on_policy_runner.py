"""Extended OnPolicyRunner with curriculum learning support."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

import torch
from rsl_rl.env import VecEnv
import rsl_rl.modules as _rsl_modules
import rsl_rl.runners.on_policy_runner as _rsl_runner_mod
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.encoder_actor_critic import EncoderActorCritic, MoEActorCritic
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

if TYPE_CHECKING:
    from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


def _register_encoder_actor_critic() -> None:
    """Ensure RSL-RL registries can resolve our custom policy."""
    try:
        if getattr(_rsl_modules, "EncoderActorCritic", None) is not EncoderActorCritic:
            setattr(_rsl_modules, "EncoderActorCritic", EncoderActorCritic)
    except Exception:
        pass
    try:
        if getattr(_rsl_runner_mod, "EncoderActorCritic", None) is not EncoderActorCritic:
            setattr(_rsl_runner_mod, "EncoderActorCritic", EncoderActorCritic)
    except Exception:
        pass


def _register_moe_classes() -> None:
    """Register MoEActorCritic and MoEPPO so ``eval()`` in OnPolicyRunner can find them."""
    from whole_body_tracking.utils.moe_ppo import MoEPPO  # noqa: F811
    for mod in (_rsl_modules, _rsl_runner_mod):
        try:
            if getattr(mod, "MoEActorCritic", None) is not MoEActorCritic:
                setattr(mod, "MoEActorCritic", MoEActorCritic)
        except Exception:
            pass
        try:
            if getattr(mod, "MoEPPO", None) is not MoEPPO:
                setattr(mod, "MoEPPO", MoEPPO)
        except Exception:
            pass


_register_encoder_actor_critic()
_register_moe_classes()


def _save_checkpoint_without_forced_upload(runner: OnPolicyRunner, path: str, infos=None):
    """Copy of OnPolicyRunner.save with upload gated by runner.upload_models_to_wandb."""
    saved_dict = {
        "model_state_dict": runner.alg.policy.state_dict(),
        "optimizer_state_dict": runner.alg.optimizer.state_dict(),
        "iter": runner.current_learning_iteration,
        "infos": infos,
    }
    if runner.alg.rnd:
        saved_dict["rnd_state_dict"] = runner.alg.rnd.state_dict()
        saved_dict["rnd_optimizer_state_dict"] = runner.alg.rnd_optimizer.state_dict()
    if runner.empirical_normalization:
        saved_dict["obs_norm_state_dict"] = runner.obs_normalizer.state_dict()
        saved_dict["privileged_obs_norm_state_dict"] = runner.privileged_obs_normalizer.state_dict()
    
    # 保存 curriculum level（如果存在）
    if hasattr(runner, "_get_motion_term"):
        try:
            motion_term = runner._get_motion_term()
            if motion_term is not None and hasattr(motion_term, "get_curriculum_level"):
                curriculum_level = motion_term.get_curriculum_level()
                if curriculum_level > 0:
                    saved_dict["curriculum_level"] = curriculum_level
        except Exception:
            pass  # 如果获取失败，忽略
    
    torch.save(saved_dict, path)
    should_upload = (
        getattr(runner, "logger_type", None) in ["neptune", "wandb"]
        and not getattr(runner, "disable_logs", False)
        and getattr(runner, "upload_models_to_wandb", True)
    )
    writer = getattr(runner, "writer", None)
    if should_upload and writer is not None:
        writer.save_model(path, runner.current_learning_iteration)


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        _save_checkpoint_without_forced_upload(self, path, infos)
        if self.logger_type in ["wandb"] and getattr(self, "upload_models_to_wandb", True):
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)  # type: ignore[attr-defined]
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))  # type: ignore[attr-defined]


# =============================================================================
# Curriculum Evaluation Support (EMA-based, no tensor modifications)
# =============================================================================


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: str | None = None,
        upload_models_to_wandb: bool = True,
        task_name: str | None = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self.task_name = task_name
        self.upload_models_to_wandb = upload_models_to_wandb
        
        # Curriculum level 变化时的参数调整
        self._last_curriculum_level: int = 0
        self._initial_learning_rate: float = self.alg.learning_rate  # 保存初始学习率
        self._initial_desired_kl: float = self.alg.desired_kl  # 保存初始 desired_kl
        self._initial_entropy_coef: float = self.alg.entropy_coef  # 保存初始 entropy_coef
        
        # 参数渐变恢复的跟踪
        self._level_up_iteration: int | None = None  # 升级时的迭代数
        self._param_decay_iters: int = 0  # 参数恢复的迭代数（从配置读取）

        # ---- Reward 收敛检测（用于 curriculum 提前升级）----
        # ---- Eval-metric 收敛检测（用于 curriculum 提前升级）----
        # 用 mpjpe/mpjae 的 eval 趋势判断收敛，比 reward 更稳定不震荡
        _cmd_cfg = train_cfg.get("commands", {}).get("motion", {})
        self._metric_conv_enabled: bool = bool(_cmd_cfg.get("curriculum_metric_convergence_enabled", True))
        # 连续 N 次 eval 改善率都 < threshold 才判为收敛
        self._metric_conv_evals: int = int(_cmd_cfg.get("curriculum_metric_convergence_evals", 3))
        self._metric_conv_threshold: float = float(_cmd_cfg.get("curriculum_metric_convergence_threshold", 0.03))
        self._metric_conv_min_iters: int = int(_cmd_cfg.get("curriculum_metric_convergence_min_iters", 3000))
        # 存储 eval 历史: [(mpjpe, mpjae), ...]
        self._eval_metric_history: list[tuple[float, float]] = []

        # ---- MoE auto-detection: upgrade PPO → MoEPPO when needed ----
        self._is_moe: bool = isinstance(self.alg.policy, MoEActorCritic)
        if self._is_moe:
            self._upgrade_to_moe_ppo()
            motion_term = self._get_motion_term()
            if motion_term is not None:
                level = motion_term.get_curriculum_level()
                if level > 0:
                    self.alg.policy.set_max_active_experts(level)
                    self.alg.policy.set_current_max_level(level)
                # Clear file-level cache on resume to ensure fresh mapping
                if hasattr(motion_term, "invalidate_file_level_cache"):
                    motion_term.invalidate_file_level_cache()

            # Phase B: mark that experts need unlocking after checkpoint load
            moe_cfg_init = train_cfg.get("policy", {}).get("moe_config") or {}
            self._phase_b_pending_unlock: bool = bool(moe_cfg_init.get("phase_b_enabled", False))

        # ---- Phase B configuration (dynamic expert addition) ----
        # Skip training-specific config when in play/inference mode (log_dir=None)
        self._phase_b_enabled: bool = False
        self._phase_b_check_interval: int = 500
        self._phase_b_new_expert_lr_scale: float = 3.0
        self._phase_b_old_expert_lr_scale: float = 0.1
        self._moe_expert_warm_start: bool = True  # Default: enable warm start
        from whole_body_tracking.utils.moe_ppo import DifficultyThresholds as _DT
        self._phase_b_thresholds: _DT | None = None
        if self._is_moe and log_dir is not None:
            # Only load training-specific MoE config when actually training
            moe_cfg = train_cfg.get("policy", {}).get("moe_config") or {}
            # Expert warm start: copy previous expert weights when upgrading level
            self._moe_expert_warm_start = bool(moe_cfg.get("expert_warm_start", True))
            self._phase_b_enabled = bool(moe_cfg.get("phase_b_enabled", False))
            if self._phase_b_enabled:
                from whole_body_tracking.utils.moe_ppo import DifficultyThresholds
                self._phase_b_thresholds = DifficultyThresholds(
                    err=float(moe_cfg.get("difficulty_err_threshold", 0.15)),
                    entropy=float(moe_cfg.get("difficulty_entropy_threshold", 0.8)),
                    gap=float(moe_cfg.get("difficulty_gap_threshold", 0.15)),
                    sustained_windows=int(moe_cfg.get("difficulty_sustained_windows", 30)),
                    trigger_ratio=float(moe_cfg.get("difficulty_trigger_ratio", 0.1)),
                    check_interval=int(moe_cfg.get("difficulty_check_interval", 500)),
                )
                self._phase_b_check_interval = self._phase_b_thresholds.check_interval
                self._phase_b_new_expert_lr_scale = float(moe_cfg.get("new_expert_lr_scale", 3.0))
                self._phase_b_old_expert_lr_scale = float(moe_cfg.get("old_expert_lr_scale", 0.1))
                self._phase_b_boost_lr_scale = float(moe_cfg.get("boost_lr_scale", 2.0))
                self._phase_b_boost_lr_duration = int(moe_cfg.get("boost_lr_duration_iters", 3000))

                # Register per-file difficulty tracking
                motion_term = self._get_motion_term()
                if motion_term is not None and hasattr(motion_term, "file_indices"):
                    num_files = getattr(motion_term, "num_motion_files", 0)
                    if num_files > 0:
                        def _get_file_indices():
                            mt = self._get_motion_term()
                            if mt is not None and hasattr(mt, "file_indices"):
                                return mt.file_indices
                            return None
                        self.alg.set_file_indices_getter(_get_file_indices)
                        self.alg.init_file_tracking(num_files)

                print(f"[Phase B] Enabled: thresholds={self._phase_b_thresholds}")

        # 将 log_dir 传递给底层 env/motion term，便于 play/eval 也能写 freeze.txt
        try:
            env_unwrapped = getattr(env, "unwrapped", env)
            if log_dir and not getattr(env_unwrapped, "log_dir", None):
                setattr(env_unwrapped, "log_dir", log_dir)
            command_manager = getattr(env_unwrapped, "command_manager", None)
            if command_manager is not None:
                motion_term = command_manager.get_term("motion")
                if motion_term is not None and hasattr(motion_term, "set_log_dir"):
                    motion_term.set_log_dir(log_dir)
        except Exception:
            pass

    # ================================================================= #
    # MoE support (auto-detected; no-op when policy is plain MLP)
    # ================================================================= #

    def _upgrade_to_moe_ppo(self) -> None:
        """Replace the standard PPO with MoEPPO, transferring all state."""
        from whole_body_tracking.utils.moe_ppo import MoEPPO

        old = self.alg
        if isinstance(old, MoEPPO):
            return

        moe_ppo = MoEPPO(
            old.policy,
            num_learning_epochs=old.num_learning_epochs,
            num_mini_batches=old.num_mini_batches,
            clip_param=old.clip_param,
            gamma=old.gamma,
            lam=old.lam,
            value_loss_coef=old.value_loss_coef,
            entropy_coef=old.entropy_coef,
            learning_rate=old.learning_rate,
            max_grad_norm=old.max_grad_norm,
            use_clipped_value_loss=old.use_clipped_value_loss,
            schedule=old.schedule,
            desired_kl=old.desired_kl,
            device=str(self.device),
        )
        moe_ppo.storage = old.storage
        moe_ppo.optimizer = old.optimizer
        moe_ppo.transition = old.transition
        if old.storage is not None:
            num_envs = old.storage.num_envs
            moe_ppo._level_buffer = torch.zeros(
                old.storage.num_transitions_per_env,
                num_envs,
                dtype=torch.long,
                device=self.device,
            )
            # Per-file tracking is initialised later in Phase B config block
        self.alg = moe_ppo

        # Register auto-fetch for level labels so parent learn() works unmodified
        def _get_level_labels():
            mt = self._get_motion_term()
            if mt is not None and hasattr(mt, "get_env_difficulty_levels"):
                return mt.get_env_difficulty_levels()
            return None
        moe_ppo.set_level_label_getter(_get_level_labels)

        print("[MoE] Auto-detected MoEActorCritic, upgraded PPO -> MoEPPO")

    # learn() is NOT overridden — both MLP and MoE use the parent
    # OnPolicyRunner.learn().  MoE level labels are auto-fetched inside
    # MoEPPO.act() via the getter registered in _upgrade_to_moe_ppo().

    # ---- Phase B: dynamic expert addition ----

    def _check_and_add_expert(self, iteration: int) -> None:
        """Check per-file difficulty and add expert or boost LR."""
        if self._phase_b_thresholds is None:
            return
        if not hasattr(self.alg, "check_difficulty"):
            return

        result = self.alg.check_difficulty(self._phase_b_thresholds)

        if result == "add":
            policy = self.alg.policy
            new_id = policy.add_expert()
            if new_id < 0:
                return

            policy.set_max_active_experts(new_id + 1)
            self.alg.rebuild_optimizer_for_new_expert(
                new_expert_lr_scale=self._phase_b_new_expert_lr_scale,
                old_expert_lr_scale=self._phase_b_old_expert_lr_scale,
            )

            if self.alg._file_difficulty_counter is not None:
                self.alg._file_difficulty_counter.zero_()

            if self.writer is not None and not getattr(self, "disable_logs", False):
                self.writer.add_scalar("PhaseB/expert_added", new_id, iteration)
                self.writer.add_scalar("PhaseB/total_experts", policy._num_experts, iteration)

            print(
                f"[Phase B] Expert {new_id} added at iteration {iteration} "
                f"(total={policy._num_experts}, max_active={policy._max_active_experts})"
            )

        elif result == "boost":
            self.alg.boost_expert_lr(
                current_iter=iteration,
                scale=getattr(self, "_phase_b_boost_lr_scale", 2.0),
                duration_iters=getattr(self, "_phase_b_boost_lr_duration", 3000),
            )
            if self.alg._file_difficulty_counter is not None:
                self.alg._file_difficulty_counter.zero_()
            if self.writer is not None and not getattr(self, "disable_logs", False):
                self.writer.add_scalar("PhaseB/lr_boost_triggered", 1, iteration)

    # ================================================================= #
    # End of MoE support
    # ================================================================= #

    def _append_file_sampling_top3_log(self, model_path: str) -> None:
        try:
            env = getattr(self.env, "unwrapped", self.env)
            command_manager = getattr(env, "command_manager", None)
            if command_manager is None:
                return
            motion_term = command_manager.get_term("motion")
        except Exception:
            return

        if not hasattr(motion_term, "get_file_sampling_top3"):
            return
        top_names = motion_term.get_file_sampling_top3()
        if not top_names:
            return

        iteration = getattr(self, "current_learning_iteration", None)
        if iteration is None:
            iteration = 0
        names = list(top_names[:3])
        while len(names) < 3:
            names.append("")
        log_line = f"{int(iteration)} {' '.join(names)}\n"
        log_path = os.path.join(os.path.dirname(model_path), "file_sampling_top3.txt")
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(log_line)
        except Exception:
            pass

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        _save_checkpoint_without_forced_upload(self, path, infos)
        self._append_file_sampling_top3_log(path)
        
        # 尝试恢复 curriculum level（在保存后，用于下次加载）
        # 这里不需要做什么，因为保存时已经包含了 level
        if self.logger_type in ["wandb"] and getattr(self, "upload_models_to_wandb", True):
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)  # type: ignore[attr-defined]
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))  # type: ignore[attr-defined]

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)  # type: ignore[attr-defined]
                self.registry_name = None

    def _get_motion_term(self) -> "MotionCommand | None":
        """Get the motion command term from the environment."""
        try:
            env = getattr(self.env, "unwrapped", self.env)
            command_manager = getattr(env, "command_manager", None)
            if command_manager is None:
                return None
            return command_manager.get_term("motion")
        except Exception:
            return None

    def _compute_success_ratio(self) -> float | None:
        """Read EMA success ratio from the motion command term if available."""
        motion_term = self._get_motion_term()
        if motion_term is None:
            return None

        success_total = getattr(motion_term, "success_total_count", None)
        failed_total = getattr(motion_term, "failed_total_count", None)
        if success_total is None or failed_total is None:
            return None
        denom = float(success_total) + float(failed_total)
        if denom <= 0:
            return None
        return float(success_total) / denom

    def _get_frozen_files(self, motion_term: "MotionCommand") -> set[str]:
        """Get set of frozen/dropped file paths."""
        frozen: set[str] = set()
        if not hasattr(motion_term, "motion_file_paths"):
            return frozen
        if hasattr(motion_term, "file_freeze_remaining"):
            frozen_mask = motion_term.file_freeze_remaining > 0
            for idx in frozen_mask.nonzero(as_tuple=False).flatten().tolist():
                if idx < len(motion_term.motion_file_paths):
                    frozen.add(motion_term.motion_file_paths[idx])
        if hasattr(motion_term, "file_dropped"):
            for idx in motion_term.file_dropped.nonzero(as_tuple=False).flatten().tolist():
                if idx < len(motion_term.motion_file_paths):
                    frozen.add(motion_term.motion_file_paths[idx])
        return frozen

    def _run_curriculum_evaluation_subprocess(
        self,
        motion_term: "MotionCommand",
        current_level: int,
        eval_files: list[str],
    ) -> tuple[float, float, list[str]]:
        """
        Run curriculum evaluation in a separate Isaac Lab process.
        
        This launches curriculum_eval.py as a subprocess to get accurate mpjpe/mpjae
        without inference_mode tensor conflicts.
        
        Args:
            motion_term: The motion command term
            current_level: The curriculum level to evaluate
            eval_files: List of file paths to evaluate
            
        Returns:
            (mpjpe, mpjae, bad_files) from the subprocess evaluation.
            bad_files contains paths that produced inf/nan metrics.
        """
        if not eval_files:
            print("[Curriculum] No files to evaluate.")
            return float("inf"), float("inf"), []
        
        # Use the checkpoint from previous iteration (which was just saved)
        checkpoint_iter = self.current_learning_iteration - 1
        eval_checkpoint_path = os.path.join(self.log_dir, f"model_{checkpoint_iter}.pt")
        if not os.path.exists(eval_checkpoint_path):
            print(f"[Curriculum] Checkpoint not found: model_{checkpoint_iter}.pt")
            return float("inf"), float("inf"), []
        
        # Create temp file for eval files list
        eval_files_path = os.path.join(self.log_dir, "curriculum_eval_files.json")
        try:
            with open(eval_files_path, "w") as f:
                json.dump({"files": eval_files, "level": current_level}, f)
        except Exception as e:
            print(f"[Curriculum] Failed to save eval files list: {e}")
            return float("inf"), float("inf"), []
        
        # Output file for results
        output_file = os.path.join(self.log_dir, f"curriculum_eval_result_level{current_level}.json")
        
        # Get motion dir
        motion_dir = getattr(motion_term.cfg, "motion_dir", None)
        if motion_dir is None:
            print("[Curriculum] motion_dir not set, cannot run subprocess evaluation.")
            return float("inf"), float("inf"), []
        
        # Get task name
        task_name = self.task_name
        if task_name is None:
            task_name = "Tracking-Flat-G1-v0"
            print(f"[Curriculum] task_name not set, using default: {task_name}")
        
        # Get eval runs from config
        cfg = getattr(motion_term, "cfg", None)
        eval_runs = getattr(cfg, "curriculum_eval_runs", 5) if cfg else 5
        num_envs = getattr(cfg, "curriculum_eval_num_envs", 1000) if cfg else 1000
        eval_gpu_cfg = getattr(cfg, "curriculum_eval_gpu", "auto") if cfg else "auto"
        
        # Determine which physical GPU to use for evaluation
        # Use CUDA_VISIBLE_DEVICES to isolate the subprocess to a single GPU
        if eval_gpu_cfg == "auto":
            # Auto-select: use the last GPU from CUDA_VISIBLE_DEVICES
            visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible_gpus:
                gpu_list = [g.strip() for g in visible_gpus.split(",") if g.strip()]
                if len(gpu_list) >= 2:
                    eval_gpu_id = gpu_list[-1]  # Use last physical GPU
                else:
                    eval_gpu_id = gpu_list[0] if gpu_list else "7"
            else:
                eval_gpu_id = "7"  # Default to physical GPU 7 (last GPU)
        elif isinstance(eval_gpu_cfg, int):
            eval_gpu_id = str(eval_gpu_cfg)
        else:
            # Extract GPU ID from cuda:N format or use directly
            cfg_str = str(eval_gpu_cfg)
            eval_gpu_id = cfg_str.replace("cuda:", "") if cfg_str.startswith("cuda:") else cfg_str
        
        # Build subprocess command
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "scripts", "rsl_rl", "curriculum_eval.py"
        )
        
        # Fallback to relative path if absolute doesn't exist
        if not os.path.exists(script_path):
            script_path = "scripts/rsl_rl/curriculum_eval.py"
        
        cmd = [
            sys.executable,
            "-u",  # Unbuffered output
            script_path,
            "--task", task_name,
            "--curriculum_level", str(current_level),
            "--output_file", output_file,
            "--motion_folder", motion_dir,
            "--eval_files_json", eval_files_path,
            "--checkpoint_path", eval_checkpoint_path,
            "--num_envs", str(num_envs),
            "--eval_runs", str(eval_runs),
            "--headless",
        ]
        
        print(f"[Curriculum] Starting subprocess evaluation...")
        print(f"[Curriculum]   Level: {current_level}, Files: {len(eval_files)}")
        print(f"[Curriculum]   Eval runs per file: {eval_runs}")
        print(f"[Curriculum]   Using GPU: {eval_gpu_id} (via CUDA_VISIBLE_DEVICES)")
        
        # Log file for subprocess output
        subprocess_log = os.path.join(self.log_dir, f"curriculum_eval_log_level{current_level}.txt")
        
        # Set CUDA_VISIBLE_DEVICES to single GPU for subprocess
        # This isolates the subprocess to one GPU, so cuda:0 = physical GPU eval_gpu_id
        subprocess_env = os.environ.copy()
        subprocess_env["CUDA_VISIBLE_DEVICES"] = eval_gpu_id
        
        try:
            # Run subprocess with output to file (so we can see progress)
            print(f"[Curriculum]   Subprocess log: {subprocess_log}")
            print(f"[Curriculum]   This may take a while (starting new Isaac Lab instance on GPU {eval_gpu_id})...")
            
            with open(subprocess_log, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=3600,  # 1 hour timeout
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
                    env=subprocess_env,  # Use env with single GPU
                )
            
            # Read and print last part of log
            if os.path.exists(subprocess_log):
                with open(subprocess_log, "r") as f:
                    log_content = f.read()
                    if log_content:
                        print(f"[Curriculum] Subprocess log (last 1000 chars):")
                        print(log_content[-1000:])
            
            if result.returncode != 0:
                print(f"[Curriculum] Subprocess failed with code {result.returncode}")
                return float("inf"), float("inf"), []
            
            # Read results
            if not os.path.exists(output_file):
                print(f"[Curriculum] Output file not found: {output_file}")
                print(f"[Curriculum] Check log file for errors: {subprocess_log}")
                return float("inf"), float("inf"), []
            
            with open(output_file, "r") as f:
                eval_result = json.load(f)
            
            mpjpe = eval_result.get("mpjpe", float("inf"))
            mpjae = eval_result.get("mpjae", float("inf"))
            bad_files = eval_result.get("bad_files", [])
            
            print(f"[Curriculum] Subprocess evaluation complete!")
            print(f"[Curriculum]   mpjpe={mpjpe:.4f}, mpjae={mpjae:.4f}")
            if bad_files:
                print(f"[Curriculum]   {len(bad_files)} problematic files detected")
            
            return mpjpe, mpjae, bad_files
            
        except subprocess.TimeoutExpired:
            print("[Curriculum] Subprocess timed out after 1 hour.")
            print(f"[Curriculum] Check log file: {subprocess_log}")
            return float("inf"), float("inf"), []
        except Exception as e:
            print(f"[Curriculum] Subprocess error: {e}")
            import traceback
            traceback.print_exc()
            return float("inf"), float("inf"), []
        finally:
            # Don't cleanup - keep files for debugging
            pass

    def _run_curriculum_evaluation_if_needed(self, iteration: int) -> None:
        """Check and run curriculum evaluation if at checkpoint iteration."""
        motion_term = self._get_motion_term()
        if motion_term is None:
            return

        # Update iteration in motion term
        if hasattr(motion_term, "set_training_iteration"):
            motion_term.set_training_iteration(iteration)

        # Check if evaluation should run
        if not hasattr(motion_term, "should_run_curriculum_evaluation"):
            return
        if not motion_term.should_run_curriculum_evaluation():
            return

        # Get frozen files to exclude
        frozen_files = self._get_frozen_files(motion_term)
        current_level = motion_term.get_curriculum_level()
        
        # 准备日志
        log_lines: list[str] = []
        log_lines.append(f"[Curriculum] ===== Evaluation at iteration {iteration} =====")
        print(f"\n{log_lines[-1]}")
        
        # Get files for current level evaluation
        eval_files = motion_term.get_curriculum_files_for_evaluation(frozen_files, current_level)
        
        if not eval_files:
            msg = f"[Curriculum] No files available for level {current_level} evaluation."
            log_lines.append(msg)
            print(msg)
            self._write_curriculum_eval_log(log_lines, current_level)
            return
        
        msg = f"[Curriculum] Level {current_level}: {len(eval_files)} files to evaluate"
        log_lines.append(msg)
        print(msg)
        
        cfg = getattr(motion_term, "cfg", None)
        
        # Check if subprocess evaluation is enabled (default: True)
        use_subprocess = getattr(cfg, "curriculum_use_subprocess_eval", True) if cfg else True
        
        bad_files: list[str] = []
        success_ratio = 0.0
        
        if use_subprocess and self.log_dir:
            # Use subprocess to run evaluation in a separate Isaac Lab environment
            mpjpe, mpjae, bad_files = self._run_curriculum_evaluation_subprocess(
                motion_term=motion_term,
                current_level=current_level,
                eval_files=eval_files,
            )
            success_ratio = 1.0  # subprocess 模式不检查成功率
            log_lines.append("[Curriculum] Using subprocess evaluation.")
        else:
            # 使用内部 EMA 评估
            msg = "[Curriculum] Using EMA-based evaluation."
            log_lines.append(msg)
            print(msg)
            mpjpe, mpjae, success_ratio, min_success_ratio, min_success_file = self._run_curriculum_evaluation_from_ema(motion_term, eval_files, frozen_files)
        
        # 处理问题文件：将产生inf/nan的文件永久freeze
        if bad_files:
            msg = f"[Curriculum] Adding {len(bad_files)} problematic files to freeze_in.txt"
            log_lines.append(msg)
            print(msg)
            if hasattr(motion_term, "add_files_to_freeze_in"):
                motion_term.add_files_to_freeze_in(bad_files)
        
        # Get thresholds
        mpjpe_thr = getattr(cfg, "curriculum_mpjpe_threshold", 0.12) if cfg else 0.12
        mpjae_thr = getattr(cfg, "curriculum_mpjae_threshold", 0.06) if cfg else 0.06
        success_thr = getattr(cfg, "curriculum_success_threshold", 0.1) if cfg else 0.1
        # Relaxed tolerance: allow the *other* metric to be slightly above its threshold.
        # User request: disable relax for level 1/2.
        tolerance = 0.0 if current_level <= 2 else 0.02
        
        # Log metrics
        if self.writer is not None and not getattr(self, "disable_logs", False):
            self.writer.add_scalar("Curriculum/eval_mpjpe", mpjpe, iteration)
            self.writer.add_scalar("Curriculum/eval_mpjae", mpjae, iteration)
            self.writer.add_scalar("Curriculum/eval_success_ratio", success_ratio, iteration)
        
        # 检查升级条件：移除 success 要求
        # 新条件：mpjpe 和 mpjae 其中一个满足，另一个在 +0.01 范围内
        # 即：(mpjpe < mpjpe_thr and mpjae < mpjae_thr + tolerance) or (mpjae < mpjae_thr and mpjpe < mpjpe_thr + tolerance)
        mpjpe_ok_strict = mpjpe < mpjpe_thr
        mpjae_ok_strict = mpjae < mpjae_thr
        mpjpe_ok_relaxed = mpjpe < mpjpe_thr + tolerance
        mpjae_ok_relaxed = mpjae < mpjae_thr + tolerance
        
        # 条件1: mpjpe 满足严格阈值，mpjae 满足宽松阈值
        condition1 = mpjpe_ok_strict and mpjae_ok_relaxed
        # 条件2: mpjae 满足严格阈值，mpjpe 满足宽松阈值
        condition2 = mpjae_ok_strict and mpjpe_ok_relaxed
        
        passed = condition1 or condition2

        # ---- 自动升级判定：per-level timeout + reward 收敛 ----
        level_start_iter = 0
        if hasattr(motion_term, "get_curriculum_level_unlocked_iter"):
            try:
                level_start_iter = int(motion_term.get_curriculum_level_unlocked_iter(current_level))
            except Exception:
                level_start_iter = 0
        elapsed_at_level = iteration - level_start_iter

        # (a) Per-level hard timeout
        per_level_list = getattr(cfg, "curriculum_auto_upgrade_per_level", None) if cfg else None
        if per_level_list and len(per_level_list) > 0:
            idx = min(current_level - 1, len(per_level_list) - 1)
            auto_after = max(int(per_level_list[max(0, idx)]), 0)
        else:
            auto_after = max(int(getattr(cfg, "curriculum_auto_upgrade_after_iters", 0) or 0), 0) if cfg else 0
        timeout_due = (not passed) and auto_after > 0 and elapsed_at_level >= auto_after

        # (b) Eval-metric 收敛提前升级（mpjpe/mpjae 连续 N 次 eval 不再改善）
        convergence_due = False
        self._eval_metric_history.append((mpjpe, mpjae))
        conv_result = self._check_metric_convergence()
        is_conv, imp_mpjpe, imp_mpjae, first_mpjpe, first_mpjae, last_mpjpe, last_mpjae, window_size = conv_result
        if (not passed) and self._metric_conv_enabled and elapsed_at_level >= self._metric_conv_min_iters:
            convergence_due = is_conv

        force_due = timeout_due or convergence_due
        
        # 打印状态
        mpjpe_mark = "✓" if mpjpe_ok_strict else ("~" if mpjpe_ok_relaxed else "✗")
        mpjae_mark = "✓" if mpjae_ok_strict else ("~" if mpjae_ok_relaxed else "✗")
        
        log_lines.append(f"[Curriculum] Thresholds check:")
        log_lines.append(f"[Curriculum]   success: {success_ratio:.4f} (ignored)")
        if min_success_file:
            log_lines.append(f"[Curriculum]   min success: {min_success_ratio:.4f} (file: {min_success_file})")
        log_lines.append(
            f"[Curriculum]   mpjpe:   {mpjpe:.4f} {mpjpe_mark} {mpjpe_thr} (relaxed: {mpjpe_thr + tolerance:.2f})"
        )
        log_lines.append(
            f"[Curriculum]   mpjae:   {mpjae:.4f} {mpjae_mark} {mpjae_thr} (relaxed: {mpjae_thr + tolerance:.2f})"
        )
        for line in log_lines[-5:]:
            print(line)
        
        # 打印收敛检测详情（如果窗口足够）
        if window_size > 0:
            thr_pct = self._metric_conv_threshold * 100
            imp_mpjpe_pct = imp_mpjpe * 100
            imp_mpjae_pct = imp_mpjae * 100
            conv_status = "CONVERGED" if is_conv else "still improving"
            log_lines.append(f"[Curriculum] Convergence check (window={window_size}):")
            log_lines.append(
                f"[Curriculum]   mpjpe: {first_mpjpe:.4f} -> {last_mpjpe:.4f}, "
                f"improvement={imp_mpjpe_pct:.2f}%, threshold={thr_pct:.1f}%"
            )
            log_lines.append(
                f"[Curriculum]   mpjae: {first_mpjae:.4f} -> {last_mpjae:.4f}, "
                f"improvement={imp_mpjae_pct:.2f}%, threshold={thr_pct:.1f}%"
            )
            log_lines.append(f"[Curriculum]   -> {conv_status}")
            for line in log_lines[-4:]:
                print(line)
        
        if passed or force_due:
            if force_due and not passed:
                reason = "reward_converged" if convergence_due else f"timeout({auto_after})"
                msg = (
                    f"[Curriculum] ⚠ Auto-upgrade triggered: level {current_level} "
                    f"not passed for {elapsed_at_level} iters (reason: {reason})"
                )
                log_lines.append(msg)
                print(msg)
            # 升级（真正是否升级由 CurriculumManager 决定）
            unlocked = motion_term.report_curriculum_evaluation(mpjpe, mpjae, iteration)
            if unlocked:
                new_level = motion_term.get_curriculum_level()
                msg = f"[Curriculum] ✓ Level {new_level} UNLOCKED!"
                log_lines.append(msg)
                print(msg)
                if self.writer is not None and not getattr(self, "disable_logs", False):
                    self.writer.add_scalar("Curriculum/level", new_level, iteration)
            else:
                # 达到最大等级或 curriculum 未启用等情况
                msg = f"[Curriculum] Level {current_level} upgrade check finished (no unlock)."
                log_lines.append(msg)
                print(msg)
        else:
            reason = []
            if not condition1 and not condition2:
                if not mpjpe_ok_strict and not mpjae_ok_strict:
                    reason.append(f"mpjpe {mpjpe:.4f} >= {mpjpe_thr} AND mpjae {mpjae:.4f} >= {mpjae_thr}")
                elif not mpjpe_ok_strict:
                    reason.append(f"mpjpe {mpjpe:.4f} >= {mpjpe_thr} AND mpjae {mpjae:.4f} >= {mpjae_thr + tolerance:.2f}")
                elif not mpjae_ok_strict:
                    reason.append(f"mpjae {mpjae:.4f} >= {mpjae_thr} AND mpjpe {mpjpe:.4f} >= {mpjpe_thr + tolerance:.2f}")
            msg = f"[Curriculum] ✗ Level {current_level} NOT passed: {', '.join(reason) if reason else 'conditions not met'}"
            log_lines.append(msg)
            print(msg)
        
        log_lines.append("[Curriculum] ==========================================")
        print(f"{log_lines[-1]}\n")
        
        # 写入日志文件
        self._write_curriculum_eval_log(log_lines, current_level)

    def _write_curriculum_eval_log(self, log_lines: list[str], level: int) -> None:
        """写入课程评估日志到文件。"""
        if not self.log_dir:
            return
        log_path = os.path.join(self.log_dir, f"curriculum_eval_log_level{level}.txt")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                for line in log_lines:
                    f.write(line + "\n")
                f.write("\n")  # 空行分隔每次评估
        except Exception as e:
            print(f"[Curriculum] Failed to write log: {e}")

    def _run_curriculum_evaluation_from_ema(
        self,
        motion_term: "MotionCommand",
        eval_files: list[str],
        frozen_files: set[str],
    ) -> tuple[float, float, float, float, str]:
        """
        基于训练过程中收集的 EMA 统计数据进行内部评估。
        
        返回 (mpjpe, mpjae, success_ratio, min_success_ratio, min_success_file)。
        升级条件在 report_curriculum_evaluation_ema 中检查：
        成功率 >= threshold AND (mpjpe < threshold OR mpjae < threshold)
        """
        eval_files = [f for f in eval_files if f not in frozen_files]
        if not eval_files:
            return float("inf"), float("inf"), 0.0, 0.0, ""
        
        file_indices = []
        valid_eval_files = []  # 保存成功找到索引的文件路径
        for file_path in eval_files:
            try:
                idx = motion_term.motion_file_paths.index(file_path)
                file_indices.append(idx)
                valid_eval_files.append(file_path)
            except ValueError:
                continue
        
        if not file_indices:
            return float("inf"), float("inf"), 0.0, 0.0, ""
        
        file_indices_tensor = torch.tensor(file_indices, dtype=torch.long, device=motion_term.device)
        
        # 获取 mpjpe 和 mpjae 的 EMA（优先使用单独追踪的版本）
        if hasattr(motion_term, "file_mpjpe_ema") and hasattr(motion_term, "file_mpjae_ema"):
            mpjpe_values = motion_term.file_mpjpe_ema[file_indices_tensor]
            mpjae_values = motion_term.file_mpjae_ema[file_indices_tensor]
        elif hasattr(motion_term, "file_error_ema"):
            # 兼容：使用旧的 file_error_ema
            errors = motion_term.file_error_ema[file_indices_tensor]
            mpjpe_values = errors
            mpjae_values = errors
        else:
            return float("inf"), float("inf"), 0.0, 0.0, ""
        
        # 计算成功率
        success_ratio_per_file = None
        mean_success_ratio = 0.0
        min_success_ratio = 0.0
        min_success_file = ""
        if hasattr(motion_term, "file_success_count") and hasattr(motion_term, "file_failed_count"):
            success = motion_term.file_success_count[file_indices_tensor]
            failed = motion_term.file_failed_count[file_indices_tensor]
            denom = success + failed + 1e-6
            success_ratio_per_file = success / denom
            mean_success_ratio = float(success_ratio_per_file.mean().item())
            
            # 找到 success 最低的文件
            min_idx = int(success_ratio_per_file.argmin().item())
            min_success_ratio = float(success_ratio_per_file[min_idx].item())
            if min_idx < len(valid_eval_files):
                min_success_file = valid_eval_files[min_idx]
                # 只显示文件名，不显示完整路径
                import os
                min_success_file = os.path.basename(min_success_file)
        
        # 过滤有效的 mpjpe/mpjae 值
        valid_mask = (mpjpe_values > 1e-6) & (mpjae_values > 1e-6)
        
        if not valid_mask.any():
            print(f"[Curriculum EMA] No valid files found")
            return float("inf"), float("inf"), mean_success_ratio, min_success_ratio, min_success_file
        
        # 计算有效文件的平均值
        mean_mpjpe = float(mpjpe_values[valid_mask].mean().item())
        mean_mpjae = float(mpjae_values[valid_mask].mean().item())
        valid_count = int(valid_mask.sum().item())
        total_count = len(file_indices)
        
        # 打印详细信息
        print(f"[Curriculum EMA] Evaluated {valid_count}/{total_count} files")
        print(f"[Curriculum EMA]   Mean success ratio: {mean_success_ratio:.4f}")
        if min_success_file:
            print(f"[Curriculum EMA]   Min success ratio: {min_success_ratio:.4f} (file: {min_success_file})")
        print(f"[Curriculum EMA]   Mean mpjpe: {mean_mpjpe:.4f}")
        print(f"[Curriculum EMA]   Mean mpjae: {mean_mpjae:.4f}")
        
        # 写入日志文件
        if self.log_dir:
            log_path = os.path.join(self.log_dir, "curriculum_ema_eval.txt")
            try:
                with open(log_path, "a") as f:
                    f.write(f"iter={self.current_learning_iteration} ")
                    f.write(f"files={valid_count}/{total_count} ")
                    f.write(f"success={mean_success_ratio:.4f} ")
                    f.write(f"mpjpe={mean_mpjpe:.4f} ")
                    f.write(f"mpjae={mean_mpjae:.4f}\n")
            except Exception:
                pass
        
        return mean_mpjpe, mean_mpjae, mean_success_ratio, min_success_ratio, min_success_file

    def _check_metric_convergence(self) -> tuple[bool, float, float, float, float, float, float, int]:
        """Check if eval metrics (mpjpe/mpjae) have converged.

        Looks at the last ``N`` eval results.  If the relative improvement
        from the first to the last entry is below the threshold for **both**
        mpjpe and mpjae, the level is considered converged.

        Returns:
            (is_converged, imp_mpjpe, imp_mpjae, first_mpjpe, first_mpjae, last_mpjpe, last_mpjae, window_size)
            If not enough data, returns (False, nan, nan, nan, nan, nan, nan, 0).
        """
        N = self._metric_conv_evals
        if len(self._eval_metric_history) < N + 1:
            nan = float("nan")
            return (False, nan, nan, nan, nan, nan, nan, 0)

        # Compare oldest vs newest in the last N+1 entries
        window = self._eval_metric_history[-(N + 1):]
        first_mpjpe, first_mpjae = window[0]
        last_mpjpe, last_mpjae = window[-1]

        # Relative improvement (positive = got better = lower error)
        imp_mpjpe = (first_mpjpe - last_mpjpe) / max(first_mpjpe, 1e-4)
        imp_mpjae = (first_mpjae - last_mpjae) / max(first_mpjae, 1e-4)

        # Converged if BOTH metrics improved less than threshold over N evals
        is_converged = imp_mpjpe < self._metric_conv_threshold and imp_mpjae < self._metric_conv_threshold
        return (is_converged, imp_mpjpe, imp_mpjae, first_mpjpe, first_mpjae, last_mpjpe, last_mpjae, N)

    def _reset_params_on_level_change(self, new_level: int, iteration: int) -> None:
        """当 curriculum level 变化时，重置/提升训练参数以加速学习新动作。"""
        if new_level <= self._last_curriculum_level:
            return  # 只在升级时调整参数
        
        motion_term = self._get_motion_term()
        cfg = getattr(motion_term, "cfg", None) if motion_term else None
        
        # 从配置中读取参数
        lr_reset_enabled = getattr(cfg, "curriculum_lr_reset_enabled", True) if cfg else True
        
        if not lr_reset_enabled:
            self._last_curriculum_level = new_level
            return
        # NOTE: We intentionally do NOT boost desired_kl / entropy_coef anymore.
        # Only learning rate reset is kept (per user request).
        self._level_up_iteration = None
        self._param_decay_iters = 0
        
        # 1. 重置学习率到初始值
        old_lr = self.alg.learning_rate
        new_lr = self._initial_learning_rate
        self.alg.learning_rate = new_lr
        for param_group in self.alg.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # 2. Keep desired_kl at initial value (no boost)
        old_kl = self.alg.desired_kl
        new_kl = float(self._initial_desired_kl)
        self.alg.desired_kl = new_kl
        
        # 3. Keep entropy_coef at initial value (no boost)
        old_entropy = self.alg.entropy_coef
        new_entropy = float(self._initial_entropy_coef)
        self.alg.entropy_coef = new_entropy
        
        print(f"\n[Curriculum Param Boost] Level {self._last_curriculum_level} -> {new_level}")
        print(f"[Curriculum Param Boost] learning_rate: {old_lr:.6f} -> {new_lr:.6f} (reset to initial)")
        print(f"[Curriculum Param Boost] desired_kl:    {old_kl:.4f} -> {new_kl:.4f} (no boost)")
        print(f"[Curriculum Param Boost] entropy_coef:  {old_entropy:.4f} -> {new_entropy:.4f} (no boost)")
        
        # 记录到 wandb
        if self.writer is not None and not getattr(self, "disable_logs", False):
            self.writer.add_scalar("Curriculum/lr_boosted", new_lr, iteration)
            self.writer.add_scalar("Curriculum/kl_boosted", new_kl, iteration)
            self.writer.add_scalar("Curriculum/entropy_boosted", new_entropy, iteration)
        
        self._last_curriculum_level = new_level

        # Reset eval metric history for convergence detection at new level
        self._eval_metric_history.clear()

        # ---- MoE: sync expert mask and routing mode on level change ----
        if self._is_moe:
            policy = self.alg.policy
            policy.set_max_active_experts(new_level)
            policy.set_current_max_level(new_level)
            
            # Expert warm start: copy previous expert weights to new expert
            if new_level > 1 and self._moe_expert_warm_start:
                policy.copy_expert_weights(from_idx=new_level - 2, to_idx=new_level - 1)
            
            # Invalidate file-level cache after file list rebuild
            mt = self._get_motion_term()
            if mt is not None and hasattr(mt, "invalidate_file_level_cache"):
                mt.invalidate_file_level_cache()
            # Switch to adaptive routing when configured level is reached
            cfg_mt = getattr(mt, "cfg", None) if mt else None
            max_sup = int(getattr(cfg_mt, "moe_supervised_until_level", 10) if cfg_mt else 10)
            if new_level >= max_sup and policy._routing_mode == "supervised":
                policy.set_routing_mode("adaptive")
                if self.writer is not None and not getattr(self, "disable_logs", False):
                    self.writer.add_scalar("MoE/routing_mode_adaptive", 1, iteration)

    def _decay_boosted_params(self, iteration: int) -> None:
        """逐渐将 desired_kl 和 entropy_coef 恢复到初始值。"""
        # No-op: we no longer boost these params on level-up.
        if self._level_up_iteration is None or self._param_decay_iters <= 0:
            return
        
        elapsed = iteration - self._level_up_iteration
        if elapsed < 0:
            return
        
        if elapsed >= self._param_decay_iters:
            # 恢复完成
            self.alg.desired_kl = self._initial_desired_kl
            self.alg.entropy_coef = self._initial_entropy_coef
            self._level_up_iteration = None  # 标记恢复完成
            return
        
        # 线性插值恢复
        progress = elapsed / self._param_decay_iters  # 0 -> 1
        
        # Keep at initial (safety).
        boosted_kl = float(self._initial_desired_kl)
        boosted_entropy = float(self._initial_entropy_coef)
        
        self.alg.desired_kl = boosted_kl + (self._initial_desired_kl - boosted_kl) * progress
        self.alg.entropy_coef = boosted_entropy + (self._initial_entropy_coef - boosted_entropy) * progress

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Extend default logging with curriculum evaluation and success ratio for wandb."""
        iteration = locs.get("it", 0)

        # Run curriculum evaluation if needed (before logging)
        self._run_curriculum_evaluation_if_needed(iteration)

        # Phase B: check difficulty and add expert / boost LR if needed
        if self._phase_b_enabled and iteration > 0:
            if hasattr(self.alg, "maybe_restore_boost"):
                self.alg.maybe_restore_boost(iteration)
            if iteration % self._phase_b_check_interval == 0:
                self._check_and_add_expert(iteration)

        # Log curriculum level
        motion_term = self._get_motion_term()
        if motion_term is not None and self.writer is not None and not getattr(self, "disable_logs", False):
            if hasattr(motion_term, "get_curriculum_level"):
                level = motion_term.get_curriculum_level()
                if level > 0:  # Only log if curriculum is active
                    self.writer.add_scalar("Curriculum/level", level, iteration)
                    
                    # 检测 level 变化，重置/提升训练参数
                    if level > self._last_curriculum_level:
                        self._reset_params_on_level_change(level, iteration)
                    
                    # 逐渐恢复 boosted 参数到初始值
                    self._decay_boosted_params(iteration)
                    # Also log the effective sampling ratios per curriculum level (helps debug
                    # "reward slows down after level-up" due to distribution shift).
                    try:
                        curriculum = getattr(motion_term, "_curriculum", None)
                        cfg = getattr(motion_term, "cfg", None)
                        curriculum_enabled = bool(getattr(cfg, "curriculum_enabled", False)) if cfg is not None else False
                        if curriculum is not None and curriculum_enabled and hasattr(curriculum, "get_level_sampling_ratios"):
                            ratios = curriculum.get_level_sampling_ratios(curriculum.max_level)
                            for lvl, ratio in sorted(ratios.items()):
                                self.writer.add_scalar(
                                    f"Curriculum/sampling_ratio_level_{int(lvl)}", float(ratio), iteration
                                )
                            self.writer.add_scalar(
                                "Curriculum/num_unlocked_files",
                                float(getattr(curriculum, "get_total_unlocked_files", lambda: 0)()),
                                iteration,
                            )
                    except Exception:
                        # Never break training due to auxiliary logging.
                        pass

        # Log success ratio
        if self.writer is not None and not getattr(self, "disable_logs", False):
            success_ratio = self._compute_success_ratio()
            if success_ratio is not None:
                self.writer.add_scalar("Train/success", success_ratio, iteration)

        # ---- MoE metrics ----
        if self._is_moe and self.writer is not None and not getattr(self, "disable_logs", False):
            if hasattr(self.alg, "mean_moe_aux_loss"):
                self.writer.add_scalar("Loss/moe_aux", self.alg.mean_moe_aux_loss, iteration)
            policy = self.alg.policy
            self.writer.add_scalar("MoE/max_active_experts", policy._max_active_experts, iteration)
            self.writer.add_scalar(
                "MoE/routing_mode", 0 if policy._routing_mode == "supervised" else 1, iteration
            )
            self.writer.add_scalar("MoE/num_experts", policy._num_experts, iteration)

            # Routing behavior diagnostics
            if policy._last_routing_entropy is not None:
                self.writer.add_scalar(
                    "MoE/routing_entropy", float(policy._last_routing_entropy.mean().item()), iteration
                )
            if policy._last_routing_gap is not None:
                self.writer.add_scalar(
                    "MoE/top1_top2_gap", float(policy._last_routing_gap.mean().item()), iteration
                )

            # Phase B per-file difficulty diagnostics
            if self._phase_b_enabled and hasattr(self.alg, "mean_difficulty_error"):
                self.writer.add_scalar("PhaseB/file_error_ema", self.alg.mean_difficulty_error, iteration)
                self.writer.add_scalar("PhaseB/file_entropy_ema", self.alg.mean_difficulty_entropy, iteration)
                self.writer.add_scalar("PhaseB/file_gap_ema", self.alg.mean_difficulty_gap, iteration)
                self.writer.add_scalar("PhaseB/difficult_file_ratio", self.alg.difficult_file_ratio, iteration)
                self.writer.add_scalar("PhaseB/lr_boosted", 1 if self.alg._boost_active else 0, iteration)

                # Critic warmup / value clamp state
                if hasattr(self.alg, "_critic_warmup_iters") and self.alg._critic_warmup_iters > 0:
                    elapsed = self.alg._phase_b_current_iter - self.alg._phase_b_start_iter
                    in_warmup = elapsed < self.alg._critic_warmup_iters
                    self.writer.add_scalar("PhaseB/critic_warmup_active", 1 if in_warmup else 0, iteration)

                # Data mixing ratio
                mt = self._get_motion_term()
                if mt is not None and hasattr(mt, "_phase_b_old_data_ratio") and mt._phase_b_old_data_ratio > 0:
                    self.writer.add_scalar(
                        "PhaseB/old_data_ratio",
                        mt._get_phase_b_old_data_ratio(),
                        iteration,
                    )

        super().log(locs, width=width, pad=pad)


# MoEMotionOnPolicyRunner is kept as an alias for backward compatibility
MoEMotionOnPolicyRunner = MotionOnPolicyRunner
