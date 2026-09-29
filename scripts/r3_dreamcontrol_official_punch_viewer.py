"""Independent Isaac Sim WebRTC viewer for the official Punch checkpoint.

This is a checkpoint replay, not the live training environment. Any viewer
failure or client disconnect is handled here and cannot block PPO training.
"""

from __future__ import annotations

import fcntl
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "third_party/DreamControl/Training"
MON = ROOT / "artifacts/monitoring"
REF = ROOT / "third_party/DreamControl/TrajGen/sample/Punch_sim2"
TRAIN_LOG = MON / "r3_dreamcontrol_official_Punch_8192_resume1600_train.log"
STATE = MON / "r3_dreamcontrol_official_punch_viewer.state"
LOG = MON / "r3_dreamcontrol_official_punch_viewer.log"
HISTORY = MON / "r3_dreamcontrol_official_punch_viewer_history.log"
LOCK = MON / "r3_dreamcontrol_official_punch_viewer.lock"
FALLBACK = TRAIN / "logs/rsl_rl/g1/2026-09-27_07-24-06-Punch-a/model_1600.pt"
VIEWER_GPU = os.environ.get("R3_VIEWER_GPU", "1")
PORTABLE_ROOT = ROOT / "cache/r3_punch_viewer_kit"
STOP = False


def stop_requested(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def latest_checkpoint() -> Path:
    run = None
    if TRAIN_LOG.exists():
        matches = re.findall(
            r"Exact experiment name requested from command line: ([^\r\n]+)",
            TRAIN_LOG.read_text(errors="replace"),
        )
        if matches:
            run = TRAIN / "logs/rsl_rl/g1" / matches[-1].strip()
    if run and run.is_dir():
        models = []
        for path in run.glob("model_*.pt"):
            try:
                iteration = int(path.stem.split("_")[1])
                if path.stat().st_size > 0:
                    models.append((iteration, path))
            except (IndexError, ValueError, OSError):
                continue
        if models:
            return max(models)[1]
    return FALLBACK


def terminate_group(proc: subprocess.Popen[bytes]) -> None:
    # python.sh is a shell wrapper. It may exit before its Kit child, so
    # waiting on the wrapper alone must not be mistaken for full cleanup.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    time.sleep(2)
    try:
        # The group can outlive the python.sh wrapper.
        os.killpg(proc.pid, 0)
    except ProcessLookupError:
        return
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if proc.poll() is None:
            proc.wait(timeout=10)


def main() -> None:
    MON.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Checkpoint viewer already running")
        signal.signal(signal.SIGTERM, stop_requested)
        signal.signal(signal.SIGINT, stop_requested)
        if len(list(REF.glob("*.pkl"))) != 100:
            raise SystemExit("Expected exactly 100 official Punch references")
        env = os.environ.copy()
        env.update(
            PUBLIC_IP="10.147.17.32",
            WANDB_MODE="disabled",
            WANDB_SILENT="true",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            TMPDIR=str(ROOT / "cache/r3_punch_viewer_tmp"),
            XDG_CACHE_HOME=str(ROOT / "cache/r3_punch_viewer_home/.cache"),
            XDG_CONFIG_HOME=str(ROOT / "cache/r3_punch_viewer_home/.config"),
            XDG_DATA_HOME=str(ROOT / "cache/r3_punch_viewer_home/.local/share"),
            MPLCONFIGDIR=str(ROOT / "cache/r3_punch_viewer_home/.config/matplotlib"),
        )
        env.pop("CUDA_VISIBLE_DEVICES", None)
        PORTABLE_ROOT.mkdir(parents=True, exist_ok=True)
        env["PYTHONPATH"] = ":".join(
            str(path)
            for path in (
                ROOT / "envs/dreamcontrol-r3/site-packages",
                TRAIN / "source/isaaclab",
                TRAIN / "source/isaaclab_assets",
                TRAIN / "source/isaaclab_tasks",
                TRAIN / "source/isaaclab_rl",
                TRAIN / "isaac_utils",
                TRAIN / "scripts/reinforcement_learning/rsl_rl",
            )
        ) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        for key in ("TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "MPLCONFIGDIR"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)

        attempts = 0
        while not STOP:
            checkpoint = latest_checkpoint()
            if not checkpoint.is_file():
                STATE.write_text("WAITING_FOR_CHECKPOINT\n")
                time.sleep(10)
                continue
            attempts += 1
            command = [
                str(ROOT / "third_party/isaacsim-4.5.0/python.sh"),
                "-u",
                "scripts/reinforcement_learning/rsl_rl/play_eval.py",
                "--task=Isaac-Motion-Tracking-Punch-v0",
                f"--device=cuda:{VIEWER_GPU}",
                "--num_envs=16",
                f"--path={checkpoint}",
                "--livestream=2",
                f"--kit_args=--portable-root {PORTABLE_ROOT} --/renderer/multiGpu/enabled=false",
                "--id=r3_official_punch_checkpoint_viewer",
            ]
            with LOG.open("wb") as output:
                proc = subprocess.Popen(
                    command,
                    cwd=TRAIN,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                STATE.write_text(
                    f"STARTING checkpoint={checkpoint} gpu={VIEWER_GPU} envs=16 "
                    f"pid={proc.pid} attempt={attempts} mode=checkpoint_replay\n"
                )
                offset = 0
                started = time.monotonic()
                reason = "process_exit"
                ready = False
                ready_at = None
                while not STOP and proc.poll() is None:
                    time.sleep(10)
                    with LOG.open("rb") as current:
                        current.seek(offset)
                        new = current.read()
                        offset = current.tell()
                    # GLFW warnings are common while Kit initializes its
                    # headless WebRTC window; only a stalled stream is fatal.
                    if b"nvstPushStreamData timeout" in new:
                        reason = "stream_error"
                        break
                    if not ready and b"Loaded 100 motions" in new:
                        ready = True
                        ready_at = time.monotonic()
                        STATE.write_text(
                            f"READY checkpoint={checkpoint} gpu={VIEWER_GPU} envs=16 "
                            f"pid={proc.pid} attempt={attempts} mode=checkpoint_replay\n"
                        )
                    # A new portable Kit cache may need several minutes to
                    # compile RTX shaders on its first launch.
                    if not ready and time.monotonic() - started > 1200:
                        reason = "startup_timeout"
                        break
                    # Keep a usable stream up for at least five minutes after
                    # a slow first start, even if training saved newer models.
                    if ready and ready_at is not None and time.monotonic() - ready_at > 300 and latest_checkpoint() != checkpoint:
                        reason = "new_checkpoint"
                        break
                terminate_group(proc)
            STATE.write_text(
                f"RESTARTING reason={reason} last_checkpoint={checkpoint} "
                f"attempt={attempts} training_unaffected=true\n"
            )
            with HISTORY.open("a", encoding="utf-8") as history:
                history.write(
                    f"{datetime.now(timezone.utc).isoformat()} reason={reason} "
                    f"checkpoint={checkpoint} attempt={attempts}\n"
                )
            if time.monotonic() - started > 120:
                attempts = 0
            if not STOP:
                time.sleep(min(10 * attempts, 60) if reason != "new_checkpoint" else 2)
        STATE.write_text("STOPPED training_unaffected=true\n")


if __name__ == "__main__":
    main()
