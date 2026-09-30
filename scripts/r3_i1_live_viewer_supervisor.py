"""Restart an independent Isaac Sim viewer without touching PPO training."""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "third_party/DreamControl/Training"
MON = ROOT / "artifacts/monitoring"
STATE = MON / "r3_i1_live_viewer_supervisor.state"
VIEW_STATE = MON / "r3_i1_live_viewer.state"
LOG = MON / "r3_i1_live_viewer.log"
LOCK = MON / "r3_i1_live_viewer_supervisor.lock"
STOP = False


def stop(_signum, _frame):
    global STOP
    STOP = True


def terminate_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=15)
    # The launcher shell may exit on SIGTERM while Kit is still alive in the
    # same process group. Remove that remainder before starting another Kit.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def viewer_process_limits() -> None:
    # Limit renderer startup CPU workers so they do not crowd out PPO.
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(available[-min(6, len(available)):]))
    os.setpriority(os.PRIO_PROCESS, 0, 10)


def main() -> None:
    MON.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        env = os.environ.copy()
        env["DREAMCONTROL_KICK_REF_PATH"] = str(ROOT / "third_party/DreamControl/TrajGen/sample/Kick_sim2")
        env["PUBLIC_IP"] = "10.147.17.32"
        env["WANDB_MODE"] = "disabled"
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[key] = "2"
        env["TMPDIR"] = str(ROOT / "cache/r3_i1_live_viewer/tmp")
        env["XDG_CACHE_HOME"] = str(ROOT / "cache/r3_i1_live_viewer/xdg")
        env["MPLCONFIGDIR"] = str(ROOT / "cache/r3_i1_live_viewer/mpl")
        for key in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        env["PYTHONPATH"] = ":".join(str(p) for p in (
            ROOT / "scripts",
            ROOT / "envs/dreamcontrol-r3/site-packages",
            TRAIN / "source/isaaclab",
            TRAIN / "source/isaaclab_assets",
            TRAIN / "source/isaaclab_tasks",
            TRAIN / "source/isaaclab_rl",
            TRAIN / "isaac_utils",
            TRAIN / "scripts/reinforcement_learning/rsl_rl",
        )) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        command = [
            str(ROOT / "third_party/isaacsim-4.5.0/python.sh"), "-u",
            str(ROOT / "scripts/r3_i1_live_training_viewer.py"),
            "--device=cuda:0", "--livestream=2", "--telemetry-port=40123",
            f"--status-path={VIEW_STATE}",
            f"--snapshot-path={MON / 'r3_i1_live_frame.png'}",
            f"--kit_args=--portable-root {ROOT / 'cache/r3_i1_live_viewer/kit'} --/renderer/multiGpu/enabled=false",
        ]
        attempt = 0
        while not STOP:
            attempt += 1
            VIEW_STATE.write_text("STARTING mode=sampled_actual_training_pose\n")
            if LOG.exists():
                archive = MON / f"r3_i1_live_viewer_previous_{time.time_ns()}.log"
                archive.write_bytes(LOG.read_bytes())
            with LOG.open("wb") as output:
                run_started = time.time()
                proc = subprocess.Popen(command, cwd=TRAIN, env=env, stdout=output,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        preexec_fn=viewer_process_limits)
                STATE.write_text(f"RUNNING attempt={attempt} pid={proc.pid} gpu=0 training_unaffected=true\n")
                offset = 0
                last_ready = time.monotonic()
                while not STOP and proc.poll() is None:
                    time.sleep(10)
                    with LOG.open("rb") as current:
                        current.seek(offset)
                        data = current.read()
                        offset = current.tell()
                    # A disconnected client can emit stream timeouts even while
                    # rendering and PPO telemetry continue. Avoid killing a
                    # healthy renderer on a single transport warning.
                    if b"nvstPushStreamData timeout" in data:
                        STATE.write_text(f"RUNNING attempt={attempt} pid={proc.pid} stream_warning=true training_unaffected=true\n")
                    frame = MON / "r3_i1_live_frame.png"
                    if (frame.exists() and frame.stat().st_mtime >= run_started
                            and time.time() - frame.stat().st_mtime > 120):
                        STATE.write_text(f"RESTARTING reason=render_capture_stale attempt={attempt}\n")
                        break
                    if VIEW_STATE.exists() and VIEW_STATE.read_text().startswith(("READY", "LIVE", "STALE")):
                        last_ready = time.monotonic()
                    if time.monotonic() - last_ready > 1200:
                        STATE.write_text(f"RESTARTING reason=startup_timeout attempt={attempt}\n")
                        break
            terminate_group(proc)
            if not STOP:
                STATE.write_text(f"RESTARTING attempt={attempt} training_unaffected=true\n")
                time.sleep(min(10 * attempt, 60))
        STATE.write_text("STOPPED training_unaffected=true\n")


if __name__ == "__main__":
    main()
