"""Server-side preparation streamed over SSH for the waiting generation controller.

The local launcher provides CONTROLLER_B64, CONTROLLER_SHA256 and BOOTSTRAP_SHA256
as ASCII string variables. This script never invokes a training entrypoint.
"""
import ast
import base64
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

import numpy as np


ROOT = Path("/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029")
STAGE = ROOT / "repair_prior29_v2_20261105"
PICK = ROOT / "stage3_pick29_20261103"
SUPPORTED = ROOT / "stage3_supported_references_20261104"
PRIOR = ROOT / "stage2_prior29_20261103"
CONTROLLER = STAGE / "src/pipeline_controller_v2.py"
LAUNCH = STAGE / "generation_pipeline_v2_launch"
OUTPUT = STAGE / "new_references_v2"
CONFIG = STAGE / "config.json"
ENV_SPEC = PRIOR / "env_spec.json"
NATIVE_SPEC = ROOT / "stage3_supermimic29_20261103/runtime_bridge_spec.json"
POSTOPT_ENV_SOURCE = PICK / "src/stage3_postopt_queue.py"
PYTHON = "/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python"
EXPECTED_TRAIN_NAME = "grab_omni29_mdm_floor_l2_guided_v2"
TRAIN_CONFIG_SHA256 = "9947c67442a62d467faee3f543f9635099e40acb55ff3ccd8197100961e96be8"
EXPECTED_OLD_SOURCE_SHA256 = {
    "native_pick": "46473f4405e584161fd6c6be821b09a8b1a248786a6b6b8d563345576846c17f",
    "native_supported": "f00c298ea59961681f9b806da8551dd845dabf94504678b927af6aa61ee1297e",
    "postopt_env": "802fd215290ab19f0e713e55f099a5d57dac252a254ba8d06ba71a0a2ead1edf",
    "native_pick_scene_env": "5d71619407c16026e3b1f14c1862fa08ffeac2e2ffd58f2560026c3421652069",
    "native_runtime": "cd5a3ba05da81222dba366706fc71658980c278b370f66a59ee5625e0d02f4d9",
    "diffusion_runtime_spec": "0f48ab5fcdb215f308b8657a034efb8fde7e2e8eaa9deeb4599f06ff477523f5",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return Path(path).read_text(encoding="utf8")


def write(path, value):
    with Path(path).open("x", encoding="utf8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    actual = sha(path)
    if expected:
        require(actual == expected, f"unexpected bytes at {path}: {actual}")
    return {"path": str(path), "sha256": actual}


def postopt_environment():
    source = ast.parse(read(POSTOPT_ENV_SOURCE), filename=str(POSTOPT_ENV_SOURCE))
    assignment = next(node for node in source.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "env" for target in node.targets))
    # Evaluate only the already bound historical environment assignment. Feed
    # an empty parent environ so secrets are never copied into the plan.
    namespace = {"os": types.SimpleNamespace(environ={}), "R": ROOT, "S": PICK, "Path": Path}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(POSTOPT_ENV_SOURCE), "exec"), namespace)
    env = namespace["env"]
    require(env["CUDA_VISIBLE_DEVICES"] == "0" and env["JAX_PLATFORMS"] == "cuda,cpu"
            and env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
            and env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.05",
            "historical JAX allocation environment differed")
    require(all(Path(path).is_dir() for path in env["PYTHONPATH"].split(os.pathsep)),
            "historical PyRoki module path missing")
    return env


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cpu_fixture(controller, plan):
    snapshot = {"own_gpu_bytes": controller.LIMITS["job_gpu"], "own_rss_bytes": controller.LIMITS["job_rss"],
                "project_gpu_bytes": controller.LIMITS["aggregate_gpu"], "project_rss_bytes": controller.LIMITS["aggregate_rss"],
                "free_bytes": controller.FLOOR + 64 * 1024**2}
    controller.resource_guard(snapshot, 64 * 1024**2)
    for key in snapshot:
        failure = dict(snapshot)
        failure[key] += -1 if key == "free_bytes" else 1
        try:
            controller.resource_guard(failure, 64 * 1024**2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"guard accepted invalid {key}")

    with tempfile.TemporaryDirectory(prefix="cpu_fixture_", dir=LAUNCH) as directory:
        root = Path(directory).resolve(strict=True)
        require(root.parent == LAUNCH and root.name.startswith("cpu_fixture_"), "fixture path escaped the launch directory")
        sizes = {}
        for case in ("quality_then_pass", "eight_quality_rejects", "runtime_error"):
            folder = root / case
            folder.mkdir()
            tried = Counter()

            def fake_add(task, count):
                require(count == 1, "pilot must generate one candidate at a time")
                tried[task] += 1
                if case == "runtime_error":
                    raise RuntimeError("synthetic API failure")
                passed = case == "quality_then_pass" and tried[task] == 2
                return {"native_pass": int(passed), "kinematic_pass": int(passed),
                        "records": [{"candidate_id": tried[task], "initial_noise_seed": 100 + tried[task],
                                     "rejection_reason": None if passed else "fixture_quality_reject"}]}

            error = None
            try:
                controller.run_pilots(fake_add, "a" * 64, folder)
            except (RuntimeError, ValueError) as exc:
                error = exc
            receipt = json.loads(read(folder / "pilot_receipt.json"))
            if case == "quality_then_pass":
                require(error is None and receipt["all_four_native_passed"] and
                        dict(tried) == dict.fromkeys(controller.TASKS, 2), "bounded quality retry did not pass")
            elif case == "eight_quality_rejects":
                require(isinstance(error, ValueError) and receipt["status"] == "pilot_quality_budget_exhausted"
                        and dict(tried) == {"pick": 8}, "quality cap did not stop")
            else:
                require(isinstance(error, RuntimeError) and dict(tried) == {"pick": 1},
                        "runtime error retried unexpectedly")
            sizes[case] = dict(tried)

        scene_provider = load_module(Path(plan["sources"]["task_conditions"]["path"]), "_deployment_scene_witness")
        records = []
        for index, task in enumerate(controller.TASKS, 1):
            folder = root / task
            folder.mkdir()
            raw = folder / "sample.npz"
            raw.write_bytes(f"synthetic interface fixture for {task}".encode())
            scene = None if task == "pick" else scene_provider.build(
                task, [index], 20261105, np.zeros((22, 3)), np.repeat(np.eye(3)[None], 22, axis=0)
            )["scene"]
            scene_sha = controller.json_digest(scene)
            object_pos = [.35, -.2, .9]
            row = {"id": f"{task}:{index}", "task": task, "candidate_id": index,
                   "source_sample": str(raw), "source_sample_sha256": sha(raw), "scene_sha256": scene_sha,
                   "grasp_frame": 47, "grasp_time_s": 2.35, "task_object_position": object_pos,
                   "initial_noise_sha256": f"{index:064x}", "feature_content_sha256": f"{100+index:064x}"}
            post = folder / "postopt"
            candidate = post / f"candidate_{index:08d}"
            candidate.mkdir(parents=True)
            robot = candidate / "robot29.npz"
            n = controller.POLICY[task]
            np.savez(robot, q=np.full((n, 29), index), root_pos=np.zeros((n, 3)),
                     root_quat=np.tile([1, 0, 0, 0], (n, 1)), hand_closed=np.zeros((n, 2)),
                     fps=20., task_object_position=object_pos, task_object_quaternion_wxyz=[1, 0, 0, 0])
            write(candidate / "report.json", {"task": task, "candidate_id": index,
                  "source_sample_sha256": sha(raw), "passed_requested_kinematic_checks": True,
                  "status": "kinematic_candidate_ready_for_native_probe", "output": str(robot),
                  "output_sha256": sha(robot), "scene_sha256": scene_sha})
            staged = controller.stage_native(plan, task, post, folder / "native_input", [row])
            native_input = staged[0][1]
            require((native_input / "robot29.npz").samefile(robot), "native robot adapter did not use a hardlink")
            if scene is not None:
                require((native_input / "raw/sample.npz").samefile(raw)
                        and json.loads(read(native_input / "raw/report.json"))["scene"] == scene,
                        "supported native adapter changed raw bytes or pinned scene")
            native = folder / "native"
            native.mkdir()
            write(native / "report.json", {"status": "native_replay_completed" if task == "pick" else "native_task_replay_completed",
                  "frames": n, "candidates": 1, "admitted_references": 1})
            write(native / f"candidate_{0 if task == 'pick' else index}_admission.json",
                  {"kinematic_report_sha256": sha(native_input / "report.json"), "scene_sha256": scene_sha,
                   "native_replay_frames": n, "native_checks": {"fixture_only": True},
                   "eligible_for_calibrated_scene_reference": True, "candidate_id": index, "task": task,
                   "source_state_sha256": row["source_sha256"], "task_object_position": object_pos,
                   "task_object_quaternion_wxyz": [1, 0, 0, 0]})
            controller.native_proofs(task, staged, native)
            require(row["admitted"] and row["grasp_frame"] == 47 and row["grasp_time_s"] == 2.35,
                    "native fixture proof or Pick grasp frame binding failed")
            records.append(row)
        controller.assert_no_duplicates(records)
    return {"status": "cpu_contract_fixture_passed", "synthetic_fixture_only": True,
            "gpu_or_physics_started": False, "tests": {"guard_boundaries": 5,
            "linux_native_input_hardlink_adapters": 4, "pilot_attempt_control": sizes}}


def main():
    require(Path(PYTHON).resolve() == Path(sys.executable).resolve(), "use the existing bound diffusion interpreter")
    require(STAGE.is_dir() and (STAGE / "src").is_dir() and not OUTPUT.exists(), "repair stage missing or generation already started")
    config_binding = bind(CONFIG, TRAIN_CONFIG_SHA256)
    config = json.loads(read(CONFIG))
    require(Path(config["stage_dir"]).resolve() == STAGE and config["run_name"] == EXPECTED_TRAIN_NAME,
            "configuration does not describe the authorized repaired run")
    training = STAGE / "runs" / EXPECTED_TRAIN_NAME
    status = training / "status.json"
    require(not (training / "failure.json").exists()
            and (not status.exists() or json.loads(read(status))["status"] != "completed"),
            "waiting bootstrap requires training pending or running and no failure")
    prior_env = json.loads(read(ENV_SPEC))
    require(prior_env["spec_hash"] == "74af067a" and prior_env["spec"]["python_path"] == PYTHON,
            "existing diffusion runtime differs from the bound environment")
    require(shutil_free(STAGE) >= 8 * 1024**3 + 456 * 1024**2,
            "insufficient disk reserve even for waiting-controller setup")

    payload = base64.b64decode(CONTROLLER_B64, validate=True)
    require(hashlib.sha256(payload).hexdigest() == CONTROLLER_SHA256, "controller payload changed in transit")
    ast.parse(payload, filename=str(CONTROLLER), feature_version=(3, 10))
    sources = {
        "controller": {"path": str(CONTROLLER), "sha256": CONTROLLER_SHA256},
        "sample": bind(STAGE / "src/sample.py", "9172eb10da06d4f7d388720d917b30b426544d8431a045fb918a6392cd92503c"),
        "postopt": bind(STAGE / "src/postopt_v2.py", "80ebeb2095e34b4e168eaa7e21b3eec62bc57725a5daf5f4c4e5524ab17dacbd"),
        "native_pick": bind(PICK / "src/stage3_pick500_replay.py", EXPECTED_OLD_SOURCE_SHA256["native_pick"]),
        "native_supported": bind(SUPPORTED / "src/stage3_supported_native_replay.py", EXPECTED_OLD_SOURCE_SHA256["native_supported"]),
        "task_conditions": bind(SUPPORTED / "src/stage3_supported_task_conditions.py",
                                "aa871aaa8a818c3146290835cef8eaa0a8d89fbc5730d80b160b2df6b8535ae0"),
        "historical_postopt_environment": bind(POSTOPT_ENV_SOURCE, EXPECTED_OLD_SOURCE_SHA256["postopt_env"]),
        "native_pick_scene_env": bind(PICK / "src/stage3_pick29_env.py", EXPECTED_OLD_SOURCE_SHA256["native_pick_scene_env"]),
        "native_pick_scene_core": bind(PICK / "src/stage3_pick29_core.py",
                                      "b6e16d820b399bf70022c9266bca67879f60cf4086f696aa78873ead758bf0a4"),
        "runtime": bind(STAGE / "src/runtime.py", "29290e2eab1796e30857f496f5f1bd604b9c0931becdc75a84590d1a6f2c6347"),
        "guidance": bind(STAGE / "src/guidance.py", "f5ec0ed61eb8bd89e090407c23bf43acdf7f8c570d0c07efb0c317ce45e3c344"),
        "contracts": bind(STAGE / "src/contracts.py", "a545af279821c5e093c5d8cc423ad65820906c184184bd9a68dc66b360f14462"),
        "validation": bind(STAGE / "src/validation.py", "26d953c60223babe72dd5121f26a59341a2a7b1154574e684865cea7ffc43ad5"),
        "conditions": bind(STAGE / "src/conditions.py", "d6c4e931d4529781890a98c33046f34d02b42e8f8f5c24bf585c6d12914dc567"),
        "train": bind(STAGE / "src/train.py", "539e789beecadf8a3c2980b0928bda71705182f245a19c54d57313be342be338"),
        "diffusion_runtime_spec": bind(ENV_SPEC, EXPECTED_OLD_SOURCE_SHA256["diffusion_runtime_spec"]),
    }
    trajgen = Path(json.loads(read(PRIOR / "config.json"))["trajgen"])
    urdf = trajgen.parent / "Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf"
    plan = {
        "schema": "repair_prior29_pipeline_plan_v2", "training_config": config_binding, "output_dir": str(OUTPUT),
        "sources": sources, "urdf": bind(urdf, "572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95"),
        "fk": bind(ROOT / "training_disk_guard_20261101/g1_fk.py",
                   "b7ef8d99a6b42120edb58cbd04641c2b935c91b4c7982b61636db4e41e22a77d"),
        "native_runtime": bind(NATIVE_SPEC, EXPECTED_OLD_SOURCE_SHA256["native_runtime"]),
        "native_device": "cuda:1", "sample_python": PYTHON, "postopt_python": PYTHON,
        "sample_env": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                       "PYTHONPATH": os.pathsep.join(prior_env["spec"]["module_paths"])},
        "postopt_env": postopt_environment(), "cohort_size": 32, "sample_batch_size": 2,
        "cohort_write_budget_bytes": 64 * 1024**2, "checkpoint_wait_seconds": 172800,
        "phase_timeout_seconds": {"sample": 86400, "postopt": 14400, "native": 900}, "seed": 20261105,
    }
    require(not (STAGE / ".generation_v2.lock").exists(), "generation controller lock already present")
    LAUNCH.mkdir(exist_ok=False)
    if CONTROLLER.exists():
        bind(CONTROLLER, CONTROLLER_SHA256)
    else:
        with CONTROLLER.open("xb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
    bind(CONTROLLER, CONTROLLER_SHA256)
    plan_path = LAUNCH / "config.json"
    write(plan_path, plan)
    plan_sha = sha(plan_path)
    sys.path.insert(0, str(CONTROLLER.parent))
    controller = load_module(CONTROLLER, "_bound_repair_generation_controller")
    controller.setup_plan(plan_path)
    write(LAUNCH / "cpu_fixture.json", cpu_fixture(controller, plan))
    require(not status.exists() or json.loads(read(status))["status"] != "completed",
            "training completed during setup; preserve configuration for an explicitly monitored launch")
    command = [PYTHON, "-B", "-u", str(CONTROLLER), "--plan", str(plan_path), "--plan-sha256", plan_sha]
    with (LAUNCH / "controller.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=str(STAGE), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"pid": process.pid, "config": str(plan_path), "config_sha256": plan_sha,
               "controller": str(CONTROLLER), "controller_sha256": CONTROLLER_SHA256,
               "bootstrap_sha256": BOOTSTRAP_SHA256, "cpu_fixture_sha256": sha(LAUNCH / "cpu_fixture.json"),
               "argv": command, "status_path": str(OUTPUT / "status.json"), "training_started": False}
    write(LAUNCH / "spawn.json", receipt)
    for _ in range(40):
        if process.poll() is not None:
            raise RuntimeError(f"waiting controller exited early with {process.returncode}; inspect controller.log")
        if (OUTPUT / "status.json").is_file():
            child_status = json.loads(read(OUTPUT / "status.json"))
            require(child_status["status"] == "waiting_for_formal_checkpoint" and child_status["pid"] == process.pid,
                    "new controller did not enter the authorized waiting state")
            receipt.update(status=child_status["status"], status_sha256=sha(OUTPUT / "status.json"))
            write(LAUNCH / "launch_receipt.json", receipt)
            print(json.dumps(receipt), flush=True)
            return
        time.sleep(.25)
    raise TimeoutError("waiting controller did not publish startup status; preserve the spawn record")


def shutil_free(path):
    import shutil
    return shutil.disk_usage(path).free


if __name__ == "__main__":
    main()

