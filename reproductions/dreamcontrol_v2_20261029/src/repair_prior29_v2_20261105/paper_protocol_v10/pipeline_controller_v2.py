"""Controller for the authorized repaired-reference generation and admission only."""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from contracts import TASK_POINTS, json_digest
from runtime import load_config, require, sha, verify_hash


TASKS = tuple(TASK_POINTS)
FLOOR = 8 * 1024**3 + 456 * 1024**2
LIMITS = {"job_gpu": 12 * 1024**3, "job_rss": 32 * 1024**3,
          "aggregate_gpu": 24 * 1024**3, "aggregate_rss": 64 * 1024**3}
POLICY = {"pick": 120, "pour": 180, "bimanual_box_lift": 180, "box_push_pull": 180}
TARGET = 500
CANDIDATE_CAP = 1000
PILOT_CAP = 8


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, data, *, exclusive=False):
    path = Path(path)
    temporary = path if exclusive else path.with_suffix(".tmp")
    with temporary.open("x" if exclusive else "w", encoding="utf8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    if not exclusive:
        temporary.replace(path)


def bound(spec):
    path = Path(spec["path"])
    require(path.is_absolute() and path.resolve(strict=True) == path, "bound paths must be absolute and resolved")
    verify_hash(path, spec["sha256"])
    return path


def import_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def setup_plan(path):
    plan = read(path)
    require(plan["schema"] == "repair_prior29_pipeline_plan_v2", "unrecognized generation plan schema")
    cfg_path = bound(plan["training_config"])
    cfg = load_config(cfg_path)
    stage = Path(cfg["stage_dir"])
    output = Path(plan["output_dir"]).resolve()
    require(output.is_relative_to(stage) and output != stage, "generation output must be a new child of the repaired stage")
    require(not output.is_relative_to(stage / "runs") and not output.is_relative_to(stage / "reports"),
            "generation output must not be a training artifact directory")
    required = {"controller", "sample", "postopt", "native_pick", "native_supported", "task_conditions"}
    require(required <= plan["sources"].keys(), "incomplete pipeline source binding")
    names = {"controller": "pipeline_controller_v2.py", "sample": "sample.py", "postopt": "postopt_v2.py",
             "native_pick": "native_pick.py", "native_supported": "native_supported.py",
             "task_conditions": "stage3_supported_task_conditions.py"}
    for key, name in names.items():
        require(bound(plan["sources"][key]).name == name, f"unexpected {key} entrypoint")
    require(bound(plan["sources"]["controller"]) == Path(__file__).resolve(), "different controller source in plan")
    require(bound(plan["sources"]["sample"]).parent == Path(__file__).resolve().parent
            and bound(plan["sources"]["postopt"]).parent == Path(__file__).resolve().parent,
            "repaired sampler and postoptimizer must share the bound source directory")
    for key in ("urdf", "fk", "native_runtime"):
        bound(plan[key])
    for key in ("sample_python", "postopt_python"):
        executable = Path(plan[key])
        require(executable.is_absolute() and executable.is_file(), f"missing {key}")
    require(type(plan["cohort_size"]) is int and 1 <= plan["cohort_size"] <= 32, "invalid cohort size")
    require(type(plan["sample_batch_size"]) is int and 1 <= plan["sample_batch_size"] <= cfg["max_sample_batch_size"],
            "sample batch size exceeds configured ceiling")
    require(plan["cohort_write_budget_bytes"] >= 32 * 1024**2, "cohort disk reservation is too small")
    durations = (plan["checkpoint_wait_seconds"],
                 *(plan["phase_timeout_seconds"][p] for p in ("sample", "postopt", "native")))
    require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in durations),
            "finite positive phase waits required")
    require(not any(key in plan for key in ("train_command", "policy_command", "ppo_command")),
            "generation plan cannot launch training")
    return plan, cfg, output


def process_table():
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,pgid=,sid=,rss=,args="], text=True, timeout=10)
    result = {}
    for line in output.splitlines():
        fields = line.strip().split(None, 5)
        if len(fields) == 6:
            pid, ppid, pgid, sid, rss = map(int, fields[:5])
            result[pid] = (ppid, pgid, sid, rss * 1024, fields[5])
    return result


def resource_snapshot(project_root, owned_group=None):
    table = process_table()
    own = {pid for pid, (_, pgid, sid, _, _) in table.items() if pgid == sid == owned_group}
    project = own | {pid for pid, (_, _, _, _, argv) in table.items() if str(project_root) + "/" in argv}
    while True:
        extended = project | {pid for pid, (parent, _, _, _, _) in table.items() if parent in project}
        if extended == project:
            break
        project = extended
    gpu = Counter()
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                   "--format=csv,noheader,nounits"], text=True, timeout=10)
    for line in raw.splitlines():
        if not line.strip():
            continue
        pid, amount = (field.strip() for field in line.split(","))
        if int(pid) in project:
            require(amount.isdigit(), "project GPU accounting is unavailable")
            gpu[int(pid)] += int(amount) * 1024**2
    return {
        "own_gpu_bytes": sum(gpu[pid] for pid in own), "own_rss_bytes": sum(table[pid][3] for pid in own),
        "project_gpu_bytes": sum(gpu[pid] for pid in project), "project_rss_bytes": sum(table[pid][3] for pid in project),
        "owned_pids": sorted(own), "free_bytes": shutil.disk_usage(project_root).free,
    }


def resource_guard(snapshot, reserve=0):
    require(snapshot["free_bytes"] >= FLOOR + reserve, "reference generation disk protection")
    for label, key in (("job_gpu", "own_gpu_bytes"), ("job_rss", "own_rss_bytes"),
                       ("aggregate_gpu", "project_gpu_bytes"), ("aggregate_rss", "project_rss_bytes")):
        require(snapshot[key] <= LIMITS[label], f"reference generation exceeded {label}")


def stop_group(child):
    # Child was created by this controller with a new session; no matching
    # by executable name and no signals to any existing training process.
    if child.poll() is not None:
        return

    def group_alive():
        try:
            os.killpg(child.pid, 0)
            return True
        except ProcessLookupError:
            return False

    if group_alive():
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 20
        while group_alive() and time.monotonic() < deadline:
            child.poll()
            time.sleep(.2)
        if group_alive():
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    child.wait()


def wait_for_checkpoint(cfg, plan, output):
    run = Path(cfg["stage_dir"]) / "runs" / cfg["run_name"]
    deadline = time.monotonic() + plan["checkpoint_wait_seconds"]
    while time.monotonic() < deadline:
        status_path = run / "status.json"
        if (run / "failure.json").exists():
            raise ValueError("repaired training has a preserved failure")
        if status_path.is_file():
            status = read(status_path)
            if status["status"] == "completed" and not (run / "training.lock").exists():
                receipt = status["last_checkpoint"]
                checkpoint = Path(receipt["path"]).resolve(strict=True)
                require(checkpoint == run / "step_000010000.pt" and status["cursor"]["step"] == cfg["max_steps"] == 10000,
                        "completed status must bind the formal repaired 10000-update checkpoint")
                proof_path = checkpoint.with_suffix(".validation.json")
                proof = read(proof_path)
                require(not checkpoint.with_suffix(".validation_failed.json").exists()
                        and proof["schema"] == "repair_prior29_exact_checkpoint_v1" and proof["all_fields_exact"] is True
                        and proof["checkpoint_sha256"] == receipt["sha256"]
                        and proof["checkpoint_bytes"] == receipt["bytes"] == checkpoint.stat().st_size
                        and proof["adam_steps"]["cursor_updates"] == proof["adam_steps"]["max_parameter_updates"] == 10000
                        and all(proof["compared_tensor_counts"][key] > 0 for key in ("model", "optimizer", "rng")),
                        "formal checkpoint complete readback or Adam evidence is missing")
                verify_hash(checkpoint, receipt["sha256"])
                save(output / "checkpoint_binding.json", {"path": str(checkpoint), "sha256": receipt["sha256"],
                     "validation_path": str(proof_path), "validation_sha256": sha(proof_path),
                     "training_status_sha256": sha(status_path), "adam_steps": proof["adam_steps"]}, exclusive=True)
                return checkpoint, receipt["sha256"]
        time.sleep(15)
    raise TimeoutError("bounded wait for completed repaired-prior checkpoint expired")


def phase(plan_path, plan_sha, plan, cfg, output, directory, label, command, env, cwd, terminal=None):
    verify_hash(plan_path, plan_sha)
    bound(plan["training_config"])
    for spec in plan["sources"].values():
        bound(spec)
    root = Path(cfg["project_root"])
    resource_guard(resource_snapshot(root), plan["cohort_write_budget_bytes"])
    with (directory / (label + ".log")).open("x", encoding="utf8") as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        began = time.monotonic()
        peak = Counter()
        cleanup = False
        completed_since = None
        try:
            save(directory / (label + "_launch.json"), {"pid": child.pid, "argv": command,
                 "cwd": str(cwd), "plan_sha256": plan_sha}, exclusive=True)
            while child.poll() is None:
                snapshot = resource_snapshot(root, child.pid)
                resource_guard(snapshot)
                for key in ("own_gpu_bytes", "own_rss_bytes", "project_gpu_bytes", "project_rss_bytes"):
                    peak[key] = max(peak[key], snapshot[key])
                save(output / "status.json", {"status": "phase_running", "phase": label,
                     "cohort": str(directory), "resources": snapshot, "peaks": dict(peak),
                     "elapsed_seconds": time.monotonic() - began, "ppo_started": False})
                if time.monotonic() - began > plan["phase_timeout_seconds"][label]:
                    raise TimeoutError(f"{label} exceeded its bounded phase duration")
                if terminal and terminal[0].is_file():
                    try:
                        finished = read(terminal[0]).get("status") == terminal[1]
                    except json.JSONDecodeError:
                        finished = False
                    if finished:
                        completed_since = completed_since or time.monotonic()
                        if time.monotonic() - completed_since > 30:
                            stop_group(child)
                            cleanup = True
                            break
                time.sleep(2)
        finally:
            stop_group(child)
            save(directory / (label + "_exit.json"), {"returncode": child.wait(), "peaks": dict(peak),
                 "completed_native_cleanup_released": cleanup}, exclusive=True)
    require(child.returncode == 0 or cleanup, f"{label} failed; see the preserved phase log")


def producer_records(producer, task, ids, checkpoint_sha):
    report = read(producer / "report.json")
    journal = producer / "records.jsonl"
    require(report["status"] == "sampling_completed_not_postoptimized" and report["task"] == task
            and report["checkpoint_sha256"] == checkpoint_sha and report["generated_candidates"] == len(ids),
            "sampler completion report disagrees with the cohort")
    verify_hash(journal, report["records_sha256"])
    rows = [json.loads(line) for line in journal.read_text(encoding="utf8").splitlines()]
    require([row["candidate_id"] for row in rows] == list(ids), "missing or misordered producer rows")
    records = []
    for row in rows:
        raw = Path(row["path"]).resolve(strict=True)
        require(raw.is_relative_to(producer) and row["task"] == task, "raw artifact outside this cohort")
        verify_hash(raw, row["sha256"])
        with np.load(raw, allow_pickle=False) as file:
            metadata = json.loads(file["metadata_utf8"].tobytes())
            features = file["features"]
            require(features.shape == (1, POLICY[task], 263) and np.isfinite(features).all(), "invalid raw feature shape")
            require(hashlib.sha256(features.tobytes()).hexdigest() == metadata["feature_content_sha256"]
                    == row["feature_content_sha256"], "raw feature bytes disagree")
        require(metadata["candidate_id"] == row["candidate_id"] and metadata["task"] == task
                and metadata["checkpoint_sha256"] == checkpoint_sha and metadata["fps"] == 20
                and metadata["initial_noise_sha256"] == row["initial_noise_sha256"], "raw metadata binding failed")
        records.append({
            "id": f"{task}:{row['candidate_id']}", "candidate_id": row["candidate_id"], "task": task,
            "caption": metadata["caption"], "caption_role": "actual_generation_condition", "retrieval_group_id": task,
            "source_sample": str(raw), "source_sample_sha256": row["sha256"], "source_frames": POLICY[task],
            "source_fps": 20, "initial_noise_sha256": row["initial_noise_sha256"],
            "feature_content_sha256": row["feature_content_sha256"], "checkpoint_sha256": checkpoint_sha,
            "scene_sha256": metadata["scene_sha256"], "grasp_frame": metadata["grasp_frame"],
            "grasp_time_s": metadata["grasp_time_s"], "initial_noise_seed": metadata["initial_noise_seed"],
            "task_object_position": metadata["task_object_position"],
            "coordinate_provenance": metadata["coordinate_provenance"],
        })
    return records


def postopt_worker(plan, manifest_path, output):
    records = read(manifest_path)
    require(bool(records), "postoptimizer worker requires a nonempty cohort")
    kind = "postopt"
    solver_module = import_file(bound(plan["sources"][kind]), "_repaired_postopt_bound")
    frames = records[0]["source_frames"]
    require(all(row["source_frames"] == frames for row in records), "mixed frame counts in postopt cohort")
    solver = solver_module.MotionPreservingSolverV2(bound(plan["urdf"]), frames, bound(plan["fk"]))
    output.mkdir(exist_ok=False)
    for row in records:
        directory = output / f"candidate_{row['candidate_id']:08d}"
        from verified_initialization_v10 import process
        report = process(plan, row, solver, solver_module, directory)
        require(report["status"] in ("generated_candidate_rejected", "kinematic_candidate_ready_for_native_probe"),
                f"postoptimization failed for candidate {row['candidate_id']}; inspect the preserved prefix")


def stage_native(plan, task, postopt_dir, stage, rows):
    scene = None
    if task != "pick":
        # The original builder owns the unchanged scene specification. Probe
        # its scene fields only, then check against the real samples' SHA.
        builder = import_file(bound(plan["sources"]["task_conditions"]), "_bound_scene_provider")
        stub = np.zeros((22, 3))
        matrices = np.repeat(np.eye(3)[None], 22, axis=0)
        scene = builder.build(task, [0], 0, stub, matrices)["scene"]
    staged = []
    for row in rows:
        src_dir = postopt_dir / f"candidate_{row['candidate_id']:08d}"
        report_path = src_dir / "report.json"
        report = read(report_path)
        require(report["source_sample_sha256"] == row["source_sample_sha256"]
                and report["candidate_id"] == row["candidate_id"] and report["task"] == task,
                "postopt source binding disagrees with raw candidate")
        row.update(kinematic_report=str(report_path), kinematic_report_sha256=sha(report_path),
                   postopt_status=report["status"], admitted=False, selected_reference=False)
        if not report["passed_requested_kinematic_checks"]:
            row["rejection_reason"] = report["rejection_reason"]
            continue
        robot = src_dir / "robot29.npz"
        verify_hash(robot, report["output_sha256"])
        require(Path(report["output"]).resolve(strict=True) == robot
                and report["scene_sha256"] == row["scene_sha256"], "postopt output path or scene mismatch")
        with np.load(robot, allow_pickle=False) as file:
            require(file["q"].shape == (POLICY[task], 29) and float(file["fps"]) == 20., "invalid postopt output shape")
            require(np.allclose(file["task_object_position"], row["task_object_position"], rtol=0., atol=1e-12),
                    "postopt object position changed")
            fingerprint = hashlib.sha256(b"".join(np.ascontiguousarray(file[key]).tobytes()
                                             for key in ("q", "root_pos", "root_quat", "hand_closed"))).hexdigest()
            object_quaternion = file["task_object_quaternion_wxyz"].tolist()
        row.update(source=str(robot), source_sha256=report["output_sha256"], state_fingerprint=fingerprint,
                   task_object_quaternion_wxyz=object_quaternion)
        directory = stage / f"candidate_{row['candidate_id']:08d}"
        directory.mkdir(parents=True, exist_ok=False)
        linked_robot = directory / "robot29.npz"
        os.link(robot, linked_robot)
        adapted = {**report, "output": str(linked_robot), "task_object_quaternion_wxyz": object_quaternion,
                   "original_kinematic_report_sha256": row["kinematic_report_sha256"],
                   "compatibility_schema": "read_only_native_input_adapter_v2"}
        if scene is not None:
            require(json_digest(scene) == row["scene_sha256"], "supported native scene hash differs from raw sample")
            raw_dir = directory / "raw"
            raw_dir.mkdir()
            linked_raw = raw_dir / "sample.npz"
            os.link(row["source_sample"], linked_raw)
            save(raw_dir / "report.json", {"scene": scene, "scene_sha256": row["scene_sha256"]}, exclusive=True)
            adapted.update(sample=str(linked_raw), sample_sha256=row["source_sample_sha256"])
        save(directory / "report.json", adapted, exclusive=True)
        row["native_compatibility_report_sha256"] = sha(directory / "report.json")
        staged.append((row, directory))
    return staged


def native_proofs(task, staged, native):
    report = read(native / "report.json")
    status = "native_replay_completed" if task == "pick" else "native_task_replay_completed"
    require(report["status"] == status and report["frames"] == POLICY[task] and report["candidates"] == len(staged),
            "native replay did not produce a complete cohort report")
    admitted = 0
    for index, (row, directory) in enumerate(staged):
        key = index if task == "pick" else row["candidate_id"]
        path = native / f"candidate_{key}_admission.json"
        proof = read(path)
        require(proof["kinematic_report_sha256"] == row["native_compatibility_report_sha256"]
                == sha(directory / "report.json") and proof["scene_sha256"] == row["scene_sha256"]
                and proof["native_replay_frames"] == POLICY[task] and bool(proof["native_checks"])
                and all(type(value) is bool for value in proof["native_checks"].values()),
                "native proof does not bind the actual staged candidate")
        if task == "pick":
            require(np.allclose(proof["task_object_position"], row["task_object_position"], rtol=0., atol=1e-12)
                    and np.allclose(proof["task_object_quaternion_wxyz"], row["task_object_quaternion_wxyz"],
                                    rtol=0., atol=1e-12), "Pick native proof object pose binding mismatch")
        else:
            require(proof["candidate_id"] == row["candidate_id"] and proof["task"] == task
                    and proof["source_state_sha256"] == row["source_sha256"], "native state binding mismatch")
        accepted = proof["eligible_for_calibrated_scene_reference"]
        require(type(accepted) is bool and accepted == all(proof["native_checks"].values()), "native decision mismatch")
        row.update(admitted=accepted, native_proof=str(path), native_proof_sha256=sha(path))
        if not accepted:
            row["rejection_reason"] = "native_geometry_rejection"
        admitted += accepted
    require(admitted == report["admitted_references"], "native cohort admission count mismatch")

def runtime_env(extra=None, *, remove_visibility=False):
    env = {**os.environ, **(extra or {}), "PYTHONDONTWRITEBYTECODE": "1"}
    if remove_visibility:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_cohort(plan_path, plan_sha, plan, cfg, output, task, number, offset, count,
               checkpoint, checkpoint_sha, native_spec):
    directory = output / task / f"cohort_{number:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    first_id = 20_000_000 + 100_000 * TASKS.index(task) + offset
    ids = range(first_id, first_id + count)
    producer = directory / "raw"
    postopt = directory / "postopt"
    native = directory / "native"
    runner = lambda label, argv, env, cwd, terminal=None: phase(
        plan_path, plan_sha, plan, cfg, output, directory, label, argv, env, cwd, terminal
    )
    sample_python = plan["sample_python"]
    source = plan["sources"]
    reuse = None
    if task == "pick" and count == 1 and offset == number and reuse is not None:
        report = bound(reuse)
        require(report.is_relative_to(Path(cfg["stage_dir"])) and report.parent.name == "raw",
                "unexpected reused sampling receipt")
        producer = report.parent
    else:
        runner("sample", [sample_python, "-B", "-u", str(bound(source["sample"])),
                           "--config", str(bound(plan["training_config"])), "--checkpoint", str(checkpoint),
                           "--checkpoint-sha256", checkpoint_sha, "--task", task, "--count", str(count),
                           "--start-index", str(first_id), "--output-dir", str(producer),
                           "--batch-size", str(min(plan["sample_batch_size"], count)), "--seed", str(plan.get("seed", 20261105))],
               runtime_env(plan.get("sample_env"), remove_visibility=True), cfg["trajgen"])
    rows = producer_records(producer, task, ids, checkpoint_sha)
    records = directory / "raw_bindings.json"
    save(records, rows, exclusive=True)
    runner("postopt", [plan["postopt_python"], "-B", "-u", str(Path(__file__).resolve()),
                        "--plan", str(plan_path), "--plan-sha256", plan_sha, "--postopt-worker",
                        "--worker-records", str(records), "--worker-records-sha256", sha(records),
                        "--worker-output", str(postopt)],
           runtime_env(plan.get("postopt_env")), str(output))
    staged = stage_native(plan, task, postopt, directory / "native_input", rows)
    if staged:
        native_key = "native_pick" if task == "pick" else "native_supported"
        terminal = "native_replay_completed" if task == "pick" else "native_task_replay_completed"
        script = bound(source[native_key])
        argv = [native_spec["python"], "-B", "-u", str(script), "--headless", "--device",
                plan["native_device"], "--output-dir", str(native)]
        if task != "pick":
            argv += ["--task", task]
        for _, candidate in staged:
            argv += ["--candidate-dir", str(candidate)]
        native_env = runtime_env(native_spec["env"], remove_visibility=True)
        native_env["PYTHONPATH"] = plan["native_import_dir"] + os.pathsep + str(script.parent) + os.pathsep + native_env.get("PYTHONPATH", "")
        runner("native", argv, native_env, native_spec["cwd"], (native / "report.json", terminal))
        native_proofs(task, staged, native)
    result = {"status": "cohort_review_completed", "task": task, "cohort": number, "records": rows,
              "generated": len(rows), "kinematic_pass": len(staged), "native_pass": sum(row["admitted"] for row in rows),
              "raw_report_sha256": sha(producer / "report.json"), "raw_bindings_sha256": sha(records),
              "native_report_sha256": sha(native / "report.json") if staged else None, "ppo_started": False}
    save(directory / "result.json", result, exclusive=True)
    return result


def reference_counts(rows):
    return dict(Counter(row["task"] for row in rows if row["admitted"]))


def assert_no_duplicates(rows):
    for key in ("id", "candidate_id", "initial_noise_sha256", "feature_content_sha256"):
        require(len({row[key] for row in rows}) == len(rows), f"duplicate {key} across generated cohorts")
    final = [row["state_fingerprint"] for row in rows if row["admitted"]]
    require(len(set(final)) == len(final), "duplicate admitted 29DoF states across cohorts")


def disk_measure(directory):
    """Measure allocated bytes without counting native hardlink copies twice."""
    seen = set()
    used = 0
    for path in directory.rglob("*"):
        st = path.stat()
        key = (st.st_dev, st.st_ino)
        if key not in seen:
            seen.add(key)
            used += st.st_blocks * 512
    return used


def run_pilots(add, checkpoint_sha, output, prior_pilots):
    receipt_path = output / "pilot_receipt.json"
    tasks = prior_pilots
    receipt = {"status": "pilots_in_progress", "pilots": tasks, "checkpoint_sha256": checkpoint_sha,
               "max_attempts_per_task": PILOT_CAP, "all_four_native_passed": False}
    save(receipt_path, receipt, exclusive=True)
    for task, details in tasks.items():
        if details["passed"]:
            continue
        for _ in range(max(0, PILOT_CAP - len(details["attempts"]))):
            result = add(task, 1)
            row = result["records"][0]
            details["passed"] = result["native_pass"] == 1
            details["attempts"].append({"candidate_id": row["candidate_id"], "passed": details["passed"],
                                        "initial_noise_seed": row["initial_noise_seed"],
                                        "kinematic": result["kinematic_pass"],
                                        "rejection_reason": row.get("rejection_reason")})
            save(receipt_path, receipt)
            if details["passed"]:
                break
        if not details["passed"]:
            receipt.update(status="pilot_quality_budget_exhausted", failed_task=task)
            save(receipt_path, receipt)
            raise ValueError(f"{task} exhausted {PILOT_CAP} single-candidate pilot attempts with quality rejection")
    receipt.update(status="pilot_native_admission_passed", all_four_native_passed=True)
    save(receipt_path, receipt)


def collect(plan_path, plan_sha, plan, cfg, output):
    checkpoint, checkpoint_sha = wait_for_checkpoint(cfg, plan, output)
    native_spec = read(bound(plan["native_runtime"]))
    device = str(plan["native_device"])
    require(Path(native_spec["python"]).is_file() and Path(native_spec["cwd"]).is_dir()
            and device.startswith("cuda:") and device[5:].isdigit(), "bound native runtime invalid")
    from resume_prefix_paper_v10 import resume
    rows, cohorts, offsets, numbers, prior_pilots = resume(
        sys.modules[__name__], plan_path, plan_sha, plan, cfg, output, checkpoint_sha, native_spec)
    largest_bytes_per_candidate = 1396736
    def add(task, count):
        nonlocal largest_bytes_per_candidate
        reserve = max(plan["cohort_write_budget_bytes"], largest_bytes_per_candidate * count)
        current = resource_snapshot(Path(cfg["project_root"]))
        resource_guard(current, reserve)
        counts = reference_counts(rows)
        remaining = sum(TARGET - counts.get(key, 0) for key in TASKS)
        save(output / "progress.json", {"status": "checking_next_cohort_budget", "task": task,
             "next_candidates": count, "minimum_runtime_free_bytes": FLOOR, "cohort_write_budget_bytes": reserve,
             "free_bytes": current["free_bytes"], "verified_admitted_counts": counts,
             "remaining_selected_references": remaining,
             "observed_largest_cohort_bytes_per_candidate": largest_bytes_per_candidate,
             "full_library_projected_bytes_at_all_future_pass": remaining * largest_bytes_per_candidate,
             "projected_bytes_scope": "allocation estimate; rejects remain on disk and can raise total"})
        number = numbers[task]
        result = run_cohort(plan_path, plan_sha, plan, cfg, output, task, number, offsets[task], count,
                            checkpoint, checkpoint_sha, native_spec)
        offsets[task] += count
        numbers[task] += 1
        result_path = output / task / f"cohort_{number:04d}" / "result.json"
        cohort_bytes = disk_measure(result_path.parent)
        largest_bytes_per_candidate = max(largest_bytes_per_candidate, (cohort_bytes + count - 1) // count)
        rows.extend(result["records"])
        assert_no_duplicates(rows)
        cohorts.append({"path": str(result_path), "sha256": sha(result_path),
                        "task": task, "generated": count, "kinematic_pass": result["kinematic_pass"],
                        "native_pass": result["native_pass"], "allocated_bytes": cohort_bytes})
        save(output / "progress.json", {"status": "cohort_verified", "candidate_counts": dict(offsets),
             "verified_admitted_counts": reference_counts(rows), "cohorts": cohorts,
             "ppo_started": False})
        return result

    run_pilots(add, checkpoint_sha, output, prior_pilots)

    for task in TASKS:
        while (needed := TARGET - reference_counts(rows).get(task, 0)) > 0:
            remaining_cap = CANDIDATE_CAP - offsets[task]
            require(remaining_cap > 0, f"{task} exhausted its fixed candidate cap")
            result = add(task, min(plan["cohort_size"], remaining_cap, needed))
            require(result["native_pass"] > 0, f"{task} had a zero-admission cohort; preserve for diagnosis")

    require(reference_counts(rows) == dict.fromkeys(TASKS, TARGET), "final native admission does not cover four sets of 500")
    for row in rows:
        row["selected_reference"] = row["admitted"]
        verify_hash(row["source_sample"], row["source_sample_sha256"])
        verify_hash(row["kinematic_report"], row["kinematic_report_sha256"])
        if row.get("source"):
            verify_hash(row["source"], row["source_sha256"])
        if row.get("native_proof"):
            verify_hash(row["native_proof"], row["native_proof_sha256"])
    manifest = output / "generation_manifest.json"
    save(manifest, {"schema": "repair_prior29_generation_v2", "checkpoint_sha256": checkpoint_sha,
         "plan_sha256": plan_sha, "training_config_sha256": plan["training_config"]["sha256"],
         "sources": plan["sources"], "records": rows, "cohorts": cohorts, "candidate_counts": dict(offsets),
         "reference_counts": reference_counts(rows), "source_fps": 20,
         "scope": "raw candidates, motion-preserving postoptimization and pose-injected native geometry admission only",
         "ppo_started": False}, exclusive=True)
    receipt = {"status": "ready_for_eval", "generation_manifest": str(manifest),
               "generation_manifest_sha256": sha(manifest), "reference_counts": reference_counts(rows),
               "checkpoint_sha256": checkpoint_sha, "evaluation_completed": False, "ppo_started": False}
    save(output / "generation_receipt.json", receipt, exclusive=True)
    save(output / "status.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--postopt-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-records", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-records-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    plan_sha = verify_hash(plan_path, args.plan_sha256)
    plan, cfg, output = setup_plan(plan_path)
    if args.postopt_worker:
        cohort = args.worker_output.resolve().parent
        require(cohort.is_relative_to(output) and args.worker_output == cohort / "postopt"
                and args.worker_records == cohort / "raw_bindings.json", "invalid postopt worker scope")
        verify_hash(args.worker_records, args.worker_records_sha256)
        postopt_worker(plan, args.worker_records, args.worker_output)
        return
    stage = Path(cfg["stage_dir"])
    with (stage / ".generation_v2.lock").open("a", encoding="utf8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        def interrupt(signum, _):
            raise InterruptedError(f"controller received signal {signum}")
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        save(output / "status.json", {"status": "waiting_for_formal_checkpoint", "pid": os.getpid(),
             "plan_sha256": plan_sha, "ppo_started": False}, exclusive=True)
        try:
            collect(plan_path, plan_sha, plan, cfg, output)
        except BaseException as error:
            (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf8")
            failure = {"status": "failed_preserved", "type": type(error).__name__, "error": str(error),
                       "ppo_started": False, "evaluation_completed": False}
            save(output / "failure.json", failure, exclusive=True)
            save(output / "status.json", failure)
            raise


if __name__ == "__main__":
    main()


