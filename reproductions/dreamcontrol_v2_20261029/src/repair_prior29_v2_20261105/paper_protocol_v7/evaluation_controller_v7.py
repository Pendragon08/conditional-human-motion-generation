"""Wait for four verified reference sets, then run the pinned CPU evaluation."""

from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from job_guard_v1 import digest, resources, write


STAGE = Path(__file__).resolve().parent.parent
ROOT = STAGE.parent
GENERATION = STAGE / "new_references_paper_v7"
REPORTS = STAGE / "reports"
CONTROLLER = STAGE / "evaluation_controller_paper_v7"
OUTPUT = STAGE / "evaluation_paper_v7"
BUILDER = STAGE / "src/eval_cache_builder_v2.py"
BUILDER_SHA = "59abd3e6655aa931b965cfa2cc0af3ff7200fc7f295dc37a4517769b9c90fd45"
PYTHON = ROOT.parent / "reactivebfm-reproduction-20261021/env/bin/python"
TARGET = dict.fromkeys(("pick", "pour", "bimanual_box_lift", "box_push_pull"), 500)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_resources(floor, pgid=None, reserve=0):
    actual = resources(ROOT, pgid)
    if actual["free_data_bytes"] < floor + reserve:
        raise RuntimeError("Insufficient protected evaluation disk space")
    if actual["job_gpu_bytes"] or actual["project_gpu_bytes"] > 24 * 1024**3:
        raise RuntimeError("Evaluation must run on CPU and respect the project GPU limit")
    if actual["job_rss_bytes"] > 32 * 1024**3 or actual["project_rss_bytes"] > 64 * 1024**3:
        raise RuntimeError("Evaluation memory limit exceeded")
    return actual


def main():
    CONTROLLER.mkdir(exist_ok=True)
    status_path = CONTROLLER / "status.json"
    status = {"pid": os.getpid(), "source_sha256": digest(Path(__file__)), "ppo_started": False}
    floor = read(STAGE / "config.json")["disk_floor_bytes"]
    tokens = REPORTS / "caption_tokens_paper_v7.json"
    token_receipt = REPORTS / "caption_tokens_paper_v7_receipt.json"
    generation_receipt = GENERATION / "generation_receipt.json"
    child = None
    with (CONTROLLER / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if OUTPUT.exists() or digest(BUILDER) != BUILDER_SHA:
                raise RuntimeError("Evaluation already exists or builder source changed")
            while True:
                generation_status = read(GENERATION / "status.json") if (GENERATION / "status.json").exists() else {"status": "initializing"}
                if generation_status["status"] == "failed_preserved":
                    raise RuntimeError("Generation failed and requires diagnosis")
                missing_generation = not generation_receipt.is_file()
                missing_tokens = not (tokens.is_file() and token_receipt.is_file())
                status.update(status="waiting_for_verified_inputs",
                              waiting_for_generation=missing_generation, waiting_for_tokens=missing_tokens)
                write(status_path, status)
                if not missing_generation and not missing_tokens:
                    break
                time.sleep(30)
            generation = read(generation_receipt)
            manifest = GENERATION / "generation_manifest.json"
            manifest_sha = digest(manifest)
            if (generation["status"] != "ready_for_eval" or generation["reference_counts"] != TARGET
                    or generation["generation_manifest"] != str(manifest)
                    or generation["generation_manifest_sha256"] != manifest_sha or generation["ppo_started"]):
                raise RuntimeError("Generation receipt does not authorize this evaluation")
            rows = read(manifest)["records"]
            selected = [row for row in rows if row["selected_reference"]]
            if Counter(row["task"] for row in selected) != TARGET or not all(row["admitted"] for row in selected):
                raise RuntimeError("Manifest does not contain 500 admitted references per task")
            proof = read(token_receipt)
            token_data = read(tokens)
            if (proof["new_output_sha256"] != digest(tokens)
                    or token_data["tokenizer"]["spacy"] != "3.8.11"
                    or token_data["tokenizer"]["model"] != "3.8.0"
                    or not {row["caption"] for row in rows} <= token_data["tokens"].keys()):
                raise RuntimeError("Caption token provenance or coverage mismatch")
            check_resources(floor, reserve=256 * 1024**2)
            cmd = [str(PYTHON), "-B", "-u", str(BUILDER),
                   "--generation-manifest", str(manifest), "--generation-sha256", manifest_sha,
                   "--assets", str(ROOT / "generated_reference_evaluation_20261105/evaluator_assets_v2"),
                   "--tokens", str(tokens), "--encoder-dir", str(ROOT / "training_disk_guard_20261101"),
                   "--output", str(OUTPUT)]
            with (CONTROLLER / "worker.log").open("xb") as log:
                child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, start_new_session=True)
                while child.poll() is None:
                    status.update(status="evaluation_running", worker_pid=child.pid,
                                  generation_manifest_sha256=manifest_sha,
                                  resources=check_resources(floor, child.pid))
                    write(status_path, status)
                    time.sleep(5)
            if child.returncode:
                raise RuntimeError(f"Evaluator exited with code {child.returncode}")
            result = read(OUTPUT / "completion_receipt.json")
            if result["status"] != "new_generation_three_stage_metrics_completed":
                raise RuntimeError("Evaluation completion receipt missing")
            status.update(status="completed", completion_receipt=result)
            write(status_path, status)
        except BaseException as error:
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            status.update(status="failed_preserved", error=str(error))
            write(status_path, status)
            raise


if __name__ == "__main__":
    main()
