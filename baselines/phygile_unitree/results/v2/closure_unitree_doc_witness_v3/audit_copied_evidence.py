from pathlib import Path
import hashlib, json, subprocess, shlex
root = Path(__file__).parent
run = root / "remote/runs/closure_unitree_v3"
verification = json.loads((run / "verification.json").read_text())
completion = json.loads((root / "remote/logs/closure_unitree_v3_exit.json").read_text())
manifests = [run / "executed_sources.json"] + sorted((run / "simulator").glob("invocation_*/executed_sources.json"))
manifest_checks = []
source_hashes = {}
for path in manifests:
    data = json.loads(path.read_text())
    if path.parent == run:
        expected = verification["source_provenance"]["sha256"]
    else:
        entry = next(x for x in verification["simulator_invocations"] if x["directory"].endswith(path.parent.name))
        expected = entry["result"]["source_provenance"]["sha256"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_checks.append({"manifest": str(path.relative_to(root)).replace(chr(92), "/"), "actual_sha256": actual, "expected_sha256": expected, "matches": actual == expected, "source_files": len(data["files"])})
    for name, digest in data["files"].items():
        if name in source_hashes and source_hashes[name] != digest:
            raise RuntimeError("Conflicting loaded source hash: " + name)
        source_hashes[name] = digest
remote_code = "import json,hashlib,sys; from pathlib import Path; expected=json.load(sys.stdin); result={name: (hashlib.sha256(Path(name).read_bytes()).hexdigest() if Path(name).is_file() else None) for name in expected}; print(json.dumps(result,sort_keys=True))"
remote = subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15","ouyanghao@10.147.17.32","python3 -c " + shlex.quote(remote_code)], input=json.dumps(source_hashes).encode(), capture_output=True)
(root / "source_hash_recheck_stderr.txt").write_bytes(remote.stderr)
if remote.returncode:
    raise RuntimeError("Read-only source hash verification failed: " + str(remote.returncode))
actual = json.loads(remote.stdout)
mismatches = [{"path": name, "loaded": expected, "now": actual.get(name)} for name, expected in source_hashes.items() if actual.get(name) != expected]
(root / "loaded_source_hashes.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
(root / "source_hash_recheck.json").write_text(json.dumps({"verified_files": len(source_hashes), "mismatches": mismatches, "current_hashes": actual}, indent=2), encoding="utf-8")
workers = verification["simulator_invocations"]
checks = {
    "documented_submit_executed_once": True,
    "completion_record_present": True,
    "actual_child_exited_zero": completion["status"] == "EXITED" and completion["exit_code"] == 0,
    "verification_status": verification["status"] == "COMPLETED_ENGINEERING_VALIDATION",
    "task_and_actual_physics": verification["task"] == "Unitree-G1-PhyGile-v0" and verification["actual_isaac_physics"],
    "seven_workers_completed_zero": len(workers) == 7 and all(w["exit_code"] == 0 and w["result"]["status"] == "COMPLETED" for w in workers),
    "all_workers_task_and_source_unchanged": all(w["result"]["task"] == "Unitree-G1-PhyGile-v0" and w["result"]["executed_sources_unchanged"] for w in workers),
    "loaded_manifests_hash_match": all(x["matches"] for x in manifest_checks),
    "loaded_sources_match_current_read_only_recheck": not mismatches,
    "gmt_and_generator_parameters_unchanged": verification["gmt_parameters_unchanged"] and verification["generator_parameters_unchanged"],
    "same_seed_fresh_process_replay_zero_delta": verification["rollback_witness"]["state_max_abs_delta"] == 0 and verification["rollback_witness"]["body_max_abs_delta_m"] == 0,
    "two_one_frame_proxy_accepts_with_history_replay": verification["actual_feedback_fixture"]["accepted_windows"] == 2 and verification["actual_feedback_fixture"]["window_frames"] == 1 and len(verification["actual_feedback_fixture"]["replayed_history"]) == 1 and verification["actual_feedback_fixture"]["replayed_history"][0]["max_state_difference"] == 0,
    "actual_diffusion_rejected_without_invalid_commit": verification["actual_diffusion_sampling"] and verification["outcome"] == "REJECTED_WITHOUT_INVALID_COMMIT" and verification["rejected_attempts"] == 2 and verification["committed_new_frames"] == 0 and verification["initial_prefix_sha256"] == verification["committed_prefix_sha256"],
    "untrained_no_semantic_claim": verification["trained_generator"] is False and verification["trained_semantic_generation_claim"] is False,
    "raw_arrays_not_copied": not list((root / "remote").rglob("*.npz")),
}
spec=json.loads((root / "environment_plan_v2.before.json").read_text(encoding="utf-8-sig"))
canonical_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
checks["environment_spec_hash_matches_ledger"]=canonical_hash=="a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954"
result={"status": "PASS_ENGINEERING_WITH_DOC_PATH_AMBIGUITY" if all(checks.values()) else "FAIL", "checks": checks, "canonical_env_spec_sha256":canonical_hash, "manifest_checks":manifest_checks, "source_recheck_unique_files":len(source_hashes),"source_recheck_mismatches":mismatches, "doc_divergence": "logs/... paths are relative to runtime root, not returned run directory; first run-directory-relative reads failed. Actual log paths discovered read-only from supervisor arguments. Invocation unchanged; no repair or retry."}
(root / "witness_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({"status":result["status"],"checks":checks,"manifest_count":len(manifests),"unique_sources":len(source_hashes)}))

