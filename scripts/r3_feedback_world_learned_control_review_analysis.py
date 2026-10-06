"""Read-only outcome audit; post-hoc diagnostics, never new control trials."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "feedback_world_model_20261006"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    learned = json.loads((OUT / "learned_control_summary.json").read_text())
    oracle = json.loads((OUT / "oracle_summary.json").read_text())
    fitted = json.loads((OUT / "summary.json").read_text())
    config_path = ROOT / "configs" / "feedback_world_learned_control_20261006.json"
    config = json.loads(config_path.read_text())
    assert learned["status"] == "COMPLETE"
    assert learned["config_sha256"] == sha(config_path)
    assert learned["oracle_config_sha256"] == oracle["config_sha256"]
    assert learned["model_hashes"] == fitted["checkpoint_hashes"]
    rows = learned["records"]
    expected = {(task, ref, arm, seed) for task in ("Kick", "Punch")
                for ref in range(46, 66) for arm in config["arms"]
                for seed in config["model_and_candidate_seeds"]}
    observed = [(r["task"], r["reference"], r["arm"], r["seed"]) for r in rows]
    assert len(rows) == 240 and len(set(observed)) == 240 and set(observed) == expected
    assert all(np.isfinite(r["recovery_tracking_AUC_rad_s"]) for r in rows)
    assert all(0 < r["executed_steps"] <= 500 for r in rows)
    baseline = oracle["records"]
    keys = [(task, ref) for task in ("Kick", "Punch") for ref in range(46, 66)]

    def macro(arm):
        source = rows if arm in config["arms"] else baseline
        return np.array([np.mean([r["recovery_tracking_AUC_rad_s"] for r in source
                                  if r["arm"] == arm and (r["task"], r["reference"]) == key])
                         for key in keys])

    arrays = {arm: macro(arm) for arm in ("ppo", "simple", "oracle", *config["arms"])}

    def comparison(candidate, control):
        rng = np.random.default_rng(20261006)
        boot = []
        for _ in range(2000):
            ids = np.concatenate([rng.choice(np.arange(j * 20, (j + 1) * 20), 20, replace=True)
                                  for j in range(2)])
            boot.append(1 - arrays[candidate][ids].mean() / arrays[control][ids].mean())
        return {"relative_AUC_improvement": float(1 - arrays[candidate].mean() / arrays[control].mean()),
                "reference_bootstrap95CI": np.quantile(boot, [.025, .975]).tolist()}

    comparisons = {f"{arm}_vs_{control}": comparison(arm, control)
                   for arm in config["arms"] for control in ("ppo", "simple", "oracle")}
    comparisons["multi5_vs_single"] = comparison("mlp_multi5", "mlp_single")
    for control, field in (("ppo", "PPO"), ("simple", "simple")):
        np.testing.assert_allclose(comparisons[f"mlp_multi5_vs_{control}"]["reference_bootstrap95CI"],
                                   learned[f"reference_bootstrap95CI_vs_{field}"], atol=1e-12)

    task_breakdown = {}
    coupled_seed_breakdown = {}
    timing = {}
    for arm in arrays:
        source = rows if arm in config["arms"] else baseline
        for task in ("Kick", "Punch"):
            rs = [r for r in source if r["arm"] == arm and r["task"] == task]
            task_breakdown.setdefault(task, {})[arm] = {
                "mean_recovery_AUC_rad_s": float(np.mean([r["recovery_tracking_AUC_rad_s"] for r in rs])),
                "virtual_target_success_rate": float(np.mean([r["virtual_target_success"] for r in rs])),
                "fall_rate": float(np.mean([r["fallen"] for r in rs])),
                "physical_trials": len(rs)}
        if arm in config["arms"]:
            coupled_seed_breakdown[arm] = {}
            for seed in config["model_and_candidate_seeds"]:
                seed_rows = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
                coupled_seed_breakdown[arm][seed] = {
                    "mean_recovery_AUC_rad_s": float(np.mean([r["recovery_tracking_AUC_rad_s"] for r in seed_rows])),
                    "virtual_target_success_rate": float(np.mean([r["virtual_target_success"] for r in seed_rows])),
                    "fall_rate": float(np.mean([r["fallen"] for r in seed_rows])),
                    "physical_trials": len(seed_rows)}
            rs = [r for r in rows if r["arm"] == arm and r["decisions"]]
            decisions = sum(r["decisions"] for r in rs)
            timing[arm] = {
                "mean_decision_wall_s": sum(r["mean_decision_wall_s"] * r["decisions"] for r in rs) / decisions,
                "max_decision_wall_s": max(r["max_decision_wall_s"] for r in rs),
                "configured_100ms_deadline_overruns": sum(r["deadline_overruns"] for r in rs),
                "decisions": decisions,
                "timed_scope": "choose_model candidate scoring only; excludes main PPO, physics, communication and hardware I/O",
                "exact_20ms_overrun_count_unavailable": True,
                "not_hardware_realtime_proof": True}

    rates = learned["rates"]["mlp_multi5"]
    ppo = oracle["rates"]["ppo"]
    checks = {
        "AUC_improvement_at_least_10pct_vs_PPO": comparisons["mlp_multi5_vs_ppo"]["relative_AUC_improvement"] >= .1,
        "reference_CI_lower_positive_vs_PPO": comparisons["mlp_multi5_vs_ppo"]["reference_bootstrap95CI"][0] > 0,
        "reference_CI_lower_positive_vs_simple": comparisons["mlp_multi5_vs_simple"]["reference_bootstrap95CI"][0] > 0,
        "virtual_success_drop_at_most_5pp_vs_PPO": rates["virtual_goal_success_rate"] >= ppo["virtual_target_success_rate"] - .05,
        "fall_rate_not_increased_vs_PPO": rates["fall_rate"] <= ppo["fall_rate"]}
    verdict = "PASS" if all(checks.values()) else "FAIL"
    assert learned["learned_control_gate"] == verdict
    result = {
        "status": "PASS", "audit_type": "primary outcome verification and post-hoc diagnostics",
        "all_240_physical_trials_present_and_unique": True,
        "config_model_and_oracle_hash_receipts_match": True,
        "formal_gate_preserved": verdict, "formal_gate_checks_recomputed": checks,
        "comparisons": comparisons, "task_breakdown": task_breakdown,
        "coupled_model_candidate_seed_breakdown": coupled_seed_breakdown, "timing": timing,
        "cross_run_initial_states_reconstructed_with_same_deterministic_prefix_not_byte_hash_verified": True,
        "reference_bootstrap_averages_bound_model_candidate_seeds": True,
        "reference_ci_does_not_cover_full_model_candidate_randomness": True,
        "reuse_of_test_references_means_staged_pilot_not_final_independent_confirmation": True,
        "virtual_target_not_physical_contact": True,
        "no_new_model_training_or_control_trials": True,
        "full_innovation_validated": False}
    (OUT / "learned_control_reviewer_diagnostic.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
