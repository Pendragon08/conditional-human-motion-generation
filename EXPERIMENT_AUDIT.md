# I1 experiment integrity audit

Date: 2026-09-29. Auditor: fresh GPT-5.6-Sol ultra agent, same-family provisional review. Scope: `reports/04_I1_feasibility.md`, `reports/05_I1_dynamics_contact_semantics_efficiency.md`, I1 scripts and `results/r3_i1_*.json`.

**Overall verdict: WARN.** Reported figures match the saved JSON, and no self-normalization or false target-impact claim was found. The experiment does **not** establish the full I1 innovation.

| Check | Verdict | Evidence and implication |
| --- | --- | --- |
| A. Target provenance | WARN | Kinematic scoring reads saved pre-generation right-foot hints (`scripts/r3_i1_robotspace_ik_pilot.py`). The ordinary Isaac task derives its goal from each reference, while the controlled variant freezes a **raw-reference-derived** goal (`scripts/r3_i1_play_eval_metrics.py`). Neither simulation result is a fixed external-hint task score. |
| B. Normalization | PASS | Distances, counts and elapsed time use physical units or fixed observation denominators. The 80% kick-event retention is explicitly an exploratory relative proxy (`scripts/r3_i1_semantic_contact_proxy.py`). |
| C. Result existence | WARN | The reviewer recomputed the table figures against existing result JSON and found no mismatch. Licensed/generated source motions and complete run logs remain on the server, so the published small artifacts alone do not independently replay the experiment. |
| D. Dead metrics | PASS | Metric helpers are called and outputs occur in result JSON. An unused upstream `--use_pretrained_checkpoint` branch contains a debugger; tested queues explicitly pass `--path`, so it did not affect these runs. |
| E. Scope | WARN | One Kick text, three generation seeds, one simulator seed and one fixed tracker; the changed-cap study covers seed 42 only. The report does not claim general semantic matching, motor feasibility, a same-budget full-I1 ablation or efficiency gain. |
| F. Evaluation type | PASS | External-condition kinematic diagnostic; semantic/contact geometry proxies; simulation-only fixed-policy test; component wall-time measurement. No human evaluation or benchmark ground-truth text score. |

The reviewer also identified that matching initial motion IDs does not synchronize later trajectories after resets. `results/r3_i1_physics_paired_summary.json` and the four frozen-goal files preserve the differing subsequent counts. Accordingly, the report describes **matched initialization**, not frame-for-frame paired rollout.

After the provisional review, `scripts/r3_i1_pkl_field_integrity.py` checked all 191 raw/corrected files. The five fields are present on both sides; `global_pose` translation/rotation, root position, task flag, event index and all non-right-leg joints are unchanged. Only six right-leg joint columns may differ. Per-file hashes and results are in `results/r3_i1_pkl_integrity_seed{42,43,44}.json`. `results/r3_i1_provenance_manifest_v2.json` records asset, script, checkpoint, state and log hashes without publishing licensed motion files or complete logs. These additions improve traceability but do **not** upgrade the provisional audit to independent reproduction.

The full 37 I1 run logs, nine state receipts and two exact server-side evaluation scripts were archived locally at `D:\具身智能\conditional-human-motion-generation\artifacts\r3_i1_forensic_logs_20260929.tar.gz` (SHA-256 `e735785530d34ece25ce4ccc11630247df1bcd167d3b0df3bcd81b62a6ac0c26`). This ignored archive is intentionally absent from GitHub; source motion assets remain on the licensed server.

Remaining actions before a full claim: (1) a fixed pre-generation external target in the simulator; (2) synchronized per-trajectory execution or an explicitly matched rollout schedule; (3) a collidable target with contact-pair and impulse recording; (4) controller training/actuator-load assessment if claiming physical executability; (5) multiple text/action classes and repeated full-pipeline latency measurements; (6) same-budget Base/module/Full I1 comparison. Until then the correct result is **I1 Gate not passed**.

Audit type: same-family provisional. Raw reviewer trace is kept locally under `.aris/traces/experiment-audit/2026-09-29_run01/` and is not published with the repository.

## 2026-09-30 matched-control completion receipt

Primary-agent verification (not an independent reviewer pass): both arms completed 2000 iterations / 393,216,000 timesteps. Saved PPO agent YAML is identical; environment YAML differs only by paired reference-library path. Two final checkpoint hashes, eight video hashes/decodable streams and all log/state success rates were verified. Source receipt: `results/r3_i1_matched_completed.json`.

All four same-reference comparisons favor Base: candidate minus Base = -10.6, -1.3, -3.3 and -3.1 percentage points. The candidate does not demonstrate downstream success-rate utility in this single-training-seed Kick experiment. **Full I1 Gate remains not passed.** Matched Base-versus-candidate controller training is now complete, but per-module ablations, original external target preservation, physical contact, multiple text classes, multiple control-training seeds and sim-to-sim remain unverified. No I2/I3 stage was started.
