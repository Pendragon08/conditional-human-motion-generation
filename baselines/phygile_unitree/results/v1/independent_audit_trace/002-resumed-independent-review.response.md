# Experiment integrity review response

**Overall verdict: WARN.** Review status is complete; acceptance remains `provisional` because the reviewer is same-family.

## A. Ground-truth provenance: PASS

Every stored reference position/velocity/body array in the eight evaluations equals the corresponding source motion array exactly after name-based selection. The evaluators capture reference and robot arrays separately (`scripts/lab_run.py:171-177`, `scripts/mjlab_evaluate.py:42-51`), and `analyze_crosssim.py:18-26` computes raw differences. No reference is generated from policy output.

## B. Own-output normalization: PASS

MPJPE and joint errors are raw Euclidean/absolute means (`scripts/lab_run.py:192-197`, `scripts/mjlab_evaluate.py:59-66`). The saved empirical normalizer in `scripts/frozen_policy.py:15-28` applies only to policy input. No metric uses the prediction's own maximum, mean, or range.

## C. Existence, keys, numbers, and status: WARN

All claimed files, keys, table values, completion flags, and pending statuses match. All 24 private raw manifest entries and eight private asset copies match sizes and hashes; all 24 metric values recompute exactly. The warning is that Isaac receipts hash only `lab_run.py` (`scripts/lab_run.py:25-30`) and Mjlab receipts hash no source (`scripts/mjlab_evaluate.py:8-17`). Current Mjlab code contains the documented list/tuple assertion repair relative to archived `code_v5`, but the completed receipts do not bind that source. The compute ledger's `a43f960b...` canonical digest (`docs/COMPUTE_LEDGER.md:3-6`) also lacks a documented canonicalizer and did not reproduce with standard sorted JSON.

## D. Dead metric paths: PASS

All reported metrics execute in each evaluator and are recomputed independently (`scripts/analyze_crosssim.py:18-30`). The training branch exists but is explicitly `NOT_STARTED` (`EXPERIMENT_TRACKER.md:6,13`); no inference receipt reports optimizer steps.

## E. Scope/seed/claim match: PASS

The evidence is exactly four short clips, two simulated engines, one deterministic episode per clip/engine, seed 2026, with failure censoring (`results/v1/crosssim_summary.json:4-7,230`). README and report language stays within a small simulation pilot and rejects paper-scale, equivalence, generalization, or real-robot claims.

## F. Evaluation type: PASS

Tracking is `real_gt` reference provenance plus `simulation_only` execution. Numerical parity is a `synthetic_proxy` implementation check. Compiled dynamics is a static simulation diagnostic. The document witness is a bounded simulation-only probe. There is no human or real-robot evaluation.

## Deterministic result

- Reference-array maximum absolute difference: `0.0`.
- Metric maximum absolute difference: `0.0`.
- Fresh frozen-policy maximum action difference: `2.682209014892578e-7`, passing fixed `1e-5` tolerance.
- Initial 520D command maximum difference: `0.0`.
- Completion counts: Isaac `2/4`, Mjlab `1/4`.
- Checkpoint: SHA-256 `bb067f364de74e2da7cf1879e6f26bb961a71eb9ff64cbfbe84dd858bfae2336`, `iter=1999`, optimizer state steps `40000`.

## Actionable findings

1. Future receipts should bind a pre-execution manifest covering the entrypoint, imported task/adapter/encoder files, contract, framework commit, and resolved environment lock.
2. Define and reproduce the environment-plan canonicalization algorithm or correct the ledger digest.
3. Keep the historical partial source-attestation qualifier; do not retrofit hashes into prior receipts.

The complete evidence, claim impacts, hashes, and exact line citations are in `EXPERIMENT_AUDIT.md` and `EXPERIMENT_AUDIT.json`.
