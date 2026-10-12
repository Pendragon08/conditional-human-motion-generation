# Experiment Audit Report: Remaining Candidate Component Screen

**Date:** 2026-10-12  
**Auditor:** fresh Codex reviewer, same-family  
**Acceptance:** provisional  
**Evaluation type:** `simulation_only`  
**Overall verdict:** **PASS — component-diagnostic integrity only**

The retrospective screen is hash-bound and exactly reproducible from the specified records. This PASS does not establish C2 efficacy, a C3 repair-yield fit, acquisition savings, controller benefit, robot transfer, or real-world performance.

## Deterministic findings

- All 37 paths bound by `component_screen.json` exist, the bound path set is exact, and every SHA-256 matches the current file bytes.
- Forty relevant pre-audit files—including the component result, all 30 requested raw metric/metadata files, `summary.json`, `versions.json`, frozen thresholds, source manifest, three scripts, and two documents—were byte-identical between the local and remote repositories.
- The frozen threshold file hashes to `451fae0a2b106dbd0170185d9a22bb6f3fe2583709e77979eb017cc3ce09315b`, exactly the value in `versions.json`.
- Independent CPU recomputation checked four pass labels on 1,080 raw rows, for 4,320 Boolean comparisons. Mismatches: **0**. The recomputed per-file and parent-summary counts also match exactly.
- Independent reimplementation of the guard rebuilt all 432 decisions, 216 final choices, all reason strings, all selected rows, the three reports, and the six replay summaries. Mismatches: **0**.

The construction bank contains 648 rows: 634 completed, 173 condition-pass, 464 tracking-pass, **0 physical-pass**, and **0 joint-pass**. The 432 replay rows contain 429 completed, 102 condition-pass, 404 tracking-pass, **0 physical-pass**, and **0 joint-pass**. Every construction and replay row fails the frozen surface-slip fraction threshold: the best observed fractions are 0.1105990783 and 0.2038216561 respectively, above the fixed 0.1 limit.

The guard retained candidate 0 for 197 cases, selected candidate 1 for 10, and selected candidate 2 for 9. Across 432 decisions, the reason classes are 402 component regressions, 10 incomplete candidates, 2 complete replacements of incomplete incumbents, and 18 componentwise improvements. Those 20 accepted transitions produce 19 nonzero final selections because one case accepts twice. Every final selection still has physical-pass and joint-pass false.

## A. Ground-truth and target provenance: PASS

The screen consumes pre-existing simulation metrics and fixed dataset-motion goals. For every seed, the three construction rounds and both replay arms have the same 72 ordered case/source identities, metadata case order, original goals, and original headings. No target is derived from a selected model output. The checks in [remaining_candidate_screen_20261012.py](../../scripts/remaining_candidate_screen_20261012.py) lines 38–57 and 72–85 enforce part of this pairing; this audit additionally compared the actual arrays and row values.

The result remains a simulator diagnostic. [physics_guided_core.py](../../scripts/physics_guided_core.py) lines 1–4 explicitly identifies surface slip as a simulator/geometry proxy rather than contact-point truth, and raw rows mark semantics as not independently verified. The proper classification is `simulation_only`, not real-ground-truth performance evaluation.

## B. Score normalization: PASS

The guard in [condition_feedback_guard_core_v29_20261012.py](../../scripts/condition_feedback_guard_core_v29_20261012.py) lines 80–98 compares eight raw SI-unit error metrics directly. The pass labels use the fixed thresholds implemented in [physics_guided_core.py](../../scripts/physics_guided_core.py) lines 142–154. There is no division by the bank maximum, minimum, mean, or any statistic of the selected output. The repairs-per-paid-call rate defined in the C3 core is not used by this screen.

## C. Result existence and numerical reconciliation: PASS

Every input and claimed output exists. The component result hashes to `d30e4d818ec79682fe27827e1babf36f3e8b660eb8035cfdd3371a00f79f7178`. The canonical digest of its 37 bound input hashes is `7e5a549017bd565f57ab2dfdbdd858856d5c140f01b6aa27b0ebebd88a030f4c`; the canonical digest of the 40-file audit comparison is `216766b14eea5ecc654815f37bf2a1af3201583c8bf3425d0350abe9ef86e1f3`.

The local and remote repositories have different, dirty HEAD states (`605c77e…` locally, `e1ef719…` remotely), and the audited files are not commit-addressed. Exact file hashes therefore define the audited state. This does not affect the byte-for-byte agreement of the reviewed inputs.

## D. Claimed execution and dead code: PASS

The screen calls `acceptable()` for all 432 decisions at [remaining_candidate_screen_20261012.py](../../scripts/remaining_candidate_screen_20261012.py) lines 58–71. C2 correction functions and the C3 fit/allocator are not called, and the result explicitly marks C2 and C3 as not run at lines 104–105. No output is attributed to those unexecuted paths.

Static inspection of the screen and its imported module path finds JSON reads, hashing, scalar comparisons, and output serialization only. There is no simulator, generation, controller-training, subprocess, torch/CUDA, or GPU API call. All metric inputs predate the component output. This corroborates zero new calls and zero GPU use for the audited script path; no external scheduler or power telemetry was supplied to exclude unrelated concurrent activity.

## E. Scope: PASS

The exact scope is 24 sources, three offsets per source, three seeds, 648 observed construction candidates, 432 component decisions, and six pre-existing independent replay batches containing 432 rows. All 24 sources are internal-validation sources. The bound source manifest shows 19 come from the official HumanML3D train split and 5 from its official test split, with 12 jump and 12 dance sources. This is not held-out upstream evaluation. The limitation is correctly stated at [remaining_candidate_screen_20261012.py](../../scripts/remaining_candidate_screen_20261012.py) lines 96–110.

## F. Claim impact

- **Supported:** the finite-bank component choice counts, reasons, and selected rows are reproducible.
- **Supported with a strict bound:** no selector can recover a physical-pass or joint-pass row from this observed 648-candidate bank because every row has both labels false. This says nothing about unseen candidates.
- **Unsupported / not run:** C2 efficacy. The audited bank is not the preregistered repeat-plus-correction C2 intervention, and the six replay batches are the existing feedback/random replay arms rather than independent replay of the newly guard-selected choices.
- **Unsupported / not run:** C3 repair-yield fitting. [repair_budget_core_v38_20261012.py](../../scripts/repair_budget_core_v38_20261012.py) lines 21–47 requires paired validation-only fixed-reference repair records and at least one observed joint repair; neither exists here.
- **Unsupported / not run:** prospective acquisition benefit, paid-call savings, elapsed-cost benefit, controller benefit, downstream benefit, robot transfer, and real-world performance.

## Stopping prerequisites

C2 remains stopped because the frozen parent evidence has no nonzero joint-pass example. Do not relax or retune the quality floor to proceed. Before any efficacy claim, the documented route requires an eligible nonzero joint example, a fresh document-following witness, the actual three-call intervention, independent replay of its selected references, paired analysis, and semantic/cost checks. These boundaries are stated in [condition_feedback_guard_simulation_v31_20261012.md](../../docs/condition_feedback_guard_simulation_v31_20261012.md) lines 3–9 and 17–23.

C3 remains stopped. It first needs resolved measurement semantics; paired validation-only fixed-reference before/after repair observations with explicit paid-call counts and at least one positive joint repair; a frozen allocator and genuinely scarce equal budget; a fresh witness; and actual acquisition, elapsed-cost, and frozen-controller downstream comparisons. These prerequisites are stated in [repair_budget_candidate_v38_20261012.md](../../docs/repair_budget_candidate_v38_20261012.md) lines 16–29.

The reviewer is same-family, so this positive integrity verdict is provisional rather than cross-family acceptance.
