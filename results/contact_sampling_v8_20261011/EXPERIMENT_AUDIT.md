# Experiment Audit Report

**Date:** 2026-10-11  
**Auditor:** fresh Codex read-only reviewer, same-family  
**Review status:** provisional  
**Project:** contact-sampling V8 timing witness  
**Overall verdict:** **WARN**

The V8 artifacts are internally coherent and honestly preserve a negative repeatability result. The audit found no phantom files, output-dependent score normalization, altered thresholds, or hidden efficacy claim. The physical contact-force channel nevertheless fails its own repeatability criterion by orders of magnitude, and the run has no comparison arm or independent seed. It can support a narrow statement about the archived timing/reset diagnostic; it cannot support contact-force validity or method efficacy.

## Outcome by evidence class

| Evidence class | Status | Finding |
|---|---|---|
| Artifact integrity | PASS | All listed files exist. Version, request, lineage, receipt, metadata, metric, trajectory, weight, external-input, commit, and foot-asset hashes checked in this audit are coherent. |
| Reset-state reproducibility | PASS, limited | The seven archived bootstrap tensors equal their first-action counterparts bit-for-bit in both repeats, and the two `initial_states.npz` files are byte-identical. The four private reset-cache comparisons are only stored as metadata scalars; their source cache tensors were not archived. |
| Completion-fence instrumentation | WARN | The bound V8 code calls `fetch_results(sim, True)` immediately after each intercepted `simulate`, and both metadata files report 242 calls. There is no per-call trace, so 242 cannot be reconstructed independently from the raw NPZ files. |
| Metric recomputation | PASS | The in-environment certificates report exact equality. An independent scratch recomputation reproduced every discrete value and classification; the largest cross-NumPy floating drift was 0.0003235 in a jerk quantile and did not affect any gate. |
| Contact-force repeatability | FAIL | With all recorded poses, references, rotations, linear velocities, and angular velocities bit-identical, `contact_force` differs by **728.0306 N** maximum and net all-body force differs by **1591.6550 N**. The declared 1 mN repeatability test is false. |
| Physical metric validity | FAIL for efficacy use | Force-gated support membership changes across repeats, moving surface-slip fractions by as much as **0.02559**. The code correctly labels this a body-force/collision-surface proxy, but the channel is not repeatable here. |
| Method efficacy | UNTESTED | Both batches run the same native arm, seed 11, round 0, and noise. There is no baseline/intervention comparison, independent seed, planned arm, replay arm, or semantic evaluation. |

## A. Ground-truth and proxy provenance: WARN

The original endpoint and heading are derived from cached HumanML motion before generation and then held fixed (`scripts/physics_guided_collect.py:57-63, 94-139`). This is dataset-derived condition evidence, with a predeclared horizontal offset, rather than a target synthesized from the evaluated output.

The tracking reference is mixed. Frames 0-59 use decoded dataset motion; frames 60-119 use the diffusion model's generated plan (`scripts/physics_guided_collect.py:147-204`). Tracking metrics intentionally begin at frame 60 (`scripts/physics_guided_core.py:121-141`), so they measure simulated-controller agreement with a model-generated command. They are a controller-tracking proxy, not real ground-truth motion quality. This does not invalidate the timing diagnostic, but outputs should explicitly classify these fields as `synthetic_proxy` or `self_supervised_proxy`.

The base collector's opening statement, “Generated targets are never evaluation targets” (`scripts/physics_guided_collect.py:1`), conflicts with the generated reference used by `tracking_mean_m` and `tracking_p95_m`. The intended narrower meaning appears to be that the externally scored endpoint and heading remain dataset-derived.

The manifest is a custom four-clip measurement set. Its internal `split` is `val`, while the four rows identify two official-train and two official-test clips. This is not an official benchmark validation split and no official benchmark evaluator is used. The retained `official_split` fields make the provenance inspectable.

## B. Normalization: PASS

The only feature normalization is `(raw - Mean) / Std` using pinned dataset statistics (`scripts/physics_guided_collect.py:57-63`). The audited Mean and Std arrays are finite; all 263 standard deviations are positive, with minimum 0.0128256, and all four normalized inputs are finite. Their hashes match `versions.json`.

Reported metrics remain in SI units or raw fractions against fixed thresholds (`scripts/physics_guided_core.py:100-155`). No reported score is divided by the model output's maximum, minimum, or mean. The surface-slip fraction uses the number of force-gated support observations as its natural denominator; its instability is a measurement issue, not score-normalization fraud.

## C. Result existence and numbers: PASS

Both receipts report 12 cases, 12 completed cases, and zero joint passes. Independent reading and recomputation found, for each repeat:

- 12/12 cases have 120 valid frames and no termination;
- 4/12 pass the condition gate;
- 12/12 pass the controller-tracking gate;
- 0/12 pass the physical gate;
- 0/12 pass the joint gate.

The final state is `COMPLETE_CONTACT_SAMPLING_WITNESS`. The sampling summary was independently reconstructed exactly from the two NPZ files: all kinematic channels compare equal, contact force does not, classifications compare equal, and the 1 mN force-repeatability flag is false. Classification equality is degenerate here because every case fails the physical and joint gates; it is not evidence that the contact measurement is repeatable.

## D. Dead-code assessment: PASS with notes

The result-producing call path is live: `run_contact_sampling_v8_20261011.py:144-157` calls `batch`, which launches the V8 collector (`:67-112`); V8 deterministically applies V5 and base-collector patches (`contact_sampling_collect_v8.py:4-43`; `contact_sampling_collect_v5.py:10-91`); the base collector invokes the runner, archives arrays, calls `metrics` once per case, and writes lineage (`physics_guided_collect.py:252-300`). `surface_metrics` is reached through `metrics`, and the numeric verifier reaches the same evaluator (`physics_guided_numeric_lineage.py:82-103`).

Unused code does not generate any reported V8 number. The runner imports `correction`, `select`, and `load` without using them. `bounded_update`, `rank_candidate`, and `unit_witness` are outside the V8 path. Planned, correction, replay, selection, and text-motion evaluator branches are not exercised.

## E. Scope: WARN

The actual scope is **four unique clips x three offsets x one seed x one generation draw**, executed twice in separate processes. There are two families (`jump`, `dance`), seed 11 only, round 0 only, and no independent replay batch. The two requests are identical and the generation-noise, prefix, goal, reference, and kinematic hashes match. Repeat B is a process-level measurement repeat, not an independent sample.

The broader three-seed/three-round values in `configs/physics_guided_20261011.json` were not executed here. `versions.json` accurately declares the reduced scope and says the formal comparison is deferred. Any description such as comprehensive, robust, or multi-seed would exceed the evidence.

## F. Evaluation type

- Dataset-derived endpoint/heading conditions: `real_gt` with synthetic offsets and a custom split.
- Generated-plan tracking: `synthetic_proxy` / `self_supervised_proxy` for controller agreement.
- Contact, penetration, continuity, and execution measurements: `simulation_only`.
- Surface slip: explicitly a body-force and collision-primitive surface proxy.
- Semantics: not evaluated by an independent evaluator.
- Human or real-robot evaluation: absent.

## Source, weight, environment, asset, and request binding

All 15 files listed in `versions.json` match their current remote hashes. The upstream CLoSD checkout is at `d8950bdb78312fcd2844f336b3fc079e7368c87f` and has no tracked changes under `closd`. The generator and controller weights match their declared SHA-256 values. All declared external motion and semantic inputs also match. The runtime logs identify Python 3.8.20, Torch 1.13.1+cu117, GPU PhysX, and the expected checkpoints.

Each request binds the current config, manifest, thresholds, and versions hashes. Each execution lineage equals its request and binds `metadata.json`, `metrics.json`, and `trajectories.npz`. The initial-state archive is indirectly bound by its SHA-256 inside bound metadata. The current foot asset hash matches the execution metadata.

There are three provenance limitations:

1. V8 inherits weight/input declarations from the prior version record; it does not independently hash all weights and the environment before each V8 subprocess. Current-state verification corroborates them but is not an execution-time attestation.
2. `resolved_config.yaml`, `receipt.json`, `metric_numerics.json`, `sampling_summary.json`, and final `state.json` are not included in a final output manifest. Their contents were independently checked here, but they are weaker lineage artifacts than the three bound outputs.
3. `source_manifest.json` and `frozen_thresholds.json` retain the inherited embedded `config_sha256` `f7e06e...`, while the execution request binds the current config `2d8fd0...`. The runner binds all three files separately, so the execution request is coherent, but the embedded parent-config field is stale and should be described as an inherited/predeclaration hash.

## Independent reset, fence, and metric checks

The two initial-state archives are byte-identical. For each repeat, all seven pairs `bootstrap_*` versus `first_action_*` have maximum absolute difference zero. The trajectory clock is exactly 0 through 119, all validity entries are true, and all termination entries are false. Metadata records three full reset calls before the first action, one diffusion call, and one generation batch.

The four `_reset_rb_*` cache tensors are not present in `initial_states.npz`; only the reported zero differences are retained. Likewise, the raw artifacts contain no sequence of fence events. Source review confirms the counter increments after immediate blocking fetches, and 242 is plausible for the observed warmup and two simulation calls per control step, but these two claims remain instrumentation assertions rather than independently reconstructable raw measurements.

The pinned upstream order was also checked read-only: simulation is followed by the V8 blocking fetch; upstream refreshes rigid-body and net-contact-force tensors before `_compute_reward`; V8 samples after that refresh. This establishes the intended read order. It does not rescue repeatability: the two force tensors still differ greatly while all recorded kinematics are identical. `contact_mutation_during_reward_max_N = 0` shows only that reward evaluation did not change the tensor; it does not establish contact-force freshness, truth, or reproducibility.

## Claim impact

- **Supported:** both V8 processes completed 12 original attempts with 120 frames each and coherent bound core outputs.
- **Supported:** the seven archived reset-state tensors are unchanged before the first action.
- **Needs qualifier:** four private reset-cache differences are reported as zero but cannot be recomputed from the archive.
- **Needs qualifier:** 242 completion fences are supported by bound code plus a scalar counter, without a raw call trace.
- **Refuted:** contact-force sampling is repeatable at 1 mN.
- **Supported but non-probative:** final classifications are equal across the repeats; all physical and joint classifications are false.
- **Unsupported:** the contact-dependent physical metrics are validated for efficacy comparison.
- **Untested:** V8 improves any method, physical quality, semantic fidelity, downstream utility, or real-robot performance.

## Action items

1. Keep V8 labeled as a simulation-only timing/reset diagnostic and preserve the negative force-repeatability result.
2. Label generated-plan tracking fields explicitly as a synthetic/controller-tracking proxy and correct the collector's overbroad opening statement.
3. Do not use force-gated surface-slip or `physical_pass` from V8 as efficacy evidence.
4. If independent cache/fence certification is needed, archive the four cache tensors and a compact per-simulate/fetch event trace.
5. Add a final manifest binding the resolved config, receipts, numeric certificates, sampling summary, and final state.
6. Explain or replace the inherited embedded config hashes in the manifest and threshold files.

The substantive verdict is same-family and therefore **provisional**. It is not cross-family acceptance.
