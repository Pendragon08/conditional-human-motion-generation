# Experiment Audit Report — V63 Support-Reference Probe

**Date:** 2026-10-12  
**Auditor:** GPT-5.6-Sol, ultra reasoning, fresh same-family reviewer  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Audit cutoff:** V65 preflight and independent continuation witness complete; before V65 `--run`  
**Overall verdict:** **FAIL**

The V63 protocol did not produce a three-arm result by this audit cutoff. Its one documented `--check` invocation completed the baseline replay, but the independent document witness retained `status: FAIL`. Only `val_baseline_s11` existed; the two repair arms, `summary.json`, and every result `analysis_*.json` were absent. A separate numerical-contract addendum and the V65 preflight/witness later passed, but neither item retroactively changes the V63 witness or supplies treatment evidence.

After this pre-run snapshot was frozen, a V65 `--run` was launched. That later execution and all outputs are outside this report. They require a separate post-run append-only audit before any interpretation.

## A–F verdicts

| Check | Status | Finding |
|---|---|---|
| A. Ground-truth provenance | **PASS** | The target is explicitly a simulated/model-derived synthetic proxy, not real MoCap or robot ground truth. |
| B. Score normalization | **PASS** | No metric is normalized by prediction maxima/minima/means. The current unexecuted analyzer reports raw means and finite/missing/total denominators. |
| C. Result existence and lineage | **FAIL** | One of three arms exists; witness is FAIL; no summary or analysis exists; the original V63 continuation gate did not bind the exact witnessed freeze. |
| D. Dead code/execution | **WARN** | Preparation and baseline paths executed. Treatment and analyzer paths did not. The analyzer is omitted from `versions.json`, although a separate pre-arm plan lock now binds its exact source. |
| E. Scope and fairness | **PASS** | Wording is limited to one validation seed and this parameterization. The current analyzer decision uses treatment-versus-baseline physical/joint improvement and excludes efficacy/novelty/G1 claims. |
| F. Evaluation classification | **PASS** | **simulation_only + synthetic_proxy**; no `real_gt`, `human_eval`, or real-world contact truth. |

## Terminal evidence and exact counts

- `DOCUMENT_WITNESS.json:$.status` is `FAIL`; `$.failure` is `AssertionError('applied_reference_canonical_exact')`; `$.all_failed_checks` contains `applied_reference_canonical_exact` and `reference_velocity_successive_exact`.
- The original failure was preserved as `DOCUMENT_WITNESS_INITIAL_FAIL.json/.md`. The expanded witness still has SHA-256 `c8f320671fba8203c74e91fa46080d3318f678adaf046ceff405ba063076a5f5` and remains FAIL.
- At the cutoff, the remote raw directory contained only `val_baseline_s11`. There was one request and one budget row: 50.4387053065 seconds, exit 0. No V63/V65 worker was present during the preflight witness check.
- Baseline receipt: 72 cases, 72 completed, 0 terminated, elapsed 47.52426338196 seconds. Counts are condition 5, physical 0, tracking 68, joint 0.
- Source denominator: 72 cases = 24 sources × 3 offsets; all 72 source inputs were complete and retained. The actual incomplete-input branch was therefore not exercised.
- `state.json` remains `WITNESS_COMPLETE_PENDING_REVIEW`. This is a launcher checkpoint, not a completed-study verdict; the independent witness and missing-arm inventory are authoritative for integrity status.
- V65 `continuation_v65_preflight.json` and `CONTINUATION_DOCUMENT_WITNESS_V65.json/.md` exist and are PASS; they record one `--check`, zero `--run`, zero simulation, and zero generation. At the cutoff, the continuation receipt, treatment requests/raw directories, `summary.json`, and result analysis output were absent.

Evidence: `/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/DOCUMENT_WITNESS.json` at `$.status`, `$.all_failed_checks`, `$.denominators`, `$.bindings`, and `$.receipt`; `DOCUMENT_WITNESS.md:3-25`; `budget_ledger.json:$[0]`; `run_support_reference_probe_v63_20261012.py:63-85`.

## Provenance and evaluation type

Preparation binds V51 `raw/val_shared_s11_r0`, reconstructs the baseline from its simulated `actual` trajectory, and deterministically derives both repair references from saved simulated velocities and force/geometry support proxies ([prepare script, lines 19–49](../../scripts/prepare_support_reference_probe_v63_20261012.py)). The files label this as `simulation_only; synthetic_proxy tracking`, and the protocol states that tracking is relative to a simulated refined synthetic reference rather than MoCap truth ([protocol, lines 3–9](../../docs/support_reference_probe_v63_20261012.md)).

The contact calculation is coherent within that declared proxy scope. Collision primitives are read in body-local coordinates, rotated into world coordinates, adjusted by primitive radius, and reduced to the lowest surface point. Because the saved linear velocity is a COM velocity, point velocity is evaluated as `v_COM + ω × (point − COM)` ([metric source, lines 1–5 and 36–100](../../scripts/physics_guided_metric_v23_20261012.py); [preparation, lines 34–49](../../scripts/prepare_support_reference_probe_v63_20261012.py)). Support is inferred from simulated vertical body force and proximity to ground. This is not an actual contact-manifold measurement or real contact truth.

## Reference construction, order, goals, and bounds

The independent witness verified the manifest-derived text/case/target-joint order, 24 sources with offsets `[-0.15, 0, 0.15]`, and a contiguous valid prefix for every case (`witness_independent_audit.py:53-79`). Preparation resolves foot indices by body name. A robustness gap remains in the preparation source: it writes by metrics-row index before asserting metrics/metadata/trajectory alignment; the later witness caught the actual files, but a future preparation should fail before writing ([preparation, lines 24–44](../../scripts/prepare_support_reference_probe_v63_20261012.py)).

All three prepared reference arrays are finite with shape `120×72×24×3`, and all carry the exact original-goal array. Frames 0–60 and all Z coordinates are exact; horizontal translation is uniform across bodies and remains below 10 cm. Observed maxima were 0.0607537857 m for the unconstrained repair and 0.0607520330 m for the endpoint-preserved repair. The endpoint-preserved final translation is exactly zero. The unconstrained final translation may change by design. These are reference-file properties; only the baseline has been executed. See `DOCUMENT_WITNESS.json:$.reference_invariants`, [core lines 10–37](../../scripts/support_reference_repair_core_v63_20261012.py), and [preparation lines 45–77](../../scripts/prepare_support_reference_probe_v63_20261012.py).

## Reset and time semantics

For the baseline, the initial reset gate passed. Seven exported reset tensors matched between bootstrap and first action, four named rigid-body cache-consistency checks were zero, the initial actual/reference XY anchor error was zero, and reference frames were exactly 0–119 (`DOCUMENT_WITNESS.json:$.runtime_integrity` and `$.checks`).

The temporal claim has a strict limit: at transition index `k`, reference and observation `k` are recorded before action `k`; actual state `k` is recorded after that action completes. The evidence supports the same transition index, not simultaneous pre-action reference/actual sensing (`DOCUMENT_WITNESS.md:25`; [V65 protocol line 19](../../docs/support_reference_continuation_v65_20261012.md)).

The current analyzer now checks each arm’s `initial_states.npz` hash, seven exported reset pairs, reset/cache gates, and exact equality of every exported initial-state array with baseline, while explicitly declining to claim hidden PhysX-state equality ([analyzer lines 28–40](../../scripts/analyze_support_reference_probe_v63_20261012.py)). These checks are sound static guards but remain unobserved until the two repair arms exist.

## Original witness versus numerical addendum

The original witness demanded bitwise zero for the canonical root mapping. It observed one differing scalar among 622,080, with maximum error `1.866893570624082e-16 m`. It also recomputed velocity using serialized float32 `dt=0.03333333507180214`, producing a maximum cross-library difference of `9.5367431640625e-7 m/s`. Those exact checks failed, and their FAIL record remains immutable.

`NUMERICAL_CONTRACT_ADDENDUM.json` is separate evidence with status PASS:

- Root mapping passes the already-frozen verifier contract `atol=1e-12, rtol=0`; no new experimental tolerance was introduced (`$.root_mapping`; frozen verifier `physics_guided_com_lineage_v24_20261012.py:94-96`).
- Saved velocity is exactly reproduced with the executed Torch/CUDA float32 expression `(current-previous)/(1/30)`, including a zero first frame (`$.velocity`).
- All listed inputs remained hash-identical; no simulator, generator, training, repair arm, or baseline rerun occurred.

This addendum used 0.9037324842 seconds of CUDA saved-array arithmetic, recorded in `numerical_adjudication_budget_ledger.json`. That use conflicts with the audit assignment’s literal no-GPU constraint even though no simulation or model execution occurred. It is disclosed as a protocol-scope warning. The addendum does not turn the historical V63 witness into PASS and does not support efficacy.

Hashes: addendum JSON `fc990207cb3dc7c9c5c353394692d562d46a58f5ea4065c19a7798d9d49f8775`; addendum Markdown `dc419d5973f670354eb60b3e6b88048b1500183a96e864144278e4981ecc4f76`; runtime witness `b88031939051092821ca9592229c27163cf316888aae71bce672093cfa8872aa`.

## Analysis and continuation code status

The originally reviewed, never-executed analyzer (`59f16230…`) had two real defects: it silently dropped missing/nonfinite values without denominators, and it computed paired gains but ignored them when selecting its decision. Those defects had no result impact because no analysis file was produced.

The current analyzer is SHA-256 `2a76407d53b7f8fe7dfe823eebb327a5b98b7da746d40dec73e4905150ddc11d` on both hosts. It now reports finite/missing/total counts ([lines 42–48](../../scripts/analyze_support_reference_probe_v63_20261012.py)), checks exported reset equality ([lines 28–40](../../scripts/analyze_support_reference_probe_v63_20261012.py)), and makes the limited decision from positive paired joint gain or physical-pass improvement versus baseline ([lines 52–78](../../scripts/analyze_support_reference_probe_v63_20261012.py)). Remote `analysis_plan_lock.json` was created before any repair arm and binds that source hash; its SHA-256 is `0452de2471d4804226bf7913c3de356483d9784aa16db2fac5afc87beb6c8b27`. The analyzer self-checks the lock at lines 14–20. The analyzer remains absent from the original `versions.json`; preserving and independently binding the plan lock is therefore required.

The original V63 runner’s `--run` gate checks only witness PASS plus the runtime-witness hash, while `validate()` trusts reference hashes stored in mutable `preparation.json` ([runner lines 19–24 and 63–81](../../scripts/run_support_reference_probe_v63_20261012.py)). It does not prove that comparison arms use the exact versions/preparation/references witnessed with baseline.

V65 source is SHA-256 `a801b76906368300b1f4aff021385be931e1d031ed0850c8b3aa6b330f39dbc6` and is byte-identical on both hosts. It closes the prelaunch freeze hole by binding the original FAIL witness, numerical addendum, runtime witness, versions, preparation, all three references, baseline outputs, lineage, and receipt before every repair arm ([V65 lines 15–48 and 61–94](../../scripts/continue_support_reference_probe_v65_20261012.py)). It also accounts for the separate numerical charge when reserving a full 300-second batch timeout. The CPU-only preflight passed with SHA-256 `4e75b64dd6e2e732f6e31968818f2198e1ac7fa4568a1373784285643b66f2d2`; the independent continuation witness passed with SHA-256 `8722339893465e8bd4dfe1db1106dd47363c7db1fd2751850e73d86286784fbd`. Both certify readiness only. At the cutoff they contained zero treatment runs and supplied no efficacy evidence.

## Reproducibility state

Local Git HEAD is `ffadb70f3f4cb7de2537529ba26a66bc6d55ee66`; remote HEAD is `e1ef719d6702d15a5ab211db0e55be438b2d1c88`. The heads differ, and the audited V63/V65 files are untracked on both checkouts. Direct SHA-256 checks establish current byte parity for the source, protocol, witness, addendum, and remediation files, but neither checkout HEAD reproduces them.

Key source hashes:

- core: `376484bd771f653ff6e6132911bf3428b829662ef503a86f19f00519b8f71e26`
- preparation: `bd7b7699dbcbf4d2638130d8c64fbb502eabb90f321bf06c40d2e654d98c9e88`
- original V63 runner: `416f6a7c233a03db01c44100cbe7d8e260d529c142f36bc00720720c2a828f03`
- current analyzer: `2a76407d53b7f8fe7dfe823eebb327a5b98b7da746d40dec73e4905150ddc11d`
- V65 continuation: `a801b76906368300b1f4aff021385be931e1d031ed0850c8b3aa6b330f39dbc6`
- V63 protocol: `33fac131739cdc96cf0fda345ad88d091d045808678f963ed6b644d8fa7f765a`
- V65 protocol: `eb5db0676772205d8d8709595e5196d96be25faf8f9b750fea20c145effd83b5`
- versions: `aad9d6d0707717f6e851032ab078317cf10849e17dbfd2e2670c6cfd58421fbb`
- preparation: `ba4ca14f72136c00c0d127a459e477cdfb91bd91b859170b3d0cab24d1b0ede7`
- baseline metrics: `fb944aefff13d6f6beab1e13ab92bd816620ac9b58a2bec9d03d0e722f99bcba`
- baseline trajectories: `7802cdb99053d49908d19824be38080881a00a4829ff3aba40aeba6e61d81136`
- baseline observation trace: `503f3bd356f54e8fdfc54099ac1873938ac6d8fd075ca2aae2e9ec170287dbe1`

## Reason codes

- `PASS_SYNTHETIC_PROXY_EXPLICITLY_LABELED`
- `PASS_NO_PREDICTION_STATISTIC_NORMALIZATION`
- `FAIL_ORIGINAL_DOCUMENT_WITNESS_RETAINED`
- `FAIL_ONLY_ONE_OF_THREE_ARMS_EXISTS`
- `FAIL_SUMMARY_AND_ANALYSIS_ABSENT`
- `FAIL_ORIGINAL_V63_RUNNER_LACKS_WITNESSED_CROSS_ARM_FREEZE_BINDING`
- `WARN_V65_PREFLIGHT_PASS_BUT_COMPARISON_AND_ANALYZER_PATHS_UNEXECUTED_AT_CUTOFF`
- `WARN_ANALYZER_OMITTED_FROM_VERSIONS_BUT_SEPARATELY_PLAN_LOCKED`
- `WARN_EQUAL_RESET_STATE_ACROSS_ARMS_NOT_YET_OBSERVED`
- `WARN_PREPARATION_ORDER_CHECKS_ARE_POSTHOC`
- `WARN_AUDITED_FILES_UNTRACKED_AND_GIT_HEADS_DIVERGE`
- `WARN_NUMERICAL_ADDENDUM_USED_CUDA_DESPITE_NO_GPU_AUDIT_SCOPE`
- `PASS_SEPARATE_NUMERICAL_CONTRACT_ADJUDICATION`
- `PASS_CURRENT_ANALYZER_STATIC_REMEDIATION`
- `PASS_V62_FAILURE_RETAINED`
- `INFO_V65_RUN_LAUNCHED_AFTER_AUDIT_CUTOFF_NOT_AUDITED`
- `NO_EFFICACY_NOVELTY_OR_G1_CONCLUSION`

## Claim impact

- **Supported:** prepared-reference order/goals/prefix/Z/relative-pose/bound/endpoint invariants; one baseline replay with 72/72 completion and joint 0/72; same-transition-index clock; baseline exported reset/cache gates.
- **Supported only by the separate addendum:** compliance with the pre-existing root-mapping tolerance and exact saved velocity under the executed CUDA `1/30` expression.
- **Unsupported:** bitwise exactness under the original witness’s NumPy checks; treatment physical or joint improvement; a continue/stop conclusion from V63 results; simultaneous reference/actual sensing.
- **Out of scope and unsupported:** novelty, C1/C2/C3 efficacy, G1 readiness, downstream benefit, population-level effect, or real-world physical validity.

## Required actions

1. Keep the original V63 witness FAIL and initial-failure files immutable; cite the numerical addendum separately.
2. Do not describe V63 as a completed comparison. Two arms, the terminal summary, and analysis are absent.
3. Preserve the completed V65 preflight and independent continuation witness bound to the exact V65 source, protocol, and preflight; also preserve the separately locked current analyzer.
4. After execution, enforce and independently verify exact exported initial-state equality and reset/cache gates across all arms; retain the explicit hidden-PhysX-state limitation.
5. Preserve per-metric finite, missing/nonfinite, and total denominators and use the current treatment-versus-baseline decision rule.
6. Independently audit the post-cutoff V65 treatment outputs, summary, analysis hashes, budgets, and local/remote copies before any mechanism decision.
7. Qualify time alignment as the same transition index, not simultaneous states.
8. Archive the untracked files against a reproducible revision and record the local/remote HEAD mismatch.
9. Do not infer efficacy, novelty, C2/C3 validity, G1 readiness, or downstream benefit from this one-seed diagnostic.

The retained V62 failure is also intact: `preparation_failure.json:2-11` records the incorrect CPU fixture assertion with zero generation, simulation, and GPU seconds. V63’s corrected fixture does not rewrite that history.
