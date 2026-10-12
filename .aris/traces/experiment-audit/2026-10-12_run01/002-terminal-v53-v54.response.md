## Integrity audit verdict

**Overall: FAIL.** The frozen V51 evaluation fails the mandatory provenance rule because a model/simulator-derived trajectory was used as the tracking reference without an explicit proxy label. V53 and V54 improve the current disclosure, but they cannot retroactively repair the original record. V54 also leaves the selection JSON outside execution-time binding.

Reviewer metadata:

- model: `gpt-5.6-sol`
- family: `openai`
- reasoning: `ultra`
- review_independence: `same-family`
- acceptance_status: `provisional`

### A–F assessment

| Area | Status | Finding |
|---|---|---|
| A. Ground-truth provenance | **FAIL** | Frozen V51 calls the simulator-refined candidate `reference` and only labels it `physics_refined`; it was not explicitly labeled a synthetic proxy. |
| B. Score normalization | **PASS** | Primary denominators are fixed at 216/method and 72/seed. No prediction mean/max/min supplies a denominator. Raw counts and finite coverage accompany summaries. |
| C. Existence and counts | **FAIL** | Core V51 terminal files and all counts pass, but the specifically requested remote V54 invocation log is absent. Local and remote Git heads also differ. |
| D. Dead code / invocation | **WARN** | Main metric functions run and appear in outputs, but the imported comparison function and V23-local ranker are dead on the V51 path. |
| E. Scope | **WARN** | Actual scope is verified, but `versions.json` retains contradictory V28 test/44/30 metadata; the custom “val” split mixes official train/test rows. V53 also used 10,000 bootstraps against the configured 5,000. |
| F. Evaluation classification | **PASS** for current disclosure | Composite evidence is `simulation_only`; tracking is `synthetic_proxy`; feedback is a `self_supervised_proxy`; only the condition target is dataset-derived. No human or real-robot evaluation occurred. |

## A. Provenance

The mandatory failure is in the original frozen V51 path:

- `D:\具身智能\conditional-human-motion-generation\scripts\physics_guided_queue.py:144-155` chooses a candidate and, unless `raw=True`, copies `arrays['actual']` into the selected NPZ key named `reference`. It labels this only as `source_type='physics_refined'`.
- V51 takes this default path at `D:\具身智能\conditional-human-motion-generation\scripts\run_paired_feedback_validation_v51_20261012.py:161-163`.
- `D:\具身智能\conditional-human-motion-generation\scripts\physics_guided_metric_v23_20261012.py:118,136-160` compares `actual` against that `reference` and turns the result into `tracking_pass` and `joint_pass`, without a proxy qualifier.
- Surface/contact handling is correctly labeled as a proxy at `physics_guided_metric_v23_20261012.py:93-100`.

The later provenance document is accurate but prospective:

- `D:\具身智能\conditional-human-motion-generation\docs\paired_feedback_v51_evaluation_provenance_20261012.md:3` explicitly says it was appended after the audit finding and cannot prove prior adequate labeling.
- Lines `7-11` classify the condition target, synthetic tracking reference, simulation-only physical metrics, and joint conjunction.
- Line `15` limits the new analysis to 24 validation sources and denies independent-test, G1, downstream, or innovation evidence.
- SHA-256: `a160b2b8c0644996f08e9a483903e61f59e529b87b0255d436f28ab8cb5bc4ba`.

Current V53/V54 presentation is correctly labeled at:

- `D:\具身智能\conditional-human-motion-generation\scripts\analyze_paired_feedback_v51_20261012.py:117-125`
- V54 result `analysis_1791770205879251102.json:240-249`

An official text-motion evaluator implementation and weights are available in `versions.json:31,44,51-56`, but they were not run. V53/V54 explicitly record the semantic/diversity guard as incomplete.

## B. Normalization

No self-normalization defect was found:

- `analyze_paired_feedback_v51_20261012.py:72-89` uses raw binary outcomes, exactly 216 observations per method, nine observations per source, and `/72` for seed-specific percentage-point differences.
- Lines `106-111` report raw continuous means together with `recorded_finite` and `missing_or_nonfinite`; all fields have 216/216 finite coverage in each arm.
- Candidate ranking at `physics_guided_metric_v23_20261012.py:164-171` divides errors by frozen thresholds, not by prediction extrema or prediction means.
- `endpoint_time_error_s` at metric lines `124-131` localizes an event using the predicted trajectory; this is an interpretation caveat, not a prediction-derived denominator.
- No suspicious near-1 normalized score was used as the primary result.

## C. Terminal artifacts and counts

The primary remote V51 tree is internally complete:

- PID `2851799`: **absent** on final process inspection.
- `state.json`: `COMPLETE_VALIDATION_QUALITY_FLOOR`.
- Terminal summary plus exactly 21 ledger entries: **YES**.
- Ledger: 21 unique expected labels, all exit code 0, 20 comparison entries plus one measurement entry, totaling `1049.6597477272153` seconds.
- Requests: 21 files.
- Raw: 21 directories × 10 files = **210 remote files**.
- Selected: six arms × JSON/lineage/NPZ = **18 remote files**.
- Construction/replay batches: **15 / 6**.
- The local metadata mirror intentionally has 105 raw JSON files and 12 selected JSON/lineage files; the remote tree contains the binary NPZ payloads.

Key hashes:

- `state.json`: `84494fd005409e3cc23761ae16f048c7ef81c2078c713c767e952e0046e7d78b`
- `summary.json`: `5a05ad6b27aff2043f9e8c7b54f3a54acc242bbb8be7adb8e384b5222c99adca`
- `budget_ledger.json`: `157824fe767267d853e5b1c5168e2785320e08da5daed5e65990e08b04193613`
- `versions.json`: `d866b631cf5eff7480869f75ce4696ed1312524be47e6222cc7c6080d0a91948`
- source manifest: `63c6766c41a9d6045c30cbe5bf92e76bb25abde7b8a3795be11c40603a725307`
- frozen thresholds: `451fae0a2b106dbd0170185d9a22bb6f3fe2583709e77979eb017cc3ce09315b`
- runtime witness: `6ad22044c8392a8cc679d03a56b99185dd55ab1be77846c1498fb756ebe379f1`
- document witness: `48c0a401bd3c2e25da8d15de826514d59f3c8feb98a2827e94fcec1298f150ed`
- launcher receipt: `3046e6094ead4485567bdf1dc43cf27f98a8194e2ee37b9679a0264a4e86ef11`

Every request hash, execution-bound lineage, receipt, metadata/metrics/trajectory hash, case count, and replay count matched the remote primary files.

Raw array schemas were consistent across all 21 batches:

- positions/reference/linear velocity/angular velocity/contact force: `(120,72,24,3)`
- quaternion: `(120,72,24,4)`
- valid mask: `(120,72)`
- terminated: `(72)`
- reference frame: `(120)`

No trajectory array contained a reward or legacy-cache key. Sidecar hashes for initial state, observation trace, sensor diagnostic, and native contacts matched metadata. All `native_contacts.json` files had empty records/errors/dtype fields. This establishes absence in the inspected outputs, not a general claim about uninspected runtime memory.

The requested remote file

`/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/paired_feedback_analysis_v53_20261012/v54_invocation_20261012.log`

does **not** exist. Its local copy exists with SHA-256 `575a90ecdc6c7bd57aa292c6d611a3f3802c3c8d6dfc607d68735d3606e3b28d`. This is why C is not a full pass.

## Pairing, candidate budget, and completion guards

All requested mechanics were independently confirmed:

- Ordinal 0 is shared by both arms at runner lines `152-155`.
- Ordinals 1–2 use matching candidate-ordinal noise, with equality asserted at runner lines `155-160`.
- Feedback/random metadata matched for `generation_noise_sha256`, original goals, source IDs, families, offsets, and headings.
- Original goals and cases remained unchanged.
- Each method has three candidates per input.
- The selected candidate ordinal distributions were:
  - feedback: seed 11 `15/24/33`, seed 12 `18/17/37`, seed 13 `12/29/31`
  - random: seed 11 `23/24/25`, seed 12 `29/23/20`, seed 13 `22/26/24`
- Every selected NPZ reconstructed the chosen bound candidate trajectory, including terminal-frame padding. Replay references matched the selected canonical motions to numerical roundoff, maximum error about `1.87e-16`.

Construction and replay must remain distinct:

| Arm | Construction complete | Replay completed | Replay joint pass | End-to-end joint pass |
|---|---:|---:|---:|---:|
| Feedback | 214 | 215 | 0 | 0 |
| Random | 213 | 214 | 0 | 0 |

The main V51 summary reports replay totals at runner lines `169-178`. V53/V54 separately define `end_to_end_joint_pass = construction_complete AND replay joint_pass` at analyzer lines `56-60`.

## D. Dead paths

- `run_paired_feedback_validation_v51_20261012.py:126` imports `compare`, but `main()` never calls it.
- `physics_guided_metric_v23_20261012.py:164-171` defines `rank_candidate`, but V51 queueing imports and calls the copy from `physics_guided_core.py` instead (`physics_guided_queue.py:12,144`).
- V23’s script-only unit witness is not part of the V51 execution path.

The actual evaluators are live: tracking, condition, physical, completion, and joint fields are created at metric lines `118-160` and appear in every raw/replay metrics record.

## E. Scope

Verified actual scope:

- two families: 12 jump and 12 dance sources
- custom validation set: 24 sources
- offsets: `-0.15, 0, +0.15`
- seeds: `11, 12, 13`
- nine rows per source per arm
- 216 inputs per arm
- one frozen configuration
- 15 construction and six independent replay batches
- no independent test, G1, downstream training, or multi-configuration sweep

The custom `split="val"` rows comprise **19 HumanML3D official-train rows and five official-test rows; zero official-val rows**. Therefore “validation” means the project’s custom split, not the official HumanML3D validation split.

`versions.json` carries contradictory inherited V28 fields:

- lines `66-68`: 44 construction, 30 replay, 72 evaluation inputs
- lines `78,84`: 432 test inputs and 48 test sources
- line `87`: “complete paired test”
- lines `99-106`: the correct V51 24-source, 15/6, validation-only scope

The V51 document itself is accurate at `docs/paired_feedback_validation_v51_20261012.md:3-7,25`.

V53 used 10,000 bootstrap samples at analyzer lines `84,114`, while the frozen configuration specifies 5,000 at `configs/physics_guided_20261011.json:38`. V54 corrects the current analysis to 5,000.

## F. Classification

| Component | Classification |
|---|---|
| Condition target | `real_gt` only in the narrow dataset-derived sense: HumanML3D source motion plus a predeclared offset; it is not robot-execution GT or an official benchmark endpoint |
| Tracking | `synthetic_proxy` |
| Feedback/correction mechanism | `self_supervised_proxy` using prior simulated execution error |
| Physical metrics | `simulation_only`; surface slip/contact are proxies |
| Composite joint outcome | `simulation_only` |
| Human evaluation | absent |
| Real-robot evaluation | absent |

## V53 status

Current V53 source SHA-256:

`8f99e1901ca091003dc431f902c525eaa877877736982447311a677278be8eec`

Current V53 output SHA-256:

`30d61febbcacf0c7e123bc42d8e1feefd5e4640e890b61ac67ef337849a540db`

The output correctly reports:

- feedback totals: completed 215, condition 63, physical 0, tracking 203, joint 0, end-to-end joint 0
- random totals: completed 214, condition 39, physical 0, tracking 201, joint 0, end-to-end joint 0
- primary gain: `0.0` percentage points
- source-cluster 95% interval: `[0.0, 0.0]`
- numeric gate: false
- semantic/diversity guard: incomplete
- acceptance gate: not established

Historical V53 gap: `analyze_paired_feedback_v51_20261012.py:44-60` rereads current metrics and selection JSON but does not compare them to execution-time replay/selection bindings. Its input hashes are a current snapshot, not an execution-time trust check.

## V54 remediation status: partial

V54 source SHA-256:

`8e3762c5c381117461c0d578550148ba3228c78d327b3c3daa1aa2375383b8dc`

Local and remote outputs match:

- analysis: `bebec154baf929d372cd30dcfe3d8011a1d57e1b510950004031c28f6a895b80`
- execution certificate: `e0049f5c7be60a7b06eaf5fe63f91c7a7fe416d973d05331e823af2b8890a9ab`

V54 successfully adds:

- parent-source pinning at lines `16-18`
- config, manifest, and threshold checks at `22-24`
- all 21 ledger/raw label checks at `25-39`
- `guard_batch` verification for every batch
- replay summary-record equality at `42-46`
- selected NPZ motion binding to the replay request at `47-50`
- candidate-output hash checks at `51-56`
- explicit 15/6/24/72 scope checks at `58-60`
- configured 5,000 bootstrap repetitions and configured gain threshold at `72,79-81`
- preservation warnings at `61,68-71`

Its result retains all V53 totals, error means, and component counts exactly; only bootstrap count changes from 10,000 to the configured 5,000. The all-zero interval remains `[0,0]`.

V54 does **not** fully bind the selected JSON to execution time. At lines `48-50`, it trusts the mutable `selected/*.lineage.json`; that file is not itself bound by the replay request or summary. Someone could change `choices[*].construction_complete` and update `selection_lineage['selection_json']` while preserving the execution-bound selected NPZ, candidate outputs, and replay outputs. V54 would still pass. The current artifacts independently matched the bound candidate records and selected motions, and all replay joint outcomes are zero, so this gap does not change the present numeric result. It prevents treating V54’s certificate alone as a fail-closed proof of the construction guard.

V54 also records the stale V28 scope as a warning rather than correcting the original metadata, and it still does not assert the exact 12/12 family composition or exact offset set.

## Git provenance

The earlier “all V53 files untracked” observation is superseded locally:

- local HEAD: `f0e3c4014428dedcc3ce150e6069af4837149cd8`
- local tracked and clean: V53 source, provenance document, and V53 result
- local untracked: V54 source, V54 result/certificate, and `AUDITED_INPUT_HASHES.json`

The remote execution checkout is different:

- remote HEAD: `e1ef719d6702d15a5ab211db0e55be438b2d1c88`
- remote V53 source/document/result and all V54 source/results are untracked
- remote and local HEADs therefore do **not** identify the same repository state

File hashes provide the usable lineage here; the remote result must not be described as reproduced directly from remote `HEAD`.

## Reason codes

- `A_V51_SYNTHETIC_REFERENCE_NOT_PROXY_LABELED` — **FAIL**
- `C_REMOTE_V54_INVOCATION_LOG_MISSING` — **FAIL**
- `V54_SELECTION_JSON_NOT_EXECUTION_BOUND` — **FAIL-CLOSED GAP**
- `V53_SNAPSHOT_ONLY_LINEAGE` — historical warning, partially remediated by V54
- `E_INHERITED_V28_SCOPE_CONFLICT` — warning
- `E_CUSTOM_VAL_NOT_OFFICIAL_VAL` — warning
- `E_V53_BOOTSTRAP_CONFIG_DRIFT` — historical warning, corrected in V54
- `D_DEAD_COMPARE_AND_DUPLICATE_RANKER` — warning
- `GIT_REMOTE_HEAD_MISMATCH` — warning
- `V51_EXISTING_OUTPUT_RESUME_GUARD_INCOMPLETE` — warning preserved by V54
- `RESULT_ALL_ZERO_JOINT_QUALITY_FLOOR` — result limitation, not a provenance defect

## Claim impacts

- **Supported:** exact paired-noise mechanics, unchanged original goals, equal three-candidate budget, 24×3×3 scope, 15 construction/6 replay batches, and current raw/replay counts.
- **Supported descriptively:** feedback has 63 condition passes versus 39 for random in this custom validation pilot.
- **Unsupported:** an accepted efficacy improvement. Both arms have zero replay joint pass and zero end-to-end joint pass; gain is zero.
- **Unsupported:** semantic quality, diversity preservation, real physical executability, real contact/slip accuracy, downstream benefit, G1 success, or innovation acceptance.
- **Unsupported:** calling the original V51 tracking evaluator ground truth or claiming that its original proxy labeling was adequate.
- **Supported prospectively:** V53/V54 now describe tracking as synthetic proxy and physical evidence as simulation-only.
- **Provisionally supported:** current V54 values match the independently inspected artifact tree, despite the remaining selected-JSON certificate gap.
- **Not established:** population-level absence of effect. The `[0,0]` interval is a floor caused by all-zero joint outcomes on these 24 custom validation sources.

## Required actions

1. Preserve the original V51 labeling failure in the permanent audit record; do not rewrite the frozen files.
2. Add an append-only analysis that reconstructs each selected candidate and its `construction_complete` value directly from execution-bound candidate metrics and the replay-bound selected NPZ. Do not trust a mutable selection-lineage JSON alone.
3. Record or copy the V54 invocation log to the claimed remote path, with an explicit exit-status receipt and hash.
4. Assert exact families, offsets, seeds, source multiplicities, and custom-versus-official split composition in the analysis.
5. Continue to expose the stale V28 fields as historical metadata, or emit a normalized append-only scope record; do not silently edit the original.
6. Commit or otherwise archive the exact remote working-tree sources and outputs so the execution state is recoverable independently of local HEAD.
7. Keep acceptance false until independent semantic/diversity evaluation and an appropriately separated test are completed.
