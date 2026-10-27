# ReactiveBFM experiment integrity audit v1

## Review metadata

- **Audit root (B):** `C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/reactivebfm_reproduction_20261021/repository/baselines/reactivebfm`
- **Generated:** 2026-10-21T23:18:58.7101456+08:00
- **Reviewer role:** fresh, adversarial, read-only experiment-integrity reviewer
- **Reviewer identity:** `/root/reactivebfm_integrity_audit_v1`
- **review_independence:** `same-family`
- **acceptance_status:** `provisional`
- **Overall verdict:** **WARN**
- **Audit mode:** primary local artifacts were read directly. No GPU job was run, no experiment/report file was changed, no credential was accessed, and the only writes are the two audit outputs in this directory.

The evidence supports a real, bounded compatibility run: official Kungfu motion records were loaded, the full planner executed 101 recorded optimizer updates, parameters changed, checkpoints exist and reload, and the reported diagnostic loss values agree with the receipts. The evidence does **not** support a paper-benchmark reproduction, motion-quality result, control result, long-horizon result, or simulator result. No pseudo-ground-truth reference, prediction-derived score denominator, claimed dead metric path, or phantom scientific benchmark was found.

The overall warning is driven by evaluation scope and auditability: the eight-example diagnostic consumes a deterministic prefix of the test split, exact sample identifiers are not logged, and server-copy equality and publisher-signed source checksums are unavailable to this reviewer. These are limits on the claim ceiling; they do not negate the observed training execution.

## A–F decision table

| Check | Status | Decision |
|---|---|---|
| A. Ground-truth provenance | **PASS** | The diagnostic is rooted in dataset-provided test motions (`real_gt`), with an analytic noise-derived flow target. No model output is used as the reference. Public-source provenance is recorded, although no publisher-signed checksum exists. |
| B. Score normalization | **PASS** | Loss denominators depend on the mask and tensor dimensions, not on predictions. The reported held-out number is an arithmetic mean of eight per-sample losses. |
| C. Claims, files, and tracker consistency | **WARN** | Local receipts, logs, metrics, hashes, settings, checkpoints, and tracker states agree. Exact held-out sample IDs are absent, and remote server/LFS equality was not independently available. |
| D. Metric implementation and called paths | **PASS** | The reported loss and auxiliary MSE metrics are on the executed flow-training path and appear in the recorded metric stream. No claimed metric is dead code. |
| E. Scope adequacy | **WARN** | Adequate for a bounded compatibility/diagnostic milestone only: one seed, one trained dataset, 101 updates, eight selected test clips, no uncertainty, no comparison, no complete test evaluation, no rollout-quality test, and no physics. |
| F. Evaluation type | **PASS** | Evidence types and claim boundaries are stated explicitly below. The only quantitative model comparison is a diagnostic flow-objective loss, not a paper benchmark. |

## A. Ground-truth provenance — PASS

1. The evaluation wrapper explicitly constructs the held-out loader with `dataset='kungfu_g1_36dim'`, `split='test'`, `shuffle=False`, and `deterministic=True`, then takes four batches of size two (`scripts/run_bounded_pilot.py:78-82`). It passes the first 25 frames as the real prefix and frames 25–74 as `x_start` to the official objective (`scripts/run_bounded_pilot.py:64-70`).
2. The official loader resolves a `MotionTextDataset` for the requested split (`upstream/reactivebfm/data/datasets/dataset.py:100-153`), reads `full_train.pkl` (`upstream/reactivebfm/data/datasets/motion_text.py:89-101`), filters names through the split file (`upstream/reactivebfm/data/datasets/motion_text.py:107-109`), and loads `Mean.npy`/`Std.npy` and the requested split (`upstream/reactivebfm/data/datasets/motion_text.py:376-392`).
3. The objective creates `target_velocity = noise - x_start`, compares the model output against it, and reconstructs `x0_pred` only after that comparison (`upstream/reactivebfm/model/motion_planner/objectives/flow/flow_matching_smooth.py:162-174`). Thus the target is analytically derived from real motion plus sampled noise; it is not a pseudo-label from the model under test.
4. The upstream README identifies the released train/val/test data and its public source (`upstream/README.md:95-108`). The Kungfu file receipt records the URL, byte count, and local SHA256 (`results/acquisition_v1/acquisition.json:130-138`). This audit independently rehashed all 24 local data files and found all sizes and digests equal to the acquisition record; the Kungfu `full_train.pkl` digest is `2fddadcd7361d3333f18cdcc556bcb0fc2d1ee30cc2fe047e191e23382a25eb5`.
5. The numeric audit records 893 Kungfu motions, 547,557 frames, 803/46/44 split counts, finite 36D arrays, and no exact key or byte-identical motion overlap across splits (`results/data_audit_v1.json:184-211`). Its checks are implemented against the deserialized arrays and split files (`scripts/audit_official_data.py:21-53`).

**Evidence classification:** `real_gt` for dataset motion. The Gaussian/noise interpolation is an algorithmic training construction and does not make the reference model-derived. The absence of a publisher-signed checksum is correctly disclosed in `README.md:27-32` and `reports/ReactiveBFM复现实验报告.md:39-40`; provenance therefore rests on the official listing URL plus independently reproduced local size/hash, not a signed release digest.

## B. Score normalization — PASS

1. Input motion is normalized with fixed dataset `Mean.npy` and `Std.npy` (`upstream/reactivebfm/data/datasets/motion_text.py:254-260`, `upstream/reactivebfm/data/datasets/motion_text.py:376-387`). Those arrays are data artifacts, not statistics computed from the evaluated predictions.
2. The core loss computes squared error and divides by the count of unmasked entries, derived from the mask and channel dimensions (`upstream/reactivebfm/utils/training/losses.py:14-31`). No prediction magnitude, prediction variance, best-of-N choice, or prediction-derived denominator is used.
3. The flow loss compares `target_velocity` with `model_output` (`upstream/reactivebfm/model/motion_planner/objectives/flow/flow_matching_smooth.py:162-172`). The wrapper then returns the ordinary mean of the eight detached sample losses (`scripts/run_bounded_pilot.py:71-75`).
4. `grad_norm` near one is explained by configured gradient clipping (`results/pilot_v1/args.json:50`; `upstream/reactivebfm/model/motion_planner/objectives/diffusion/fp16_util.py:227-238`) and is unrelated to evaluation-score normalization.

**Finding:** there is no prediction-derived denominator and no score rescaling that could mechanically make weak predictions look better.

## C. Claims, files, and tracker consistency — WARN

### Consistent local evidence

- The run receipt reports 101 optimizer updates, 202 prefix-selection calls, 169 selected samples, and explicitly marks paper benchmarks, physics evaluation, and full training false (`results/pilot_v1/run_receipt.json:2-12`). Its command records flow mode, 101 steps, warmup 25, batch 2, seed 42, `cross_prob=0.1`, and no EMA (`results/pilot_v1/run_receipt.json:13-45`), consistent with the saved arguments (`results/pilot_v1/args.json:8-16`, `results/pilot_v1/args.json:43-84`).
- The before/after means and all eight component values are present (`results/pilot_v1/run_receipt.json:48-78`), and the receipt labels the protocol as a teacher-forced official flow objective rather than a motion-quality score (`results/pilot_v1/run_receipt.json:60-62`, `results/pilot_v1/run_receipt.json:76-78`).
- All 101 lines of `results/pilot_v1/training_metrics.jsonl:1-101` parsed as JSON with a common metric schema, finite recorded losses, `official_step` 0–100, and `optimizer_updates_seen` 1–101. The stdout independently shows the resolved loader/model settings (`results/pilot_v1/pilot_v1.stdout.txt:1-31`), final step 100 and save (`results/pilot_v1/pilot_v1.stdout.txt:2855-2860`), and the complete final receipt (`results/pilot_v1/pilot_v1.stdout.txt:2861`). The stderr contains progress output and no traceback.
- The checkpoint manifest gives exact sizes and SHA256 values (`results/pilot_v1/checkpoint_manifest.json:3-14`). Independent local rehashing matched both: model `0800452b89d765569d8d1457a5f49a8042d9d8a2c12f7c65145ccea411f291d2`, optimizer `91fee6849637282cb490bb20d370930934473e9d9729f3ccf747a1728ec211ac`.
- The reload receipt matches the training-end parameter hash, records 365 optimizer states all at step 101, and produces a finite `[1,36,1,50]` tensor (`results/checkpoint_reload_v1/receipt.json:2-20`). The tracker receipt limits itself to CPU reconstruction and qpos-policy forward (`results/tracker_cpu_v1/receipt.json:2-6`).
- The experiment tracker marks the bounded evidence tasks done while leaving full training, MuJoCo closed loop, and paper benchmarks as TODO (`results/EXPERIMENT_TRACKER.json:3-15`). A late metadata-only update changed `github_backup` at line 12 from `IN_PROGRESS` to `DONE`; it is outside this scientific-evidence verdict and did not change any scientific status.
- The report reproduces the receipt values and limitations rather than converting them into paper results (`reports/ReactiveBFM复现实验报告.md:73-95`, `reports/ReactiveBFM复现实验报告.md:99-112`, `reports/ReactiveBFM复现实验报告.md:127-135`). No phantom benchmark number was found.

### Warning conditions

1. **Exact diagnostic identities are not recorded.** `get_dataset_loader` defaults `return_keys=False` (`upstream/reactivebfm/data/datasets/dataset.py:100-126`), and the wrapper does not override it (`scripts/run_bounded_pilot.py:78-80`). The receipt contains losses but no motion keys (`results/pilot_v1/run_receipt.json:48-78`). The sample set is reconstructable from the pinned loader/data, but the receipt alone is not self-identifying.
2. **Remote-copy equality is unavailable.** The checkpoint manifest provides one digest plus local and server paths (`results/pilot_v1/checkpoint_manifest.json:3-14`); this audit could rehash only the local files. The report statement that local and server files are byte-identical (`reports/ReactiveBFM复现实验报告.md:101-107`) therefore remains supported by the retained record but was not independently re-established. The same limitation applies to the published-LFS equality assertion in `results/tracker_acquisition_v1.json:2-6` and `reports/ReactiveBFM复现实验报告.md:110`.
3. **The data-audit script hard-codes one provenance-like flag.** `normalization_preserved_from_official_release` is assigned `True` rather than calculated (`scripts/audit_official_data.py:47-55`). Actual local Mean/Std hashes match the acquisition receipt, but that boolean alone is not independent proof of publisher identity.

## D. Metric implementation and called paths — PASS

1. The entrypoint creates the requested flow process and invokes `ScheduledForcingLoop.run_loop()` (`upstream/reactivebfm/train/train_planner_scheduled_forcing.py:38-52`). The factory selects `FlowMatchingSmoothStandard` when `model_type='flow'` (`upstream/reactivebfm/model/motion_planner/factory.py:18-28`, `upstream/reactivebfm/model/motion_planner/factory.py:132-148`).
2. The training loop calls `self.diffusion.training_losses`, backpropagates `terms['loss']`, retains returned model output for self-rollout, and accumulates every other term for logging (`upstream/reactivebfm/train/scheduled_forcing_loop.py:547-605`). It forwards those values to the run’s metric sink (`upstream/reactivebfm/train/scheduled_forcing_loop.py:637-658`).
3. The flow objective computes `flow_mse`, composes `loss`, and calls `masked_motion_metrics` (`upstream/reactivebfm/model/motion_planner/objectives/flow/flow_matching_smooth.py:162-172`, `upstream/reactivebfm/model/motion_planner/objectives/flow/flow_matching_smooth.py:225-240`). The auxiliary metrics are implemented in `upstream/reactivebfm/utils/training/losses.py:38-115` and occur in every recorded metric row.
4. The held-out path directly calls the same official `training_losses` implementation before and after training (`scripts/run_bounded_pilot.py:59-85`).

`angle_l2` in `upstream/reactivebfm/utils/training/losses.py:9-12` is an unused generic helper, but no result or claim relies on it. No claimed metric was found on a dead path.

## E. Scope adequacy — WARN

- The saved configuration is one seed, batch size two, one dataset, 101 steps, and three primitives (`results/pilot_v1/args.json:8-16`, `results/pilot_v1/args.json:53-60`, `results/pilot_v1/args.json:75-84`). The receipt explicitly marks full training, physics, and paper benchmarks false (`results/pilot_v1/run_receipt.json:9-11`).
- The eight diagnostic clips are not a random or representative test sample. The dataset sorts eligible motions by length when required (`upstream/reactivebfm/data/datasets/motion_text.py:141-143`, `upstream/reactivebfm/data/datasets/motion_text.py:153-159`); deterministic access selects the first caption and center crop (`upstream/reactivebfm/data/datasets/motion_text.py:207-218`, `upstream/reactivebfm/data/datasets/motion_text.py:247-260`); the wrapper disables shuffle and takes the first four batches (`scripts/run_bounded_pilot.py:78-82`). They are therefore a deterministic sequential subset after minimum-length filtering, with length-order selection pressure.
- The test split was viewed before and after training. Although the values are not backpropagated and no evidence shows hyperparameter tuning from them, these eight examples are no longer untouched for any future final test claim. A future benchmark should freeze a separate validation protocol and evaluate an untouched test set once.
- There is no repeated seed, confidence interval, baseline comparison, complete test-set aggregation, semantic metric, generated-rollout quality assessment, long-horizon stability test, MuJoCo feedback loop, perturbation experiment, or real-robot evaluation.
- The documents state this boundary accurately (`README.md:3-5`, `README.md:55-68`; `reports/ReactiveBFM复现实验报告.md:7-10`, `reports/ReactiveBFM复现实验报告.md:87-93`, `reports/ReactiveBFM复现实验报告.md:127-135`). Scope is therefore adequate for the stated compatibility milestone and inadequate for efficacy or paper-performance claims.

## F. Evaluation classification — PASS

| Artifact/result | Evidence type | Scientific meaning |
|---|---|---|
| Numeric dataset audit | `real_gt` | Data-integrity diagnostics only; no model performance. |
| Held-out loss, 1.7152809352 → 1.3540775850 | `real_gt` + diagnostic loss | Paired, teacher-forced official flow-objective values on eight real dataset clips with a fixed noise seed. Not a motion-quality or paper benchmark. |
| Per-step training losses and auxiliary MSEs | `real_gt` training diagnostics | In-sample optimization traces; not evaluation benchmarks. |
| Runtime witness | `execution-only` | Full-model CUDA forward/backward and finite gradients; optimizer updates explicitly zero (`results/document_witness_v1/witness_document_v1.json:21-34`). |
| Checkpoint reload/forward | `execution-only` | Serialization and finite CPU forward only (`results/checkpoint_reload_v1/receipt.json:2-20`). |
| Tracker CPU dry run | `execution-only` | Checkpoint reconstruction and qpos-policy forward, with no physics or TensorRT (`results/tracker_cpu_v1/receipt.json:2-6`). |
| Paper FID/MM-Dist/R@3, control, perturbation, and tracking metrics | unavailable / not run | No paper benchmark result exists in the audited artifacts. |

No `synthetic_proxy`, `self_supervised_proxy`, `simulation_only`, or `human_eval` result was used to support the reported milestone. The flow objective includes sampled noise, but its motion source is `real_gt` and its result is classified only as a diagnostic loss.

## Claim impacts

### Supported at the audited scope

- A real official Kungfu batch reached the full planner, official loss, backward pass, and optimizer path for 101 updates; model parameters changed (`results/pilot_v1/run_receipt.json:2-8`, `results/pilot_v1/run_receipt.json:80-87`).
- Scheduled prefix replacement and RTC-related paths executed under the saved configuration; the counter evidence is consistent with logs and receipt (`results/pilot_v1/run_receipt.json:4-6`, `results/pilot_v1/args.json:70-81`).
- The local checkpoints exist at the recorded sizes and hashes and the retained reload evidence shows a usable checkpoint with step-101 optimizer state (`results/pilot_v1/checkpoint_manifest.json:3-14`, `results/checkpoint_reload_v1/receipt.json:2-20`).
- All 226 files named by `upstream_manifest.json` were independently rehashed in place with zero mismatches. The manifest itself hashes to `a5b29ac3ec9d35d992dbb7ac590f6f1c44b5aedea8c462f7ca143820c8f9dc9`.

### Supported only with a diagnostic qualifier

- The decrease from 1.7152809352 to 1.3540775850 means the fixed eight-example, teacher-forced flow objective decreased in this run. It does not establish perceptual quality, semantic alignment, generalization, long-horizon stability, or physical executability.
- The data audit supports finite 36D arrays, exact split-key separation, exact-array separation, and the reported aggregate counts. It does not establish subject isolation, source-clip isolation, near-duplicate absence, or physical feasibility; the report says so at `reports/ReactiveBFM复现实验报告.md:49-54`.

### Unsupported or unavailable

- Any claim of reproduced FID, MM-Dist, R@3, tracking error, control success, perturbation robustness, survival, MuJoCo closed-loop behavior, TensorRT-engine execution, real-robot performance, full mixed-data training, or paper-level quality. The tracker correctly leaves the relevant work TODO (`results/EXPERIMENT_TRACKER.json:13-15`).
- Independent confirmation of server/local checkpoint equality and the remote published-LFS object was unavailable. This is recorded as unavailable, not inferred as success.
- Publisher-signed checksums for the released data were unavailable. Local bytes match the project’s acquisition records, but those records are not a cryptographic attestation from the publisher.

The current README and report do not assert the unsupported scientific results; they repeatedly distinguish the diagnostic milestone from paper benchmarks.

## Input availability and immutable-read note

- The specifically listed path `upstream/reactivebfm/data/datasets.py` does not exist. The loader is a package. This audit read the discovered subordinate paths `upstream/reactivebfm/data/datasets/{__init__.py,registry.py,dataset.py,motion_text.py,batch.py}` instead. The missing literal path is recorded rather than treated as a successful input.
- The server paths named in local records were unavailable to this reviewer. No conclusion was inferred from their absence.
- No `upstream/evaluation/` implementation exists in the local snapshot; `upstream/README.md:90-92` describes evaluation tools as planned for later release.
- README final-read SHA256: `4b4d03ec2cbe596f5c01ebcc0df7b50373c856e7105f6bf4aca8f43fd9ae8765` (4,035 bytes).
- Report final-read SHA256: `42e72441b2abb2cba4deb852935f830f4703f22a4febc0524a351ec2046eb374` (10,772 bytes).
- The two document hashes were unchanged between the initial audit read and finalization. Complete input and primary-artifact hashes are in `EXPERIMENT_AUDIT.json`.
- `results/EXPERIMENT_TRACKER.json` was observed first at SHA256 `fcc377061aaa6f163fcde3b1dbe7873e46a83f8f920d47c4a60eee80a0048a5f` (503 bytes, `github_backup=IN_PROGRESS`) and finalized at SHA256 `a7d8ffc4ba2a4fa373ae14615dd840a907dcaf5011607face9cd87640f0868e2` (496 bytes, `github_backup=DONE`). All scientific task states were identical across the two snapshots.

## Final acceptance statement

**acceptance_status: provisional.** Accept the artifacts as evidence of a bounded, real-data execution and checkpoint-usability milestone. Do not promote the result to a paper benchmark, motion-quality evaluation, or physical-control result. Resolve the warnings by logging exact sample keys, reserving an untouched final test set, running the published evaluation protocol when available, adding repeated seeds/comparators, and retaining independently checkable remote-object evidence.
