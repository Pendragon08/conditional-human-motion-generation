# Contact Sampling V5 Experiment Integrity Audit

**Date:** 2026-10-11  
**Auditor:** fresh Codex audit agent, same-family review  
**Review status:** provisional  
**Mode:** read-only remote inspection and CPU recomputation; no collector or GPU run; no remote mutation

## Overall verdict: WARN

- **Core execution artifact integrity: PASS.** The V5 requests, frozen inputs, current source hashes, output hashes, receipts, initial-state archives, and saved metrics are internally consistent.
- **Derived-summary traceability: WARN.** `scripts/analyze_contact_repeats_20261011.py` is not listed in `versions.json`, and `repeatability_summary.json` has no execution-lineage binding. Independent recomputation nevertheless reproduced every reported comparison value.
- **Supporting force/slip measurement validity: FAIL.** Synchronization did not make the body-contact-force tensor or force-derived support/slip proxy repeatable. This does not invalidate the archived negative diagnostic; it prevents those force/slip values from supporting efficacy, physical-quality, or “measurement repaired” claims.

The stored claim is appropriately narrow: a timing witness, not an efficacy or ground-truth contact result (`results/contact_sampling_v5_20261011/versions.json:39-50`; `docs/contact_sampling_v5_20261011.md:13-17`).

## A-F checks

### A. Ground-truth and reference provenance: WARN

- External endpoint goals are derived from cached dataset motion before generation (`scripts/physics_guided_collect.py:57-63,94-106`), stored in full, and used by the endpoint/heading gates (`scripts/physics_guided_core.py:100-120,142-153`). The same original-goal hash appears in all four compared archives: `74b8ca3ee31aa96c4a1271dd38510f3cd0f1b573b2e09c4758ce8ec43d6de70b`.
- The four diagnostic sources are an internal validation subset containing two clips whose official split is train and two whose official split is test. This is not an official validation benchmark. The documentation accurately calls them “4 custom validation sources” and makes no generalization claim (`docs/contact_sampling_v5_20261011.md:15`).
- Frames 60-119 of the tracking reference come from the model-generated plan (`scripts/physics_guided_collect.py:147-204`), and tracking metrics compare simulation state with that plan (`scripts/physics_guided_core.py:131-141`). These scores are plan-adherence proxies, not independent motion ground truth.
- Contact force is simulator output. Slip uses the velocity of the geometrically lowest collision-primitive point gated by body Z-force and near-ground height (`scripts/physics_guided_core.py:70-97`). The code and result label it a proxy, and the summary says actual contact-point truth is unavailable (`results/contact_sampling_v5_20261011/repeatability_summary.json:58-61`). No fake ground truth was found.

### B. Score normalization: PASS

- Dataset `(raw-mean)/std` normalization is input preprocessing (`scripts/physics_guided_collect.py:57-63`), not score normalization.
- Evaluation reports raw SI errors, means, quantiles, counts, fractions, and fixed thresholds (`scripts/physics_guided_core.py:100-154`). No metric is divided by a maximum, minimum, or mean of the evaluated model output.
- Support Jaccard is computed as intersection/union and is accompanied by raw intersection-derived disagreement counts (`scripts/analyze_contact_repeats_20261011.py:25-48`).

### C. Result existence and numeric reproduction: WARN

- All 32 requested V5 files exist. `state.json` is complete, and both budget-ledger rows have exit code 0.
- Every file recorded in `versions.json` matches its current SHA-256. The version snapshot hash is `c1045db727f215eda61e66f737e9c1107141a1f450226befcfc9957b3b83db48`, exactly matching both execution requests and lineage records (`raw/repeat_{a,b}/lineage.json:2-20`).
- Config, manifest, and frozen-threshold hashes match the requests. Frozen thresholds exactly equal `base_thresholds`; no threshold changed.
- For repeat A and B, all lineage output hashes, receipt trajectory hashes, initial-state archive hashes, metric hashes, and receipt counts match the files.
- Independent in-memory recomputation of all 12 metrics per V5 repeat using the current frozen evaluator produced zero changed floating values and zero classification differences. The recorded object hashes are `e1c32e71...efb7` for repeat A and `3e27dba6...5e6` for repeat B (`raw/repeat_{a,b}/metric_numerics.json:2-14`). The two baseline metric sets also recomputed exactly; the older baseline has no stored `metric_numerics.json`.
- Independent recomputation reproduced `sampling_summary.json` and `repeatability_summary.json` exactly. The traceability warning remains because the latter summary and its analyzer are not bound by `versions.json` or lineage.

### D. Dead metric code: PASS

- `surface_metrics()` is called by `metrics()`; `metrics()` is called during collection and cached numeric verification (`scripts/physics_guided_core.py:70-155`; `scripts/physics_guided_collect.py:274-288`; `scripts/physics_guided_numeric_lineage.py:82-103`).
- `compare_metrics()` is called by the numeric verifier (`scripts/metric_numeric_equivalence.py:6-37`; `scripts/physics_guided_numeric_lineage.py:97-103`).
- The repeat analyzer calls both `compare()` instances and `support()` (`scripts/analyze_contact_repeats_20261011.py:12-60`). No reported V5 metric was found to come from an uncalled function.

### E. Scope and claim discipline: PASS

- Actual scope is four internal sources × three offsets = 12 inputs, seed 11, two construction batches, and no independent replay batch (`results/contact_sampling_v5_20261011/versions.json:39-46`).
- The documentation explicitly rejects generalization, innovation, true-contact, and efficacy conclusions and says the formal 432-input comparison is deferred (`docs/contact_sampling_v5_20261011.md:13-17`). Scope language does not exceed the evidence.

### F. Evaluation classification: PASS (mixed, explicitly limited)

- **Dataset-derived reference:** real dataset source for the fixed endpoint goal and prefix.
- **Plan tracking:** self-supervised/control-reference proxy after frame 60; not independent generation ground truth.
- **Force/slip:** simulation-only body-force and collision-geometry proxy; no real contact-point ground truth.
- **Repeatability result:** simulation-only diagnostic comparing identical configured runs.

## Frozen-condition verification

| Item | Audit result |
|---|---|
| Generator weights | Actual SHA-256 `3ca68ae54975946b3fe5c6cd64a6eafc439b5fc43c166bc9a98d1c43aeef6618`; matches `versions.json` |
| Controller weights | Actual SHA-256 `7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac`; matches `versions.json` |
| Text-motion evaluator weights | Actual SHA-256 `43e43d198f07b7e080749d1709326783d2bfb5810f56ed0c56c68951ee961d0d`; matches `versions.json` |
| Upstream CLoSD | Commit `d8950bdb78312fcd2844f336b3fc079e7368c87f`; no tracked `closd/` diff |
| External motion/semantic inputs | Every recorded file exists and matches its stored SHA-256 |
| Resolved config | All four compared archives are byte-identical, SHA-256 `7892b948f8e96cf15acda7f37b72505164e49b8c4ef076f1f39871c89a0958ea` |
| Cases, original goals/headings | Exact equality across both unsynchronized and both synchronized archives |
| Initial exported state | All archived bootstrap, first-action, and first-action-contact-force arrays are bitwise equal across all four runs; every recorded reset/cache difference is zero |
| Generation prefix/noise | Recorded prefix SHA `f446cdc6...f681` and noise SHA `1a13fe43...0ec1` are equal across all four runs; generated reference arrays are bitwise equal. Raw prefix/noise tensors were not archived, so independent hash-to-tensor replay was not possible in this read-only CPU audit. |

The dynamically constructed V4 and V5 collector sources were rebuilt without executing the collector. Their runtime-source diff contains one executable addition only: `torch.cuda.synchronize()` after reward refresh and before tensor copies. Static adapter files also differ in title and line endings. V5 metadata binds adapter SHA `651bbaf6...328a4`; both baselines bind V4 SHA `f2ecbb37...762be`; all bind the same historical collector SHA `a5d426cf...77d43`.

## Independent full-array and support-mask verification

All comparisons use the complete 120-frame arrays, not a rerun or a summary-file interpretation.

| Pair | Kinematic/reference max difference | Contact-force max difference | Support Jaccard | Support-label disagreements | Frame-case disagreements | Max slip-fraction difference |
|---|---:|---:|---:|---:|---:|---:|
| Unsynchronized baseline pair | 0 for actual/reference/quaternion/linvel/angvel | 744.8496704101562 N | 0.8430458430458431 | 808 / 5760 | 656 / 1440 | 0.035011263395018866 |
| Synchronized V5 repeat pair | 0 for actual/reference/quaternion/linvel/angvel | 719.0886840820312 N | 0.8536632891660172 | 751 / 5760 | 624 / 1440 | 0.044688917894213365 |

For the synchronized pair, force arrays differ from frame 0 through frame 119, in 1,427 of 1,440 frame-case pairs and 10,899 of 103,680 scalar force elements. Initial archived contact force is nevertheless bitwise equal. Synchronization therefore did not establish the declared 1 mN repeatability criterion (`results/contact_sampling_v5_20261011/sampling_summary.json:7-12`).

All classification flips are zero, but all 12 cases have `physical_pass=false` and `joint_pass=false` in every archive. Stable all-false classifications do not validate the unstable continuous force/support/slip measurements.

## Claim impact

| Claim | Verdict |
|---|---|
| V5 changes the collector read timing by adding a producer-stream synchronization only | Supported at runtime-source level |
| Weights, config, cases, goals, initial exported state, generated reference, and recorded noise/prefix hashes are frozen across compared runs | Supported, with the raw-noise/raw-prefix archival limitation above |
| Kinematic/reference trajectories are repeatable in this 12-case configuration | Supported; full arrays are bitwise identical |
| Synchronization repairs contact-force sampling or makes it repeatable at 1 mN | Refuted by the archived repeats |
| Force-derived support/slip is a valid stable physical-quality measure for efficacy gating | Unsupported |
| V5 establishes efficacy, real-contact accuracy, generalization, or innovation | Unsupported and explicitly disclaimed |
| The negative timing witness shows synchronization alone is insufficient in this configuration | Supported |

## Action items

1. Keep G1/downstream efficacy work paused wherever it depends on force-derived support/slip or the physical/joint gate.
2. Treat V5 as a retained negative diagnostic. Do not pool it with efficacy results or present classification equality as force-measurement validation.
3. Before reusing contact metrics, define and validate a deterministic/contact-grounded acquisition method, ideally with per-contact data or an independently checked simulator contact signal.
4. In a future non-execution analysis manifest, bind `analyze_contact_repeats_20261011.py`, the four trajectory hashes, and `repeatability_summary.json`. Preserve the existing experiment outputs unchanged.

## Audited snapshot hashes

- `versions.json`: `c1045db727f215eda61e66f737e9c1107141a1f450226befcfc9957b3b83db48`
- config: `2d8fd0e49ce7ed37393e1962dcd711b5a6e8f26de7e0ca4ef48964a6fd081502`
- manifest: `a7065f535ee7adbc8bac95c5be2bc527cf740d43eabe3a40105ce6c471bf95de`
- frozen thresholds: `451fae0a2b106dbd0170185d9a22bb6f3fe2583709e77979eb017cc3ce09315b`
- V5 collector adapter: `651bbaf6cd0a2ba675fdbd5d735d66ee5a3aeecab755060aa2b5628be8d328a4`
- V4 collector adapter: `f2ecbb37e4fd15de8433b97b386ed6b13af1b343a83a4685e9888f77c7a762be`
- base collector: `a5d426cf942ecf0b3ac121b389aad5dd6459ce8ea34047c5d5a88b0b1cc77d43`
- metric core: `e6be20aa8a060bad20881cb81bab4e3328282a163b0d3e9571c7c590443436ed`
- repeat analyzer (not version-bound): `8cdd2ea795cad2c16e0d9ee736568ab2c700dcdb7ce84c1d7f5e68be8d60aa22`
- synchronized repeat A trajectory: `abaebaf9260916e4ccaa6c2e8ba9d2d99231514d40379ca7cb6c9c1269cd3695`
- synchronized repeat B trajectory: `fb895c38fe434823f478f3d8b2a5a21db189d96aff61e09ac1e8f5bbe295dfdf`
- unsynchronized baseline trajectory 1: `a3ddc14c4323f2d398f3ae7d56934c46ecd5ec0318d5f015fb6b18c0ed36d569`
- unsynchronized baseline trajectory 2: `7ed29031c2a5c4b19965b5e6e23372e4d9c51609c2bb5505b0998c26267469bf`

