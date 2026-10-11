# Fresh document-following witness — CPU tensor export V3

Date: 2026-10-12. Witness agent: contact_cpu_export_v3_document_witness.

## Verdict

Documented invocation: EXECUTION PASS. Exactly one `--check` invocation, exit 0, actual stdout `FIXTURE_DOCUMENT_EXECUTION_COMPLETE repeat_a PASS`. No edits to invocation, scripts or environment; no retry, `--run`, package installation or additional simulation by this witness.

Independent repeat_a calibration: all nine conditions PASS under the explicitly recorded settled-window convention below. No failed calibration checks. This is a synthetic box/plane measurement fixture, not evidence of humanoid validity, scientific innovation, human motion quality or downstream efficacy.

## Exact documented invocation executed

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_force_fixture_v3_20261012.py --check
```

Read context was restricted to run-experiment/SKILL.md, its compute-env-contract.md, the provider ledger and docs/contact_force_fixture_v3_20261012.md. Subsequent inspection read actual output arrays/metadata and file bytes only for hashes; simulation source was not read to derive checks.

## Independent raw-array checks

NPZ loaded remotely with `allow_pickle=False`; computations used its actual arrays converted to float64 and fixed numeric thresholds, without using summary booleans. State shape 120x4x13, net force 120x4x3, constraint force sensor 120x4x6. All arrays finite.

Doc ambiguity: “settled” does not identify frame indices or the target box-center height. For this independent check, settled frames are zero-based 60:120 (last 60), expected center height 0.05m, stationary/sliding/airborne fixture indices 0/1/2. This convention is an interpretation and should not be confused with a document-specified window. Sliding velocity uses all 120 frames and full 3D deviation from [0.2,0,0]m/s; horizontal force uses all 120 frames and all four fixtures. Airborne tests use first 20 frames of fixture 2. Sensor force uses first three wrench components; support compares signed Z with 49.05N.

| Condition | Independent measured value | Threshold | Result |
|---|---:|---:|---|
| All frictionless horizontal net force norms | max 0.000256169267N | <1N | PASS |
| Settled stationary center-height error | max 0.002000507712m | <=0.01m | PASS |
| Settled stationary linear speed | max 0.001265828662m/s | <0.02m/s | PASS |
| Settled stationary net support relative error | max 0.000143037537 | <0.10 | PASS |
| Settled stationary sensor support relative error | max 0.000281548767 | <0.10 | PASS |
| Sliding velocity deviation | max 0.001265956651m/s | <=0.02m/s | PASS |
| First20 airborne net force norm | max 0N | <1N | PASS |
| First20 airborne sensor force norm | max 0N | <1N | PASS |
| First20 airborne center height | min 1.413443446m | >0.5m | PASS |

Independently computed booleans agree with summary booleans; this comparison was made after computing the checks.

## Hash binding

Raw remote NPZ SHA256 and downloaded local NPZ SHA256 both equal `9d16ec394f9aa5aa5a508638d7da9d35f4066b6b7e580dcdbde81629f5e86ef7`, matching summary.arrays_sha256. NPZ size 23002 bytes.

Each of five bound version hashes equals both current remote and current local file hashes:

| Bound file | SHA256 |
|---|---|
| scripts/contact_force_fixture_v3_20261012.py | 291f6c9ca8430a2ace6212ba305946c012986f8926540170b07093d503f85ea8 |
| scripts/run_contact_force_fixture_v3_20261012.py | 02e54bb8c0b7a2e2b60cc1cbf7be160853b81f6cb4a90fb508953052b5608915 |
| scripts/run_reset_alignment_checks_v4.py | 5d0a961b6ce9903fd666fdbae56736a68eddd1402ddf72793fc000ee0dea98d2 |
| configs/env-closd-baseline-20261010.json | e5f9e6b711b7c6221cfb01e3deab404c5849e521ec08a3d25dca1820a279fcf6 |
| docs/contact_force_fixture_v3_20261012.md | ae1156f166a182dbce1d7839fee4ef892fee82640750d3028bd17d5457ba77d4 |

Remote environment spec was independently parsed and canonicalized with sorted JSON keys and no whitespace: SHA256 `81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780`, equal to the warm-runtime ledger. No rebuild.

## Limits and concurrent metadata observation

Completion sentinel is distinguished from physical calibration; calibration conclusion comes from the independent calculations above. This witness does not validate tensor semantics for humanoid contact, articulated force sensors, slip proxies, controller/generator behavior or real hardware force. GPU0-idle and free-disk prerequisites were not separately measured by this document witness; the wrapper returned exit0. Full repeat comparison belongs to the root agent.

During output inspection, a directory listing showed only repeat_a artifacts, while a subsequent state/budget read showed REPEATS_COMPLETE and repeat_b budget entry. The root confirmed it executed `--run` concurrently after observing WITNESS_COMPLETE and completed repeat_b. The reads were therefore not an atomic snapshot. This witness did not execute repeat_b or infer repeatability from those metadata fields.

Machine-readable independent results, metrics and hashes: INDEPENDENT_DOCUMENT_CHECK.json. Small repeat_a fixture_arrays.npz and summary.json are preserved locally. No efficacy counted.
