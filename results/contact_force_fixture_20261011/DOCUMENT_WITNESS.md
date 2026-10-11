# Independent document witness: contact force fixture

Witness date: 2026-10-11, Asia/Shanghai. Fresh delegated document witness.

**Recording completion: PASS_RECORDING. Physical calibration: PASS_CALIBRATION.**
The documented wrapper returned exit 0 and printed `FIXTURE_DOCUMENT_EXECUTION_COMPLETE repeat_a PASS`. The completion sentinel is a recording receipt; physical calibration is separately established here by recomputing all eight documented checks from the saved arrays. A `FAIL_RETAINED` summary would remain a physical failure despite wrapper exit 0.

## Invocation and scope

Read the run-experiment skill, its shared compute environment contract, `.aris/compute/closd-baseline-20261010.md`, and `docs/contact_force_fixture_20261011.md`. Executed the following documented command verbatim **once**, with no retry, source/config/doc edit, or `--run`:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_force_fixture_20261011.py --check
```

The wrapper's unchanged resource guard checks GPU0 free memory at least 24,576 MiB and utilization at most 5% before launching the fixture; budget and disk guards are also present before launch. This witness did not install packages, train models, invoke a paid API, test hardware, or modify Overleaf results. Subsequent verification used only read-only file/hash inspection and NumPy calculations on saved arrays. The requested repository-level `AGENTS.md` was absent; the supplied compute ledger and protocol identify the documented runtime and invocation.

## Saved recording and metadata

Remote output: `/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/contact_force_fixture_20261011/repeat_a/`.

| Array | Shape | Type | Finiteness |
|---|---|---|---|
| `state` | 120 x 4 x 13 | float32 | all finite |
| `net_contact_force` | 120 x 4 x 3 | float32 | all finite |
| `constraint_force_sensor` | 120 x 4 x 6 | float32 | all finite |

Fixture order: stationary_grounded, frictionless_slide, high_airborne_drop, upward_takeoff. Measured actor masses: 5 kg each; weights: 49.05 N each. Metadata weights equal measured masses times 9.81 exactly. Metadata records 120 steps, dt 0.01666666753590107 s, 2 substeps, contact_collection 2, generator_used=false, controller_used=false, thresholds_preregistered=true, and simulator_force_sensor_is_not_real_force_ground_truth=true. Source inspection confirms world-frame sensors with forward-dynamics forces disabled and constraint-solver forces enabled.

Wrapper budget ledger contains one measurement row for repeat_a, exit_code=0, 2.621175456792116 s. Fixture elapsed_seconds is 0.791958836838603. At inspection, wrapper state was WITNESS_COMPLETE. The compressed array archive contains 22,949 bytes.

## Independent analytical checks

Stationary and sliding checks use the last 30 saved frames. Airborne checks use the first 20 frames. All recomputed booleans equal `summary.json`, and the recomputed overall status agrees with its PASS status.

| Frozen check | Recomputed value | Criterion | Result |
|---|---:|---:|---|
| settled_stationary_height_within_1cm | maximum height error 0.002000506967306137 m | <0.01 m | PASS |
| settled_stationary_speed_below_2cm_s | maximum linear speed 0.0012657002080231905 m/s | <0.02 m/s | PASS |
| net_force_weight_relative_error_below_10pct | maximum relative error 0.00027570034264303003 | <0.10 | PASS |
| sensor_force_weight_relative_error_below_10pct | maximum relative error 0.0002815332130797655 | <0.10 | PASS |
| sliding_velocity_retained_within_2cm_s | maximum vx error from 0.2 m/s: 0.000003591179847717285 m/s | <0.02 m/s | PASS |
| high_airborne_initial_force_below_1N | maximum absolute net-force component 0 N | <1 N | PASS |
| high_airborne_initial_sensor_below_1N | maximum absolute sensor-force component 0 N | <1 N | PASS |
| high_airborne_initial_height_above_ground | minimum high-drop height 1.4134434461593628 m | >0.5 m | PASS |

Settled stationary normal net-force range is 49.043155670166016 to 49.06352233886719 N. Sensor normal-force range is 49.036190032958984 to 49.063514709472656 N. These recomputed ranges exactly agree with summary metadata. The upward-takeoff fixture is recorded; it has no separate preregistered acceptance check. Its first recorded height is 0.34591251611709595 m, maximum recorded height 0.4953082501888275 m, and first recorded vertical speed 1.7547498941421509 m/s. These are descriptive observations only.

## Hash binding

All five current remote bound files match `versions.json`, and all five local files match their remote byte hashes. The source hash also matches the summary's source_sha256. The saved archive hash independently recomputed from bytes equals the summary's arrays_sha256.

| Bound file | SHA256 |
|---|---|
| scripts/contact_force_fixture_20261011.py | 4b04f7aa4bb7a34425afb9a4f63cf852dd94c3a13598b2d460d12a3bdcefb9c1 |
| scripts/run_contact_force_fixture_20261011.py | 74b7a4957e6da438ed41b67f0371516599422765b984c40401374b2c77bd7f3b |
| scripts/run_reset_alignment_checks_v4.py | 5d0a961b6ce9903fd666fdbae56736a68eddd1402ddf72793fc000ee0dea98d2 |
| configs/env-closd-baseline-20261010.json | e5f9e6b711b7c6221cfb01e3deab404c5849e521ec08a3d25dca1820a279fcf6 |
| docs/contact_force_fixture_20261011.md | c48670026588309ef63853a46dde133b271a7d0f8b64ce3d81c6350477b468cb |
| repeat_a/fixture_arrays.npz | 7ee565053a0c1424b1466a676be7b4dc5081b62473384fbb00ddfc31a9aec722 |

## Document versus execution

No divergence found for the documented invocation, expected recording sentinel, array shapes/finiteness, measured masses, metadata, hashes, or stated analytical acceptance checks. All attempts remain retained. Only repeat_a was witnessed; repeat_b is outside this witness.

This result calibrates the specified rigid box/plane fixtures under the recorded simulator settings. It establishes no human contact-point truth, humanoid articulated sensor validity, slip-proxy validity, motion quality, model efficacy, real-robot behavior, or downstream utility.
