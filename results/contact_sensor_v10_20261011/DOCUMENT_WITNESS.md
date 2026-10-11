# V10 fresh document-following witness — 2026-10-11

Status: PASS_RECORDING_ONLY. A fresh delegated agent read the run-experiment skill, its sibling shared-references/compute-env-contract.md, the unchanged runtime ledger, and docs/contact_sensor_v10_20261011.md. The documented invocation was executed verbatim exactly once. No retry, --run, source edit, package installation, or V9 rerun was performed by this witness.

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_sensor_v10_20261011.py --check
```

The original process was polled to completion, never duplicated. Exit code was 0. Its full stdout was:

```text
CONTACT_SAMPLING_BATCH_PASS repeat_a
CONTACT_SAMPLING_DOCUMENT_CHECK_PASS
```

Remote state was WITNESS_PASS. The measurement ledger retained one repeat_a entry, 32.20337660424411 seconds, exit 0. Receipt elapsed_seconds was 29.490207195281982. There was no V10 first failure to conceal or retry.

## Independent artifact inspection

Direct, read-only Python/NumPy inspection was performed on remote results/contact_sensor_v10_20261011/raw/repeat_a, without invoking the collector or its cache-certification helper again.

| Actual gate | Observed result |
|---|---|
| Original attempts | 12 distinct case IDs, four manifest sources × three offsets; complete receipt, 12 saved metric rows |
| Valid lengths | [120,120,120,120,120,120,120,120,120,120,120,120] |
| Terminations | All 12 false |
| Reference clock | Exactly 0…119 |
| Seven reset state tensors | Bootstrap versus first action maximum difference exactly zero for root, DOF position/velocity, rigid-body position/rotation/linear/angular velocity |
| Reset snapshot provenance | All seven bootstrap and first-action raw-byte hashes independently match metadata; arrays finite |
| Four official rigid caches | Metadata reports exactly zero position/rotation/linear/angular velocity differences; adapter source independently reviewed for the live comparisons |
| Initial reset gate | true; initial actual/reference horizontal anchor maximum difference 0 m |
| Sensor bodies | L_Ankle, L_Toe, R_Ankle, R_Toe |
| Policy observations | self_obs_v=1 in resolved YAML and metadata; observation size 574; native mode-1 observation branch excludes force sensors |
| Sensor selection | sensor_metrics_used_for_selection=false |
| Physics completion | 242 fetch_results(True) completion fences; producer-stream synchronization retained; no reward contact mutation measured |
| Same inputs as V8 | Exact equality for full cases/text, original/proposed goal and heading, noise/prefix/goal hashes, asset hash and recorded effective SimParams |
| Actual efficacy gates | completed 12/12; condition_pass 4/12; physical_pass 0/12; tracking_pass 12/12; joint_pass 0/12 |

The four cache differences are inspected live-comparison metadata, not a separate archive of those private cache tensors. The seven reset-state equalities were independently recomputed from initial_states.npz. Neither certifies hidden PhysX state.

| sensor_diagnostic.npz array | Shape | Type | All finite |
|---|---|---|---|
| sensor | [120,12,4,6] | float32 | yes |
| root | [120,12,13] | float32 | yes |
| dof_pos | [120,12,69] | float32 | yes |
| dof_vel | [120,12,69] | float32 | yes |
| action | [120,12,69] | float32 | yes |

The sensor absolute maximum was 115.27473449707031. This establishes finite nonempty recording, not calibration. All trajectories.npz numeric/boolean arrays were finite. Actual/reference/linear/angular velocity/contact shape was [120,12,24,3], quaternion [120,12,24,4].

## Source, configuration and lineage

All 16 files in versions.json were independently rehashed and matched, including all listed collector/wrapper/metric/lineage sources, both configuration files and the documentation. Manifest hash matched versions.json. Original execution request matched requests/repeat_a.json exactly; its config, manifest, thresholds and versions bindings matched their actual bytes. The lineage is bound_request_at_execution with confirmatory=false. All three lineage output bindings (metadata, metrics, trajectories) matched actual bytes. The saved numeric certificate reported exact metric agreement, unchanged scores/classifications/thresholds.

Generator and controller weight files were independently rehashed and matched the frozen versions; all four external motion-input files and five external semantic-input files matched. The actual cached asset XML third_party/CLoSD/closd/data/robot_cache/smpl_humanoid_0.xml matched the foot_asset_sha256 and the V8 asset hash.

Upstream HEAD was d8950bdb78312fcd2844f336b3fc079e7368c87f. `git diff --name-only HEAD -- closd` returned empty. Canonical parsed environment-spec SHA256 was 81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780, matching the ledger. No environment rebuild was required.

The recorded effective SimParams equal V8: dt=0.01666666753590107, substeps=2, GPU pipeline=true, contact_collection=2, PhysX GPU=true. Resolved YAML also retains solver_type=1, num_position_iterations=4, num_velocity_iterations=0. The collector records this subset of effective SimParams; this witness does not claim an exhaustive serialization of every PhysX setting.

| Artifact or source | Actual SHA256 |
|---|---|
| sensor_diagnostic.npz | 2f059b8a1cd79428bc4097b0f30bbf8b1035bbae97e521964a4ed05fbe7cea80 |
| trajectories.npz | 18e8cb67e489a3d3f12657eeb2b151f0cddd734eab5e7985787c85ae56a8fea9 |
| initial_states.npz | 866765c4e3c965f532efae1502cd2200c52bca9271ec41d195c52fa26740abed |
| resolved_config.yaml | 7892b948f8e96cf15acda7f37b72505164e49b8c4ef076f1f39871c89a0958ea |
| metadata.json | db886ac9af22739e59c52bda92d3a3b42b7bab0e01143e250494e68fb9129ae8 |
| metrics.json | 8d94ea73c879a87495d67ba347e8d7b39723f685841eda84785af3107e0d2d62 |
| versions.json | c7750a9bbc6768f640ec4b2552028311fc74336e4fc327fa3f0709ce89dbd9d2 |
| source_manifest.json | a7065f535ee7adbc8bac95c5be2bc527cf740d43eabe3a40105ce6c471bf95de |
| frozen_thresholds.json | 451fae0a2b106dbd0170185d9a22bb6f3fe2583709e77979eb017cc3ce09315b |
| V10 collector | 9d76cc05fd82391f1a1c9f1b1485d0a4a5ca59eef0f559895d8c1f3a69148d9a |
| V10 runner | 17218ebe2f764b35302d9618065e9a9aa1acd69cfe3d4922ddb0aaf347ab3788 |
| Document | 0251a7cbd23efc559a8326f3866f2352940cb15e6f5d058ba4a5d51e5df88748 |
| Asset XML | 5cb2b04e824c79c4381c78f6991f385082ba0c0dae351f29e0c4200539c65092 |
| Generator | 3ca68ae54975946b3fe5c6cd64a6eafc439b5fc43c166bc9a98d1c43aeef6618 |
| Controller | 7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac |
| Generation noise | 1a13fe432251f25e4aa92decf9dc07e208ed82a979948e32b82868b6c21b0ec1 |
| Generation prefix | f446cdc60658dd0336b02596256b18401341fe7101d12906dbd0b1ae3aa0f681 |
| Original goal | 74b8ca3ee31aa96c4a1271dd38510f3cd0f1b573b2e09c4758ce8ec43d6de70b |

## Preserved failure and claim boundary

V9 remains FAILED_RETAINED. Its sole ledger attempt is 0.06615031138062477 seconds, exit 1. Its log retains AssertionError at contact_sensor_collect_v9.py line 11, `assert adapter.count(ending) == 1`, before runtime. This witness did not rerun or alter V9. V10 source removes only the final matching adapter invocation using rsplit, avoiding the duplicate literal in the parent.

No documented recording claim diverged from observed repeat_a behavior. Measurement readiness is limited to this recording path. Sensors use world-frame solver-constraint forces, forward-dynamics forces disabled, constraint-solver forces enabled. Articulated internal joint reactions may contribute; readings are not pure contact ground truth. The native upstream refresh_force_sensor_tensor call was independently located in humanoid.py, and the sensor branch is used in observations only for mode 3.

This witness does not certify force validity, repeat_b repeatability, equality of complete kinematics to V8, innovation effectiveness, task semantics, downstream utility, or robot hardware. The simple 5 kg box/plane support fixture does not calibrate articulated sensors. Root must complete the independent repeat_b and all measurement gates before interpreting contact-based efficacy. The physical/joint gates currently fail and no efficacy result follows from exit 0.
