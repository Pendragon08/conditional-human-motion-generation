# Independent document-following witness: V13 slip proxy diagnosis

Date: 2026-10-12. Witness: fresh delegated agent /root/contact_slip_v13_document_witness.

## Execution receipt

I read only the designated local instruction/source files before execution:
- docs/contact_slip_diagnostic_v13_20261012.md
- .aris/compute/closd-baseline-20261010.md
- scripts/contact_slip_diagnostic_v13_20261012.py
- scripts/physics_guided_core.py

I executed the documented SSH command exactly once, with no retry:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/contact_slip_diagnostic_v13_20261012.py
```

Tool process handle: session_id 73575; completed and closed with exit code 0. Exact final stdout:

```text
SLIP_PROXY_OFFLINE_DIAGNOSTIC_PASS {"inputs": 432, "original_proxy_pass": 0, "optimistic_near_lowest_pass": 1, "original_metric_recompute_max_abs_error": 0.0}
```

There were no errors, retries, extra simulations, source edits, package installations, model calls, threshold edits, or licensed-array downloads. I subsequently issued one separate read-only SSH Python inspection of the saved diagnostic, bound input files and bundled SDK documents; that inspection also returned exit code 0. I did not execute the diagnostic again.

## Actual saved result and count audit

Server result: /data/code/ouyanghao/oyh/conditional-human-motion-generation/results/contact_slip_diagnostic_v13_20261012/diagnostic.json

SHA256: c3642ed522e4c843802bee21e6303d89ab84c07cabebc5925f655f198f4604be

Saved status: DIAGNOSTIC_ONLY. Actual row count = 432, distinct (label, case_id) count = 432, validation summary attempts = 432. The summary contains full_quality_floor = true; this is an input assertion, not this witness's certification of the quality floor.

| Label | Saved metric rows | Diagnostic rows |
| --- | ---: | ---: |
| val_planned_s11 | 72 | 72 |
| val_native_s11 | 72 | 72 |
| val_planned_s12 | 72 | 72 |
| val_native_s12 | 72 | 72 |
| val_planned_s13 | 72 | 72 |
| val_native_s13 | 72 | 72 |

All six metadata/metrics pairs matched their validation-summary hashes. All 20 diagnostic input_hashes matched actual server file bytes, including all six server-local trajectories.npz files. Support observations total 55,589, minimum 23 per attempt. Recounting the actual saved rows using the bound frozen threshold gives original proxy pass = 0/432 and optimistic near-lowest diagnostic pass = 1/432. The saved original recomputation maximum absolute error is exactly 0.0; the executed source asserts this error is below 1e-12.

All entries bind dt = 0.03333333507180214 seconds and context_steps = 60. The measurement uses the generated portion from min(context_steps, valid_length - 1), while retaining every case from the full summary. Foot geometry keys are L_Ankle, L_Toe, R_Ankle, R_Toe. No result-based case selection is present in the inspected source.

Frozen slip/support values: sliding_speed_m_s = 0.05; sliding_fraction = 0.1; contact_force_n = 10; surface_near_ground_m = 0.02. Frozen-threshold file SHA256: 451fae0a2b106dbd0170185d9a22bb6f3fe2583709e77979eb017cc3ce09315b. The source reads these values without rewriting the threshold file; saved thresholds_changed = false.

## Known recorded box controls

The script uses existing repeat_a/fixture_arrays.npz, stationary and slide frames [60,120), airborne frames [0,20). Its bound fixture hash matches actual server bytes:
9d16ec394f9aa5aa5a508638d7da9d35f4066b6b7e580dcdbde81629f5e86ef7

| Recorded control | Support observations | Proxy mean speed (m/s) | Sliding fraction | Airborne fraction |
| --- | ---: | ---: | ---: | ---: |
| known_stationary | 60 | 0.0006039751541892994 | 0 | 0 |
| known_frictionless_slide | 60 | 0.1999885083798524 | 1 | 0 |
| known_airborne | 0 | null | null | 1 |

All three saved calibration checks are true: stationary classified not sliding, frictionless slide classified sliding, airborne has no support. These controls establish separation for translating non-articulated boxes under the frozen proxy. They do not certify articulated/rolling feet or humanoid quality discrimination.

## Source binding and interpretation

The server diagnostic source hash agrees with the saved source_sha256 and the designated local script:
1bc85f8b125aacac948c89decc6a96f0a85f6cc7345d9d855dbebed77569bdc1

The server shared-core hash agrees with the designated local core:
e6be20aa8a060bad20881cb81bab4e3328282a163b0d3e9571c7c590443436ed

The local instruction-document hash is 7bcb9aa0f2a2f355809a479e41c268df5a2591ef5b724bd74e82c4eaa648c6fb; the designated warm-runtime note hash is 44a3fa71cae414c27c1ad2c5cc642f4b69617de0afb097c3c12a79677957e503. The warm-runtime note states reuse of the pinned runtime; this offline diagnosis did not launch or certify a simulator runtime.

The inspected shared core computes velocity at the geometrically lowest candidate surface point using linear velocity plus angular velocity cross lever arm. Support requires body vertical force above the threshold and lowest surface height below the near-ground threshold. The V13 diagnostic computes the slowest candidate within 0.002 m of the lowest point only as an optimistic lower bound. This is not a measured native contact point and must not replace the frozen score.

Both bundled SDK-document hashes match the saved sdk_evidence. The bundled descriptions identify local_pos0 as a local contact-feature position, lambda as contact force magnitude, com as a body-space center of mass, and get_env_rigid_contacts as an environment contact query. The diagnosis reads these documentation files; it does not establish runtime availability of native contact points or their relationship to exported velocity origins. Coarse finite differences at approximately 30 Hz do not establish an SDK velocity-frame bug.

Saved no_simulation_or_model_calls = true; innovation_confirmed = false; overleaf_results_sync = false. Source inspection is consistent with offline array analysis and documentation reads. This witness certifies the documented execution, actual saved result, hash bindings, all-432 inclusion, and the limited box controls only. It does not validate humanoid quality discrimination, any candidate innovation, semantic quality, downstream training utility, or robot hardware effects. Next required evidence remains known articulated support/rolling examples and native contact-point availability. No efficacy claim is made.

## Complete input binding audit

All values below were independently computed from server bytes and equal the diagnostic's declared values.

| Server-root-relative input | SHA256 |
| --- | --- |
| results/contact_cpu_validation_v12_20261012/frozen_thresholds.json | 451fae0a2b106dbd0170185d9a22bb6f3fe2583709e77979eb017cc3ce09315b |
| results/contact_cpu_validation_v12_20261012/validation_summary.json | 6cfe8863a5f4f9c8be00fed370687a014017c70ee1fafafae4811fbfa43ed4dc |
| results/contact_cpu_validation_v12_20261012/raw/val_planned_s11/metadata.json | e2cb0c23e88710661ee9f19b941a00305cc2316904b1c424c5c9436674bde7bb |
| results/contact_cpu_validation_v12_20261012/raw/val_planned_s11/metrics.json | 5cdddd123b3faaf59a2977f82bb7e1758da94dd526118568d8a6913c014dbeab |
| results/contact_cpu_validation_v12_20261012/raw/val_planned_s11/trajectories.npz | 45486979e705ce28f3e5e10976814ce901f3008ea902e5e23d326a5c0a4311bd |
| results/contact_cpu_validation_v12_20261012/raw/val_native_s11/metadata.json | adb93390a59b4bb96aced0818f93c6575dffb867de53416500fced80ef27ea45 |
| results/contact_cpu_validation_v12_20261012/raw/val_native_s11/metrics.json | d6600f2aaf39674529b45b6d65e86a156ef963a01c63f98e1ac15c7b12ee2d23 |
| results/contact_cpu_validation_v12_20261012/raw/val_native_s11/trajectories.npz | e7fc61dc6513611b3d70c8a75b5c52bcd455d38dfc646cb7bdf1c0c5c50258db |
| results/contact_cpu_validation_v12_20261012/raw/val_planned_s12/metadata.json | 858521f9f38373e2f7a4bd6903d5aedd39f4bbc80a6823f6c94c181e824a1b79 |
| results/contact_cpu_validation_v12_20261012/raw/val_planned_s12/metrics.json | c607bfec16bf505fbf0e8bb74567e8fec19309624b62ca0e098232e1a924972c |
| results/contact_cpu_validation_v12_20261012/raw/val_planned_s12/trajectories.npz | 414a10df804bce7ea040bd53eda955c64db844bfdf90d1fede6a2c355e5fcfd5 |
| results/contact_cpu_validation_v12_20261012/raw/val_native_s12/metadata.json | a2ff8785aff47853233a6ae352131e0c3f371d59b254315e69a8972aa9e3287e |
| results/contact_cpu_validation_v12_20261012/raw/val_native_s12/metrics.json | 349798b4a822fa26a882166cffc388017738a78cf183229be19c1a307239c598 |
| results/contact_cpu_validation_v12_20261012/raw/val_native_s12/trajectories.npz | 48a56e95d639e5e0063cf9db4a42000dc72a4b6a714079f99d2a6c800faf37c5 |
| results/contact_cpu_validation_v12_20261012/raw/val_planned_s13/metadata.json | df2db815a34aa15c3b8dbf1f16084899c0f42184f6682b0c445908416195de12 |
| results/contact_cpu_validation_v12_20261012/raw/val_planned_s13/metrics.json | 6cd2c5d5b9a245e71d3fa0921d8619b4b2f811fcf7fff083bde137f65180c385 |
| results/contact_cpu_validation_v12_20261012/raw/val_planned_s13/trajectories.npz | ce1ac5f89058a500da226a1be07b6778addc690682a29eba2e8ab00dd195b55f |
| results/contact_cpu_validation_v12_20261012/raw/val_native_s13/metadata.json | 70e7f94d79bf91da5997330b52d4d68a72d124290abc1f1f3cf010a6eeb8e4e2 |
| results/contact_cpu_validation_v12_20261012/raw/val_native_s13/metrics.json | b8898baefc0cd9d59d94d17eb907ecfc9953d62026c3d2645332a24b0835a2e9 |
| results/contact_cpu_validation_v12_20261012/raw/val_native_s13/trajectories.npz | 25222439b196035d4018d8b079a9fce41c198238b9244a81b608f28960f3da67 |

SDK document hashes:
- /data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/isaacgym/docs/api/python/struct_py.html: a8e2c2e12b4a26b053572adf60968e63fb4054bd73c7512ebdd3da6c53c0f7de
- /data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/isaacgym/docs/api/python/gym_py.html: 39a2bbbe26c61d77d322c776dfc0d7a890c0070a31ccace4631467dea6ef8da1

Licensed motion arrays remain server-local. Other worktree files were not changed by this witness.

