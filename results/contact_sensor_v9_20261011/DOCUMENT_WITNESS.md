# V9 independent document witness: FAILED_RETAINED

Date: 2026-10-11. A fresh agent read the run-experiment skill, its compute environment contract, the unchanged closd-baseline runtime ledger, and docs/contact_sensor_v9_20261011.md, then executed the documented invocation exactly once.

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_sensor_v9_20261011.py --check
```

## Observed result

The SSH invocation exited 1. It did not time out. Both expected sentinels are absent: CONTACT_SAMPLING_BATCH_PASS repeat_a and CONTACT_SAMPLING_DOCUMENT_CHECK_PASS.

The authoritative remote state is FAILED_RETAINED, with error RuntimeError('Probe failure retained: repeat_a'). The budget ledger contains exactly one measurement attempt, label repeat_a, exit_code 1, duration 0.06615031138062477 seconds. The wrapper traceback is:

```text
File "scripts/run_contact_sensor_v9_20261011.py", line 181, in <module>
  main(parser.parse_args().check)
File "scripts/run_contact_sensor_v9_20261011.py", line 163, in main
  batch('repeat_a', 'native')
File "scripts/run_contact_sensor_v9_20261011.py", line 107, in batch
  raise RuntimeError('Probe failure retained: ' + label)
RuntimeError: Probe failure retained: repeat_a
```

The preserved repeat_a.log identifies the earlier underlying failure:

```text
File "scripts/contact_sensor_collect_v9.py", line 11, in <module>
  assert adapter.count(ending) == 1
AssertionError
```

Independent read-only inspection finds adapter.count(ending) == 2 in contact_sampling_collect_v8.py. The literal appears once inside the assignment of ending and once as the final exec statement. The V9 adapter requires exactly one occurrence and stops before executing the parent collector. This is a source-composition/document-vs-reality divergence, not evidence of model, policy, sensor, physics, or environment failure.

## Artifact and hash inspection

No raw/repeat_a directory, metadata.json, receipt.json, trajectories.npz, or sensor_diagnostic.npz was produced. The retained files are budget_ledger.json, frozen_thresholds.json, queue.lock, repeat_a.log, requests/repeat_a.json, source_manifest.json, state.json, and versions.json.

Every one of the 16 source/protocol/config/document hashes in remote versions.json matches the corresponding current remote file. The authoritative hashes are:

| File | SHA256 |
|---|---|
| scripts/reset_aligned_collect_v4.py | f2ecbb37e4fd15de8433b97b386ed6b13af1b343a83a4685e9888f77c7a762be |
| scripts/physics_guided_collect.py | a5d426cf942ecf0b3ac121b389aad5dd6459ce8ea34047c5d5a88b0b1cc77d43 |
| scripts/physics_guided_core.py | e6be20aa8a060bad20881cb81bab4e3328282a163b0d3e9571c7c590443436ed |
| scripts/physics_guided_queue.py | 349266a29d56806b9f081fdc0ec69c7788746849f8ff50035248ef25c8602cf8 |
| scripts/physics_guided_lineage.py | da4365d425dfe91397f7651d0a8e873dabf289ba2d30a63f84c6b19d68778ef6 |
| scripts/r3_closd_isolated_entry.py | 601fd5557b71a8259d12575beb3848f206795a7c9f7cefd02e538e8c97d280e6 |
| scripts/run_reset_alignment_checks_v4.py | 5d0a961b6ce9903fd666fdbae56736a68eddd1402ddf72793fc000ee0dea98d2 |
| scripts/run_contact_sensor_v9_20261011.py | 20d6b483a2b34d8aa7f09d5643a7eabff0654301e711b9bba6747e88d293e0ce |
| scripts/contact_sensor_collect_v9.py | 04a98be5a702b6a0b60d8753d7e19f3cef096e04f823b366febe8ef7f22ea0ae |
| scripts/contact_sampling_collect_v5.py | 651bbaf6cd0a2ba675fdbd5d735d66ee5a3aeecab755060aa2b5628be8d328a4 |
| scripts/contact_sampling_collect_v8.py | 8373f5301e1022d7414c9415305edf08855db2e0a8d3efe000fa0b4dff861109 |
| scripts/metric_numeric_equivalence.py | fa84c42cc2c894adb4458413ddbac16abf760a4ded71449ee467c1c8451cc8b5 |
| scripts/physics_guided_numeric_lineage.py | 5cff8cd9fbf2eded3f3ec503c2b6afb8005f463963744b8f326c3a0c2a7a0512 |
| configs/physics_guided_20261011.json | 2d8fd0e49ce7ed37393e1962dcd711b5a6e8f26de7e0ca4ef48964a6fd081502 |
| configs/env-closd-baseline-20261010.json | e5f9e6b711b7c6221cfb01e3deab404c5849e521ec08a3d25dca1820a279fcf6 |
| docs/contact_sensor_v9_20261011.md | 967ad6cd3886105ff765814b667bd0839ab498bf9b6a066774810595cf611c3a |

versions.json SHA256 is 49affd08a2171d09cdd7561de07e4f738ca03a670cc1d747c6ff5b4d1ef4297b. source_manifest.json SHA256 is a7065f535ee7adbc8bac95c5be2bc527cf740d43eabe3a40105ce6c471bf95de, matching both versions.json and the reset_alignment_v4 measurement_manifest.json. The manifest contains the original four validation source clips 008349, 010957, 008203, and 003483 with their original captions and goals. The request binds val/native, seed11, round0, noise_round0, the matching config/manifest/version hashes, and frozen thresholds SHA256 451fae0a2b106dbd0170185d9a22bb6f3fe2583709e77979eb017cc3ce09315b. The thresholds equal the original physics_guided frozen thresholds byte for byte.

The inherited versions declare upstream commit d8950bdb78312fcd2844f336b3fc079e7368c87f, assets revision de7106b947b6f70700b5320d1cd61fef4a9ebc9b, generator weight SHA256 3ca68ae54975946b3fe5c6cd64a6eafc439b5fc43c166bc9a98d1c43aeef6618, controller weight SHA256 7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac, and text-motion evaluator SHA256 43e43d198f07b7e080749d1709326783d2bfb5810f56ed0c56c68951ee961d0d. Remote third_party/CLoSD git status --short is empty. These are preserved declarative bindings; model/asset loading and realized binding checks were not reached by this failed invocation.

## Unexecuted measurement gates

The document promises 12 original attempts and 120 frames if completed. None were recorded. Therefore seven reset-state and four reference-cache differences, realized target/text/noise identity, effective SimParams, policy observation mode/size, sensor bodies/metadata, and runtime model/asset hashes cannot be verified from V9 raw outputs.

Source inspection shows the intended passive sensors at L_Ankle, L_Toe, R_Ankle, R_Toe, with forward-dynamics forces disabled, solver-constraint forces enabled, and world coordinates. The source asserts self_obs_v == 1 and would record policy_observation_mode and sensor_metrics_used_for_selection=false. These are source intentions only because execution stopped before instrumentation. Realized sensor[120,12,4,6], root[120,12,13], DOF/action shapes, finiteness, and NPZ hashes are unavailable.

This witness fails the recording gate. It cannot certify repeatability, pure contact-force calibration, innovation efficacy, robot hardware validity, or downstream utility. The box/plane fixture's support reading does not validate articulated humanoid sensors. No result or efficacy claim is supported.

## Preservation

No retries, --run, source fixes, package/model/environment installation, source/version/result overwrite, deletion, or Overleaf edits were performed. Subsequent SSH calls were read-only inspection. The original failed V9 invocation and its artifacts remain retained. This local witness report is the sole file written by this witness agent.
