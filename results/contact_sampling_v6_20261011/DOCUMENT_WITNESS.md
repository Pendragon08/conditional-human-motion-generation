# V6 independent document-following witness

Date: 2026-10-11
Verdict: FAIL — the documented invocation exits successfully but its saved evidence does not satisfy the V6 completion-fence measurement contract.

## Scope and invocation

The fresh witness read only the run-experiment skill, its shared compute-environment contract, the closd-baseline environment ledger, the V6 runtime protocol, and result/manifest evidence after invocation. It performed the documented SSH command verbatim exactly once:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_sampling_v6_20261011.py --check
```

Exit code: 0. Standard output contained both expected sentinels:

```text
CONTACT_SAMPLING_BATCH_PASS repeat_a
CONTACT_SAMPLING_DOCUMENT_CHECK_PASS
```

No retry, source edit, installation, additional collection, or --run was performed by this witness. Subsequent remote commands only listed/read result evidence and recomputed hashes/array summaries.

Authoritative server output: `/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/contact_sampling_v6_20261011`.

## Verified saved evidence

- `receipt.json`: status complete, cases 12, completed 12, joint_pass 0; collection elapsed 29.326314687728882 seconds.
- `trajectories.npz`: 120 frames for each of 12 retained cases; all valid lengths equal 120; no termination; all arrays finite; reference_frame exactly 0..119. Positions have shape [120, 12, 24, 3], orientations [120, 12, 24, 4], contact_force [120, 12, 24, 3]. No additional attempts were launched by this witness. versions.json declares all_attempts_retained=true.
- `metadata.json`: initial_reset_gate_pass=true. All seven manual-reset versus first-action differences are exactly zero: root states, DOF position, DOF velocity, rigid-body position, rotation, velocity, and angular velocity. Four initial rigid-cache consistency differences are also zero.
- Initial-state archive SHA256 matches metadata: 866765c4e3c965f532efae1502cd2200c52bca9271ec41d195c52fa26740abed.
- `metric_numerics.json`: PASS_ROUNDOFF_ONLY, no differences, maximum absolute difference 0, classifications_identical=true, thresholds_changed=false, saved_scores_unchanged=true. This is a numerical recomputation certificate only.
- The lineage request equals requests/repeat_a.json. Actual file hashes match all three lineage output hashes and request-bound manifest, threshold, and version hashes.
- `budget_ledger.json`: measurement repeat_a, exit_code 0, 32.04443717002869 seconds, equivalent to approximately 0.008901233 GPU-hours for one GPU. All output files together occupied 2,031,105 bytes when checked. This attempt remains chargeable and preserved.

Verified output hashes:

| Artifact | SHA256 |
|---|---|
| metadata.json | 193e61a56015dbfef0d2fc3dcca6d5878f87cd1b67b66fb162b3965f0b15258e |
| metrics.json | caa85ddc251585e699fe8e8c85ccb9ca3906972e750a773760e5c94bd8381d56 |
| trajectories.npz | 7da49e1acef9ece7500a85e746ac381ea28501fff83e2c744707dc1130430720 |

## Document versus reality divergences

1. **No V6 completion-fence evidence.** The protocol requires recorded completion fences, at least 120 for the completed 120-frame collection. No completion-fence counter, fence event record, or reset-fence record appears in metadata or the trajectory archive. The fence requirement cannot pass.
2. **Wrong collector binding.** Saved adapter_source_sha256 is 651bbaf6cd0a2ba675fdbd5d735d66ee5a3aeecab755060aa2b5628be8d328a4, exactly the versions.json binding for scripts/contact_sampling_collect_v5.py. versions.json binds the V6 launcher and V6 document but contains no V6 collector entry. Thus the result does not certify that the V6 collector ran. Separately, the parent reported observing the live child command invoking contact_sampling_collect_v5.py; the independent saved evidence is consistent with that deployment defect. This witness did not read or repair launcher/queue source.
3. **Missing contact-sampling instrumentation.** Saved metadata exports neither CUDA synchronization/sampling stage details nor contact-tensor mutation measurements around reward. The protocol promises retention of V5 CUDA synchronization and measurement of reward mutation; those claims are not established by the saved evidence.
4. **Simulator parameters are partly documented.** resolved_config.yaml records the requested GPU pipeline, substeps=2, and PhysX settings (solver_type=1, position iterations=4, velocity iterations=0, contact_offset=0.02, rest_offset=0.0). The log confirms GPU PhysX/cuda:0/GPU pipeline. Metadata lacks a separately exported effective runtime SimParams record, so runtime parameter verification is incomplete.
5. **Success state overclaims the witness.** state.json says WITNESS_PASS and the launcher prints DOCUMENT_CHECK_PASS despite the absent fence evidence and missing V6 collector binding. Exit/sentinel success is therefore insufficient to declare this V6 protocol validated.

## Interpretation and disposition

Preserve repeat_a and its budget charge. This document-following witness fails the V6-specific measurement contract. No inference about completion-fence efficacy, force repeatability, task semantics, robot hardware, model/controller quality, or downstream utility follows from this invocation. A single batch cannot establish repeatability; its zero joint pass count does not validate a physical metric. Do not silently rerun this directory or proceed to repeat_b on the strength of its success sentinel.
