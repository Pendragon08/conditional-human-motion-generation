# V7 independent document-following witness: FAIL

Date: 2026-10-11. Fresh document-following agent. Measurement diagnostic only.

Read the run-experiment skill, shared compute-environment contract, provider ledger and V7 invocation document. Project-root AGENTS.md was absent; the ledger/document supplied the SSH/runtime information.

## Exact invocation, once

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_sampling_v7_20261011.py --check
```

Executed exactly once. Exit **1**. Neither CONTACT_SAMPLING_BATCH_PASS repeat_a nor CONTACT_SAMPLING_DOCUMENT_CHECK_PASS appeared. The retained root state is FAILED_RETAINED with RuntimeError('Probe failure retained: repeat_a'). The collector traceback ends with:

```text
AssertionError: Full exported reset state/cache differs before first action
```

The traceback identifies contact_sampling_collect_v6.py; the queue also explicitly invokes that collector. The V7 invocation repair is present, but initialization does not pass.

## Document versus observed evidence

Remote raw directory:
`/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/contact_sampling_v7_20261011/raw/repeat_a`.

It contains only initialization_failure.json, initial_states.npz and resolved_config.yaml.

| Document requirement | Observed result |
|---|---|
| Exit 0 and both pass sentinels | FAIL: exit 1, both absent |
| Correct collector | PASS: V6 collector appears in queue/traceback; recorded adapter hash matches the bound source |
| 12 retained first attempts | Twelve case IDs and finite bootstrap/first-action snapshots preserved; no collected trajectories or completion receipt. Completed collection unverified |
| Seven first-action state differences are zero | PASS: independently recomputed from NPZ, all seven exactly zero; bootstrap array hashes match diagnostic hashes |
| Exported reset gate passes | FAIL: initial_reset_gate_pass=false; all four native rigid-body cache comparisons are nonzero |
| Fence protocol recorded | Present in failure diagnostic: fetch_results(True) after native physics step and native reset; no extra simulate or tensor refresh |
| At least 120 recorded completion-fence calls | UNVERIFIED: completion_fence_calls absent; first-action gate aborts before reward recorder writes this count |
| Simulator parameters recorded | Present; values below |
| Receipt and numerical certificate | UNAVAILABLE: no receipt, metadata, numerical certificate, trajectories or metrics |
| Environment/source/threshold bindings | All 15 bound source hashes and source-manifest hash match; canonical env hash matches ledger; frozen thresholds byte-identical to original |

The seven zero differences alone do not satisfy the conjunctive gate. Recorded cache differences are:

| Tensor versus native reset cache | Maximum absolute difference |
|---|---:|
| _rigid_body_pos | 0.035006970167160034 |
| _rigid_body_rot | 1.4754348993301392 |
| _rigid_body_vel | 3.7104039192199707 |
| _rigid_body_ang_vel | 6.5190958976745605 |

The retained NPZ lacks the native reset cache arrays, so these cache differences are reported collector evidence, not independently recomputable from the NPZ. The seven exported-state comparisons and corresponding hashes were independently verified.

The diagnostic records three full resets before first action, zero upstream warmup diffusion calls, twelve zero progress counters, motion IDs 0..11, zero motion start times and zero global offsets. Four source IDs 008349, 010957, 008203 and 003483 each have offsets -0.15, 0 and 0.15; generation/environment/top-level seed 11.

Simulator parameters: dt=0.01666666753590107, substeps=2, use_gpu_pipeline=true, contact_collection=2, physx_use_gpu=true. Episode/control dt is separately 0.03333333507180214. contact_mutation_during_reward_max_N is initialized to zero before this failure; reward was not reached, so it does not witness measured zero mutation.

## Hashes, preservation and limits

Independently computed raw-file hashes:

| File | SHA256 |
|---|---|
| initial_states.npz | 8ed380ac7f6118adb4b8d7a3748a7b9a59d86914b56e6403c8cb490aa5aa2a8d |
| initialization_failure.json | 31ded6c8835abb85d884253c97881342a637e9cb621beea9b8ab7eadd5583eac |
| resolved_config.yaml | 7892b948f8e96cf15acda7f37b72505164e49b8c4ef076f1f39871c89a0958ea |

The initial-state archive hash matches the diagnostic; all NPZ arrays are finite and have twelve rows. Queue SHA256: 53189d77ff2f1334f567535b4f0a65b0e53b1a16921fce0f5e77978a6857becf. Collector SHA256: 32c920c81fee452ef8e0bff66120f7f663430dfbceb74ca281f640496a982704. Canonical env-spec SHA256: 81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780.

Budget ledger retains repeat_a, exit 1, 28.033439487218857 seconds (about 0.007787 GPU-hours on its single bound GPU); the failed attempt remains chargeable to both applicable cumulative caps.

No retry, source modification, --run, package install, deletion or threshold change was performed. V6 and all previous outputs were preserved. No repeat_b was executed by this witness. V7 cannot be declared ready. This failed diagnostic establishes no force-repeatability, efficacy, task-semantics, hardware or downstream-utility claim.

## Independent read-only verification evidence

```json
{
  "raw_file_sha256": {
    "initial_states.npz": "8ed380ac7f6118adb4b8d7a3748a7b9a59d86914b56e6403c8cb490aa5aa2a8d",
    "initialization_failure.json": "31ded6c8835abb85d884253c97881342a637e9cb621beea9b8ab7eadd5583eac",
    "resolved_config.yaml": "7892b948f8e96cf15acda7f37b72505164e49b8c4ef076f1f39871c89a0958ea"
  },
  "aux_file_sha256": {
    "versions.json": "9d9804be63810b6f95d8d99e1fb5d1dd5baf4347b2c26e90ac05262a54cc9cc0",
    "source_manifest.json": "a7065f535ee7adbc8bac95c5be2bc527cf740d43eabe3a40105ce6c471bf95de",
    "frozen_thresholds.json": "451fae0a2b106dbd0170185d9a22bb6f3fe2583709e77979eb017cc3ce09315b",
    "repeat_a.json": "f9b1aae2490036d918ceaef2f3220b8e8a5d1e15b8c40b9c74cdc388ec42e2b7",
    "state.json": "872cb84c599f5501ceb00a81a065413ac6b2689084ee0baa30991dca862f48b6",
    "budget_ledger.json": "bd94156380e98f8b8cfcbffe0bf8b8f1c17b8404e307f1807199320561c8c4e0",
    "repeat_a.log": "dba0de547d1206ddfadbfa568d5d5b5adcf362fb2647da557aef49da483aea27"
  },
  "bound_source_hash_checks": {
    "scripts/reset_aligned_collect_v4.py": true,
    "scripts/physics_guided_collect.py": true,
    "scripts/physics_guided_core.py": true,
    "scripts/physics_guided_queue.py": true,
    "scripts/physics_guided_lineage.py": true,
    "scripts/r3_closd_isolated_entry.py": true,
    "scripts/run_reset_alignment_checks_v4.py": true,
    "scripts/run_contact_sampling_v7_20261011.py": true,
    "scripts/contact_sampling_collect_v6.py": true,
    "scripts/contact_sampling_collect_v5.py": true,
    "scripts/metric_numeric_equivalence.py": true,
    "scripts/physics_guided_numeric_lineage.py": true,
    "configs/physics_guided_20261011.json": true,
    "configs/env-closd-baseline-20261010.json": true,
    "docs/contact_sampling_v7_20261011.md": true
  },
  "source_manifest_hash_match": true,
  "frozen_thresholds_equal_original": true,
  "env_spec_canonical_sha256": "81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780",
  "initial_archive_hash_match": true,
  "npz_shapes": {
    "bootstrap__humanoid_root_states": [
      12,
      13
    ],
    "bootstrap__dof_pos": [
      12,
      69
    ],
    "bootstrap__dof_vel": [
      12,
      69
    ],
    "bootstrap__rigid_body_pos": [
      12,
      24,
      3
    ],
    "bootstrap__rigid_body_rot": [
      12,
      24,
      4
    ],
    "bootstrap__rigid_body_vel": [
      12,
      24,
      3
    ],
    "bootstrap__rigid_body_ang_vel": [
      12,
      24,
      3
    ],
    "first_action__humanoid_root_states": [
      12,
      13
    ],
    "first_action__dof_pos": [
      12,
      69
    ],
    "first_action__dof_vel": [
      12,
      69
    ],
    "first_action__rigid_body_pos": [
      12,
      24,
      3
    ],
    "first_action__rigid_body_rot": [
      12,
      24,
      4
    ],
    "first_action__rigid_body_vel": [
      12,
      24,
      3
    ],
    "first_action__rigid_body_ang_vel": [
      12,
      24,
      3
    ],
    "first_action_contact_forces": [
      12,
      24,
      3
    ]
  },
  "npz_finite": {
    "bootstrap__humanoid_root_states": true,
    "bootstrap__dof_pos": true,
    "bootstrap__dof_vel": true,
    "bootstrap__rigid_body_pos": true,
    "bootstrap__rigid_body_rot": true,
    "bootstrap__rigid_body_vel": true,
    "bootstrap__rigid_body_ang_vel": true,
    "first_action__humanoid_root_states": true,
    "first_action__dof_pos": true,
    "first_action__dof_vel": true,
    "first_action__rigid_body_pos": true,
    "first_action__rigid_body_rot": true,
    "first_action__rigid_body_vel": true,
    "first_action__rigid_body_ang_vel": true,
    "first_action_contact_forces": true
  },
  "verified_seven_differences": {
    "_humanoid_root_states": 0,
    "_dof_pos": 0,
    "_dof_vel": 0,
    "_rigid_body_pos": 0,
    "_rigid_body_rot": 0,
    "_rigid_body_vel": 0,
    "_rigid_body_ang_vel": 0
  },
  "manual_state_array_hash_checks": {
    "_humanoid_root_states": true,
    "_dof_pos": true,
    "_dof_vel": true,
    "_rigid_body_pos": true,
    "_rigid_body_rot": true,
    "_rigid_body_vel": true,
    "_rigid_body_ang_vel": true
  },
  "first_action_array_sha256": {
    "_humanoid_root_states": "f4a76036b5c8bd15503dde058ff8178d6f9471e1615a8b3bcdf0a8efd358a817",
    "_dof_pos": "c67950ecc2be81e44e31735fc73c76ccdebe5e8e72091b5d055f453ec021af69",
    "_dof_vel": "976f1cfc280a9691eb48d45880faac482c4603fac6d774e8dd7f2009a18e6f22",
    "_rigid_body_pos": "ce673d0ad9e2c5db94f0fa546a5d5fc74c6781033e4f7ea6f743632e4677d69c",
    "_rigid_body_rot": "2d917218cb918eabb38e28f0e4341ddc07c12aa6e8a56a3fa1f20cfe75937c56",
    "_rigid_body_vel": "29b241cfb4eb2129db7e144cb4e108625f02486bbc5c56359759d04fade1d8e3",
    "_rigid_body_ang_vel": "7e852da9f927bfea1424619a89bc852fa67237c5143e93049ec289ac48393d46"
  },
  "case_ids": [
    "008349:-0.15",
    "008349:0.0",
    "008349:0.15",
    "010957:-0.15",
    "010957:0.0",
    "010957:0.15",
    "008203:-0.15",
    "008203:0.0",
    "008203:0.15",
    "003483:-0.15",
    "003483:0.0",
    "003483:0.15"
  ],
  "absent_expected_files": [
    "metadata.json",
    "receipt.json",
    "numeric_certificate.json",
    "numerical_certificate.json",
    "trajectories.npz",
    "metrics.json"
  ],
  "completion_fence_calls_recorded": null,
  "initial_reset_gate_pass": false,
  "cache_consistency": {
    "_rigid_body_pos": 0.035006970167160034,
    "_rigid_body_rot": 1.4754348993301392,
    "_rigid_body_vel": 3.7104039192199707,
    "_rigid_body_ang_vel": 6.5190958976745605
  },
  "protocol": "fetch_results(True) after native physics step and native reset; no extra simulate or tensor refresh",
  "sim_params": {
    "dt": 0.01666666753590107,
    "substeps": 2,
    "use_gpu_pipeline": true,
    "contact_collection": 2,
    "physx_use_gpu": true
  },
  "root_status": {
    "status": "FAILED_RETAINED",
    "error": "RuntimeError('Probe failure retained: repeat_a')",
    "time": 1791732163.8719213
  },
  "budget": [
    {
      "stage": "measurement",
      "label": "repeat_a",
      "seconds": 28.033439487218857,
      "exit_code": 1
    }
  ]
}
```

