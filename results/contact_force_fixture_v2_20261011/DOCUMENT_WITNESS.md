# Document-following witness: contact-force box fixture v2

Result: PASS for documented execution and independently recomputed box calibration, first attempt repeat_a retained. This is an instrument-buffer diagnostic, not method efficacy. No retry, source edit, environment change or --run was performed by this witness.

## Document execution

The exact documented command was executed exactly once:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_force_fixture_v2_20261011.py --check
```

Observed process exit code 0 and `FIXTURE_DOCUMENT_EXECUTION_COMPLETE repeat_a PASS`. Budget ledger contains one measurement attempt, repeat_a, 2.721931781619787 seconds, exit 0. The completion marker alone was not used as calibration evidence. Original raw arrays and summary remain on the remote and were copied locally into repeat_a/. Local array checksum matches the remote and recorded summary.

Read run-experiment skill, its shared environment contract, provider ledger, invocation document and fixture source. The delegated contract path ending `SKILL.md/shared-references/compute-env-contract.md` did not exist; the skill explicitly resolves its actual reference to `C:/Users/Pende/.codex/skills/shared-references/compute-env-contract.md`, which was read. There was no documented CLI divergence. The contract path correction was a read-path correction only and did not alter the invocation.

## Independent raw-array verification

A separate read-only NumPy computation loaded the retained NPZ without importing or running the simulator fixture. Every recorded check was recomputed from states, net-contact forces and constraint-only force sensor readings. Complete values are retained in WITNESS_RECOMPUTE.json. All values are finite; state shape is (120,4,13), net_contact_force (120,4,3), constraint_force_sensor (120,4,6). Measured actor masses recorded by the simulator are four times 5.0 kg, so each expected support force is 49.05 N under 9.81 m/s² gravity. The geometric density construction independently agrees: 0.2 × 0.2 × 0.1 m³ × 1250 kg/m³ = 5 kg.

| Predeclared check | Independently recomputed worst value | Threshold | Result |
|---|---:|---:|---|
| Frictionless horizontal net force, all frames/boxes/components | 0.0001239193 N | <1 N | PASS |
| Resting height error, stationary box final30 frames | 0.002000507 m | <0.01 m | PASS |
| Resting linear speed, stationary box final30 frames | 0.0012657002 m/s | <0.02 m/s | PASS |
| Net normal support error relative to 49.05 N, final30 frames | 0.0002767114 (0.0276711%) | <10% | PASS |
| Constraint sensor normal support error, final30 frames | 0.0002815332 (0.0281533%) | <10% | PASS |
| Sliding vx error relative to 0.2 m/s, final30 frames | 0.00000359118 m/s | <0.02 m/s | PASS |
| High-drop airborne contact force, first20 frames/components | 0 N | <1 N | PASS |
| High-drop airborne constraint sensor force, first20 frames/components | 0 N | <1 N | PASS |
| High-drop airborne center height, first20 frames | minimum1.41344345 m | >0.5 m | PASS |

All nine independently computed booleans exactly match summary.json. Settled stationary net normal force ranges49.03642654–49.05689240 N; sensor normal force ranges49.03619003–49.06351471 N. The fourth upward-takeoff trajectory is retained and finite; no new takeoff pass threshold was invented.

## Source and output identity

Each of the five current remote file hashes matches versions.json, and its current local counterpart matches too:

| File | SHA256 |
|---|---|
| scripts/contact_force_fixture_v2_20261011.py | 552b2803c9308297b6254776361c627683f0f651171a3b970b028ce2f56050b1 |
| scripts/run_contact_force_fixture_v2_20261011.py | fc2c781671b6bde83aca675d6d0db35be960a8f693358a8734b9cbad8a4c85b3 |
| scripts/run_reset_alignment_checks_v4.py | 5d0a961b6ce9903fd666fdbae56736a68eddd1402ddf72793fc000ee0dea98d2 |
| configs/env-closd-baseline-20261010.json | e5f9e6b711b7c6221cfb01e3deab404c5849e521ec08a3d25dca1820a279fcf6 |
| docs/contact_force_fixture_v2_20261011.md | a570761284472d26efd61ec07f1aee10f7224cbf789656304ce5d2163cfc394d |

NPZ SHA256: `4f7c2b1e4976ad6ed1a7f345d388559a43fc67c68e891154d010ae24e09383e6`, matches summary and retained local copy. Summary source_sha256 matches fixture source. Canonical parsed/sorted/compact JSON environment SHA256 is `81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780`, unchanged from warm-runtime ledger. Remote repository HEAD at inspection: `e1ef719d6702d15a5ab211db0e55be438b2d1c88`; per-file hashes above bind the actual diagnostic overlay rather than assuming repository HEAD alone binds it.

## Interpretation boundary

The test clears only the derived net-contact export tensor before simulate; native simulate/fetch/refresh and constraint sensors remain as documented. Passing support and airborne checks prevents falsely accepting zeroed contact as a correction. This witness certifies only the recorded known-mass box/plane calibration for repeat_a. It does not establish humanoid contact-point truth, articulated sensor validity, motion quality, real-force ground truth, real robot utility or a method innovation. A causal instrumentation claim still requires repeat_b and comparison of state and constraint-sensor arrays against the original fixture as specified in the document; that comparison is outside this witness invocation.
