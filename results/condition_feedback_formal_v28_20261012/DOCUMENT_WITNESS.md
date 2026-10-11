# Independent V28 document-following witness, 2026-10-12

Status: **PASS for the documented measurement check only.** This is not evidence of C1 efficacy, semantic validity, hardware transfer, downstream benefit or innovation.

The fresh witness first read only the run-experiment skill, the frozen closd-baseline environment ledger and `docs/condition_feedback_formal_v28_20261012.md`. It then executed the exact documented command once, without retry, repair, source/config/threshold changes or `--run`:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_condition_feedback_formal_v28_20261012.py --check
```

Process exit code was **0**, with `CONDITION_FEEDBACK_FORMAL_DOCUMENT_CHECK_PASS` and queue state `WITNESS_PASS`. Captured console evidence is `DOCUMENT_CHECK_STDOUT.txt`. The sole measurement ledger entry is `val_planned_s11_r0`, 50.048971803858876 seconds, exit 0, no stop reason. All 72 first attempts are retained: 70 complete episodes and two terminations; valid lengths are 95, 109 and 120. These are recording counts, not efficacy conclusions.

After completion, the witness ran an independent read-only CPU audit of the bound files and outputs, without invoking the collector or queue again. All **60 checks passed**. See `DOCUMENT_WITNESS_AUDIT.json` and its reproducible read-only audit script `document_witness_audit.py`.

- All 14 ordered trajectory/sensor/action fields have identical dtype, shape and bytes to the frozen V24 planned seed11 batch. Nine fields are in `trajectories.npz`: actual, reference, quaternion, linvel, angvel, contact_force, valid, terminated and reference_frame. Five are in `sensor_diagnostic.npz`: sensor, root, dof_pos, dof_vel and action. The reference clock is exactly 0..119.
- Saved metrics are identical to V24 both as parsed objects and file SHA256. Independent recomputation through the bound calibrated V23 metric implementation returns zero changed floating values, zero maximum absolute difference and unchanged classifications. The existing calibrated numeric-verifier certificate passes. Frozen thresholds are identical to V24 and equal the original protocol configuration; no threshold changes were made.
- The original goal hash was independently recomputed from the float32 goal array and matches both the V28 metadata and V24: `d008fae01d78f5f8f752aa4791b824540aec5791fd36173306881bcb09584809`. Goals, headings, full source cases/text/offsets, episode duration, context steps and dt match V24. The original manifest is identical; the 72 validation cases expand in original order. The preselected 48 test sources, three offsets and three seeds imply the documented 432 inputs per arm.
- Native body properties, including COM, are retained for all 72 environments x24 bodies, finite and identical to V24. `dense_manifold_recording` is explicitly false in metadata. Empty `native_contacts.json` diagnostic records represent disabled dense recording, **not absence of physical contacts**. Its metadata hash binding passes; body-force and calibrated surface-proxy metrics remain present.
- Every file in the version binding, including V28 documentation, sources, calibrated verifier and configuration, matches its recorded SHA256. Upstream commit is `d8950bdb78312fcd2844f336b3fc079e7368c87f`, with no tracked `closd` diff. Generator, controller and evaluator weights, all external motion/semantic inputs, manifest, prior budget and fixed-reference summary bindings match. The execution request matches lineage and its config/manifest/threshold/version hashes; certification is `bound_request_at_execution`. Receipt and output hashes independently match.

No divergence was found in the documented check's expected output claims. One inherited metadata field requires careful interpretation: `versions.json` still contains `evaluation_inputs_per_arm=72` from the validation version, alongside the V28-specific `test_inputs_per_arm=432`. The actual manifest/config and V28-specific field match the documented test protocol; this witness does not rewrite the inherited field. No formal comparison/replay or future analysis was executed or certified. The check does not reach the analyzer's optional local TeX result export and does not certify any future reporting adapter.

Key execution hashes:

| Artifact | SHA256 |
| --- | --- |
| versions.json | afb2193a00d15a4c9e2fe8addbab5a1968ac122f59d80278670b7f9aa3d9421a |
| runtime_witness.json | ce07574de3dde3d800b4b6baa567f60b80131f6290fcc45558b6f1c74a9a6468 |
| receipt.json | 39bd5791023e8338e897ffe02590aaac77569920ccb8a0b03f3ef7140a02a435 |
| trajectories.npz | 45486979e705ce28f3e5e10976814ce901f3008ea902e5e23d326a5c0a4311bd |
| sensor_diagnostic.npz | 46b69111fe876af5250accb2c3a645005c14de0319e6ee2da158c1566bdee689 |
| metrics.json | 565a08acfcd9741f0da1c16362fe72d8ee9eae99f80dbee7986a6f660f65fb93 |

The report and compact audit evidence are saved in the local and server `results/condition_feedback_formal_v28_20261012/` directories. All earlier failed attempts and witness records were preserved.
