# V17 passive humanoid contact-manifold diagnosis

Purpose: read native ground-contact features/offsets and per-body COMs at the same completed physics step as V11 state snapshots. Preserve frozen models, original goals, observations, actions, reset alignment, physics and original metrics. This is not innovation efficacy or a replacement foot-slip score.

Prerequisites: V16 independently verified native structured field names and unchanged known-box arrays; V11 CPU export setting remains frozen. Use the existing runtime; no installation. Budget ledger includes all earlier attempts, including V14/V15 failures. Data disk free >=20GiB, study output <15GiB, total <=72 GPUh and measurement <=4 GPUh. Existing GPU jobs remain untouched.

Fresh document-following witness executes this exact invocation once:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_point_human_v17_20261012.py --check
```

Expected exit 0 and `CONTACT_SAMPLING_DOCUMENT_CHECK_PASS`; 12 validation inputs, 120 frames, seed11, native actual-prefix mode. Preserve failed outputs and do not retry. Inspect saved native field schema, errors, body IDs, COM count, metadata/native-contact binding, reset gate, source hashes and saved trajectory equality against V11 repeat_a. First attempt is recording-only. Native read success alone does not validate articulated slip, velocity reference point, real contact force or downstream usefulness.

After witness passes, the owner may run the same invocation replacing `--check` with `--run`. It certifies cached repeat_a and performs repeat_b, saving `sampling_summary.json`; repeat contacts/state/action should be identical. SDK local_pos excludes contact thickness; offset denotes surface displacement. Do not assume exported linear velocity references the body origin or COM without separate calibration. Native contact lambda and all-substep net-contact force may differ in temporal aggregation.

No Overleaf experiment results are written. Synthetic fixtures may be published; raw human motion stays on the server.
