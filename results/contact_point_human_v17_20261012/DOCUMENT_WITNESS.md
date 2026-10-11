# Independent V17 document-following witness

Verdict: **DOCUMENT INVOCATION FAILED; collector completed and passive recording evidence is available.** This is not a calibrated foot-slip score or innovation result.

## Exact invocation and terminal state

Executed once, without modification, retries, extra simulation, installation or upstream edits:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_point_human_v17_20261012.py --check
```

Live execution handle: 54659. Terminal exit: **1**. Expected `CONTACT_SAMPLING_DOCUMENT_CHECK_PASS` absent. Remote wrapper state: `FAILED_RETAINED`, error `AssertionError()`.

The child collector completed 12/12 cases (120 frames) and printed `PHYSICS_GUIDED_COLLECTION_PASS 12 0`. Its ledger entry records exit 0 and 39.1640844941 seconds. The wrapper then failed in `physics_guided_numeric_lineage.py:148`: its exact output-key comparison expected only metadata, metrics and trajectories, whereas the collector lineage correctly added the new `native_contacts.json` artifact. This is a verifier/collector interface mismatch, not native contact unavailability. The document expectation of exit 0 is false for this version. No repeat_b or --run was executed.

## Read-only evidence

All human arrays stayed on the server. Only compact JSON comparisons were exported locally.

- Native contact records: **53,764**; errors: **0**; frame range 0..119.
- Actual structured field schema: env0, env1, body0, body1, localPos0, localPos1, minDist, initialOverlap, normal, offset0, offset1, lambda, lambdaFriction, friction, torsionFriction, rollingFriction.
- Recorded body IDs are within 0..23 or -1 (ground); both ground and inter-body contacts occur. All recorded contact force magnitudes are finite; 10,759 records have positive magnitude, including 10,189 ground records.
- Body-property snapshots: **12 environments x 24 bodies**; COM and mass retained. Foot COM offsets are nonzero. This does not prove which reference point exported linear velocities use.
- All 20 source/config/document SHA bindings matched. Both generator and tracker weight SHA bindings matched. Canonical environment SHA256: `81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780`.
- Manifest binding and all four lineage output hashes verified. Metadata SHA binding to native contact JSON verified.
- Initial reset gate passed; all seven manually reset vs first-action state maximum differences are **0**.
- Original goal, generation noise and generation prefix hashes match V11 repeat_a.
- CPU exports/GPU PhysX settings remain unchanged; dt 0.0166666675 s, substeps 2, contact collection 2; completion fence calls 242.
- Against V11 repeat_a, all nine trajectory arrays are exactly equal: actual, reference, quaternion, linvel, angvel, contact_force, valid, terminated, reference_frame. All five sensor/policy arrays are exactly equal: sensor, root, dof_pos, dof_vel, action. All arrays finite, all maximum absolute differences **0**.

## Boundary

Passive native-contact reading did not alter the saved V11 trajectory or actions in this attempt. API/schema access and instrumentation invariance are supported. Articulated slip calibration, velocity reference-point calibration, repeat-native-contact equality, condition/physical joint efficacy and downstream learning utility remain unproven. Thresholds and original goals were not changed. No Overleaf writes.

Compact evidence: `document_witness_evidence.json`, `document_witness_extra.json`. Collector and wrapper failures remain preserved remotely; any repair must use a separate version.
