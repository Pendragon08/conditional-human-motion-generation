# Independent V16 document-following witness — 2026-10-12

Result: the one documented invocation exited 0 with `NATIVE_CONTACT_FIXTURE_DOCUMENT_EXECUTION_COMPLETE AVAILABLE_ON_KNOWN_BOX`. The saved native API evidence independently supports availability on the known box/plane fixture. This witness is a calibration result, not an innovation-efficacy, humanoid-contact, real-force, or downstream-utility claim.

## Execution and scope

I read `docs/contact_point_fixture_v16_20261012.md`, `.aris/compute/closd-baseline-20261010.md`, the V16/V15/V14 adapters, the V16 launcher, and the V3 parent. No root `AGENTS.md` exists in this repository. The documented warm runtime was used with no installs or rebuilds. I executed exactly the following PowerShell invocation once, with no retry and no extra simulation:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_point_fixture_v16_20261012.py
```

The command finished within the first tool response (4.308 seconds); there was no live session handle to report. Outer exit code: 0. Saved subprocess exit code: 0. Charged measurement time: 2.8739501368254423 seconds. Terminal state: `COMPLETE_DIAGNOSTIC`, native status `AVAILABLE_ON_KNOWN_BOX`.

The launcher checks remaining measurement/total budgets, /data free space >=20 GiB, new output below15 GiB, and GPU0 headroom before simulation. Their assertions did not fail. Its final state overwrites the transient resource snapshot, so this witness does not invent a numerical prelaunch utilization/free-space reading. No workloads were terminated. The underlying log confirms GPU PhysX on cuda:0 with GPU pipeline disabled, a warm extension (`ninja: no work to do`), and `CONTACT_BOX_FIXTURE PASS`. Three inherited `Incomplete articulation start index in function setRigidBodyStates` warnings remain in the retained log; no Python traceback or native contact parsing error is saved.

Only read-only retrieval and local CPU analysis followed the single run. No experiment source, physics, thresholds, existing result, human array, score, model, or Overleaf content was changed. V14/V15 negatives remain historical evidence.

## Actual native fields and contacts

The saved dtype names are:

`env0, env1, body0, body1, localPos0, localPos1, minDist, initialOverlap, normal, offset0, offset1, lambda, lambdaFriction, friction, torsionFriction, rollingFriction`.

For each required logical field, exactly one actual name matches after removing underscores and lowercasing. `local_pos0 -> localPos0` and `local_pos1 -> localPos1`; body0/body1/env0/env1/normal/lambda/offset0/offset1 map identically. V16 uses the actual dtype names rather than guessing a replacement field. Unused SDK fields remain in the recorded dtype inventory; this witness does not claim values for fields the adapter does not export.

`native_contacts.json` contains 3312 records and zero errors. All exported normal, magnitude, local-position and offset values are finite. Native magnitudes range from 0 to 562.5131225585938 N; normal norms are approximately1.0000001192. Each record has body0=0 (the one box) and body1=-1 (ground). Environment pairs are `(0,-1)`, `(1,-1)`, `(2,-1)`, `(3,-1)`, consistent with the corresponding box environment and shared plane. All body/environment IDs satisfy these constraints.

Record counts by fixture are stationary960, frictionless slide960, high airborne drop664, upward takeoff728. Stationary steps60–119 contain480 positive box/plane records across all60 steps, with individual positive magnitudes4.9778923988342285–7.287693977355957 N. High airborne steps0–19 have zero native records and zero exported net contact force. All twelve saved checks are true, including the unchanged nine V3 physics checks and three native availability checks.

Local positions are shape feature positions that exclude thickness; offsets locate the friction surface according to the documented SDK interpretation. I did not reconstruct or certify friction-point world coordinates. A finite feature position is not a humanoid contact-point calibration. Native manifold magnitudes are not assumed equal to the all-substep tensor export. Neither interface is real-force ground truth.

## Source and array verification

All8 version-manifest hashes match local bytes, including the untouched V3 parent and V14/V15 adapters. The V16 source hash agrees with both the manifest and summary:

`ef5fd4de241d542420c703192c64da752f057cbeb8292583527dcc3f449adb81`.

V3 parent hash:

`291f6c9ca8430a2ace6212ba305946c012986f8926540170b07093d503f85ea8`.

V16 launcher hash:

`413fc9cc44b2a675fdb66b5a0376f18348b66ea2f6f9200e51edae3f58d5e959`.

V16 document hash:

`77e18094913a7879b4f0fcb6fc6216f41b61fbb7b802d7ee5e59bc8bcbb26298`.

The env-spec file byte hash is `e5f9e6b711b7c6221cfb01e3deab404c5849e521ec08a3d25dca1820a279fcf6`, matching the launcher manifest. This raw file hash is distinct from the canonical serialization hash documented in the warm-runtime ledger; this witness does not conflate the two.

The saved NPZ hash independently agrees with the summary and both retained V3 archives:

`9d16ec394f9aa5aa5a508638d7da9d35f4066b6b7e580dcdbde81629f5e86ef7`.

CPU comparisons loaded all archives with `allow_pickle=False`. Against both V3 `repeat_a` and `repeat_b`, the following finite float32 arrays are exactly equal, with maximum absolute difference0:

| Export | Shape | V3 repeat_a | V3 repeat_b |
|---|---|---|---|
| state | 120 x4 x13 | exact | exact |
| net_contact_force | 120 x4 x3 | exact | exact |
| constraint_force_sensor | 120 x4 x6 | exact | exact |

This establishes that the additional read-only native query did not change these saved state/force exports in this fixture. It does not certify hidden PhysX state or general determinism outside these runs.

Native JSON SHA256: `9c4e3b7018b5f2b9bc609f3bf831ba45cca4ab169bd5f376093b437cf2b3082f`.

Underlying log SHA256: `433007b91d7880b5b1b0fee06facf0dc06511424c8598616a9b9fad8f30012e9`.

Machine-readable independent checks, exact hashes, field mappings, IDs and array comparisons are saved alongside this witness in `INDEPENDENT_DOCUMENT_CHECK.json`. The result does not erase the retained V14 variable-collision failure or V15 NumPy field-name parsing failure; it resolves only those adapter defects within the prescribed fixture scope.
