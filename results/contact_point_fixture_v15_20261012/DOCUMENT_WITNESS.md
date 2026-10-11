# Independent V15 document-following witness — 2026-10-12

Witness agent: /root/contact_point_v15_document_witness. Required fresh agent-follows-doc pass from run-experiment skill. Read the V15 invocation document, warm runtime ledger, V15 and V14 adapters, V15 runner, and V3 physics source before execution.

## Single documented execution

Executed this exact command once, with no retry and no additional simulation:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_point_fixture_v15_20261012.py
```

Terminal exit code 0; command returned in 4.061987 s before a live process/session handle could be returned. Exact stdout:

```text
NATIVE_CONTACT_FIXTURE_DOCUMENT_EXECUTION_COMPLETE UNAVAILABLE_OR_CALIBRATION_FAILED_RETAINED
```

Runner state: COMPLETE_DIAGNOSTIC. Inner fixture summary: FAIL_RETAINED. Measurement budget ledger records 2.621633481234312 s, exit code 0. Exit 0 certifies diagnostic completion only; native contact availability/calibration is not PASS.

## Retained schema failure

native_contacts.json contains zero persisted records and 414 caught exceptions, all ValueError('no field of name local_pos0'). Counts by fixture/environment are 120 stationary, 120 sliding, 83 airborne drop and 91 upward takeoff. Stationary steps 60–119 produce 60 such errors; airborne initial steps 0–19 produce none. This shows get_env_rigid_contacts returns iterable contact rows when contacts occur, and the adapter reaches local_pos0 indexing after successfully evaluating body0/body1/env0/env1/normal/lambda. It does not show the native API is absent. The V15 exporter fails because its snake-case local position field does not exist in the returned structured row.

Actual contact body IDs, contact point values, normal finiteness and manifold force finiteness cannot be independently certified because no row is persisted. native_airborne_has_no_ground_force=true is not a successful manifold validation when the recorded list is empty. No ground-body (-1) contact was successfully exported or verified. Retain UNAVAILABLE_OR_CALIBRATION_FAILED_RETAINED as exporter/calibration failure; do not relabel availability PASS or claim lack of SDK support.

Read-only SDK introspection without creating a simulator confirmed Python RigidContact properties body0/body1/env0/env1 are int; normal/local_pos0/local_pos1/offset0/offset1 are Vec3; lambda is float. Vec3 NumPy dtype is [('x','<f4'),('y','<f4'),('z','<f4')]. RigidContact has no dtype attribute and np.dtype(RigidContact) yields object, so these Python properties do not reveal the actual returned structured dtype. The method docstring declares numpy.ndarray[RigidContact]. The installed gym_38.so contains separate camel-case strings localPos0/localPos1/minDist/initialOverlap/lambdaFriction, while its Python properties and bundled property documentation use local_pos0/local_pos1/min_dist/initial_overlap/lambda_friction. Camel-case structured fields are a binding-based inference; the actual runtime returned dtype was not saved by V15 and is not certified by this witness. Read-only introspection's initial dtype-attribute query ended with AttributeError; no simulation or experiment retry followed.

## Saved known-physics evidence and invariance

All nine original V3 box-physics checks are true. Settled stationary net z force is 49.04329299926758–49.056922912597656 N for a 5 kg box (expected 49.05 N); constraint sensor z force is 49.036190032958984–49.063514709472656 N. Initial airborne net force and sensor force absolute maxima are both 0.0 N, with minimum height 1.4134434461593628 m in steps 0–19. This validates the tensor-level known box fixture, not manifold positions or humanoid contact.

A CPU-only postprocessing comparison of saved V15 arrays against BOTH saved V3 repeat_a and repeat_b shows:

| Array | Shape | dtype | All finite | Exact equality to each V3 repeat | Maximum absolute difference |
|---|---|---|---|---|---|
| state | 120 × 4 × 13 | float32 | true | true | 0.0 |
| net_contact_force | 120 × 4 × 3 | float32 | true | true | 0.0 |
| constraint_force_sensor | 120 × 4 × 6 | float32 | true | true | 0.0 |

Read-only instrumentation did not change any of these exported arrays. Local copied NPZ SHA256 is 9d16ec394f9aa5aa5a508638d7da9d35f4066b6b7e580dcdbde81629f5e86ef7, matching V15 summary and both V3 arrays. All seven local source/doc/config hashes match the recorded remote versions.json. V15 source SHA256 aa1659ee67bb02fee091d6524ad3e2adf115753b998e0f59ab87d2186518c555; V14 remains 87cc0bab8d5dd04158610f334eec19b8dfb52c8aea983c8d60cce8d5f1b1a0ea.

The V15 adapter changes only the V14 contact loop record variable name, avoiding V14's exported contact tensor shadowing. This witness made no experiment source edits, package installs, extra simulations, threshold changes, licensed humanoid data transfers or Overleaf writes. Failed V14 source/logs were not altered. Copied only V15 synthetic box-fixture outputs, underlying log, state, budget ledger and version manifest for local audit.

Feature positions exclude thickness; offsets identify the friction surface. Last-substep manifold force must not be equated with all-substep tensor force. No innovation efficacy, real force ground truth, humanoid calibration or downstream utility is established by this diagnostic.
