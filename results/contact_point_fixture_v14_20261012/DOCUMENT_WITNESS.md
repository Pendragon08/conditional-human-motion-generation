# Independent V14 document-following witness — FAILED_RETAINED

Date: 2026-10-12 (Asia/Shanghai). Independent fresh document-following agent.

## Verdict

The exact documented invocation was executed **once**, exited **1**, and did not emit `NATIVE_CONTACT_FIXTURE_DOCUMENT_EXECUTION_COMPLETE`. This is an **instrumentation code failure**, not evidence that the native API is unavailable and not a calibration PASS. The failure, log, budget charge and empty first_attempt directory remain on the server. No retry, additional simulation, source edit, package installation, training or licensed human-array publication was performed.

## Document followed and invocation

Read `docs/contact_point_fixture_v14_20261012.md`, `.aris/compute/closd-baseline-20261010.md`, both V14 scripts, the V3 parent fixture, and the shared runtime wrapper. No AGENTS.md was found at the repository root or checked D: ancestors. The frozen warm runtime and single-attempt policy were followed.

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_point_fixture_v14_20261012.py
```

The invocation completed in approximately 4.02 seconds; no live process/session handle remained. The saved measurement ledger charged 2.522087898105383 seconds, exit_code 1. Before launch, GPU0 used 699 MiB of 49140 MiB, GPU1 used 8842 MiB, and `/data` had 41 GiB available. The wrapper's existing resource guard and disk/budget assertions permitted this attempt; existing workloads were preserved.

## Actual retained evidence and failure mechanism

Server result root: `/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/contact_point_fixture_v14_20261012`.

`state.json` contains `{"status":"FAILED_RETAINED","exit_code":1}`. `first_attempt.log` confirms PyTorch 1.13.1+cu117, GPU PhysX on cuda:0, CPU export pipeline, and an unchanged cached gymtorch extension (`ninja: no work to do`). It retains three incomplete-articulation warnings and this terminal error:

```text
File ".../scripts/contact_point_fixture_v14_20261012.py", line 82, in main
AttributeError: 'numpy.void' object has no attribute 'detach'
```

The V3 parent names its net-contact tensor `contact`. The injected V14 loop uses `for contact in contacts:`, replacing that tensor binding with a structured NumPy contact row. On the following completed physics step, the unchanged parent calls `contact.detach()` while saving net force and fails. This establishes that a NumPy contact row was iterated in the first step, but its values were never durably written. It does not establish contact availability across the fixture, positive support forces, valid IDs, finiteness, or full calibration.

Read-only existence checks confirmed that all three expected outputs are absent:

- `first_attempt/native_contacts.json`
- `first_attempt/fixture_arrays.npz`
- `first_attempt/summary.json`

Consequently, there is no V14 saved-state/native-contact evidence to validate. Stationary support, airborne native forces, returned body IDs, field finiteness and read-only state equivalence against V3 remain **UNVERIFIED**. The document's exit0 expectation diverges from reality.

## SDK semantics and types (read-only, no simulation)

The bundled IsaacGym Preview4 `docs/api/python/gym_py.html` documents `get_env_rigid_contacts` as returning a NumPy array of RigidContact records. `struct_py.html` documents both body0 and body1 as rigid-body indices in the environment, with **-1 indicating the ground plane**; env0/env1 are environment IDs, with -1 shared/unrecognized. The one-box fixtures would require each reported body index to be either 0 or -1; actual returned IDs cannot be checked because records were not saved.

A read-only fresh `gymapi.RigidContact()` object confirmed body0/body1/env0/env1 are Python int; lambda is float; normal, local_pos0/local_pos1 and offset0/offset1 are Vec3. These are SDK object-property types, not verified dtypes of the missing recorded NumPy rows. An initial read-only attempt to inspect `gymapi.RigidContact.dtype` failed with AttributeError because this class exposes no such attribute; that diagnostic failure was preserved in the witness tool record and caused no simulation.

SDK documentation states that local_pos excludes thickness and is where the normal force is applied; offset is the displacement from the feature position to the surface where friction acts. Feature positions cannot be equated with friction application points. No assumption was made that native last-substep manifold forces equal the all-substep net-contact export. Native simulation forces are not real-force ground truth and do not establish humanoid or robot contact calibration.

## Existing V3 saved-array inspection

Read-only CPU inspection of V3 repeat_a/repeat_b, without rerunning physics, found:

| Array | Shape | dtype | Finite | V3 repeats equal |
| --- | --- | --- | --- | --- |
| state | (120, 4, 13) | float32 | true | true |
| net_contact_force | (120, 4, 3) | float32 | true | true |
| constraint_force_sensor | (120, 4, 6) | float32 | true | true |

The maximum repeat difference is 0 for each array. Existing V3 airborne first20-step max absolute net force is 0 N. Its saved summary reports all nine checks true; settled stationary net force is 49.04329299926758–49.056922912597656 N and sensor force is 49.036190032958984–49.063514709472656 N for a 5 kg box. These V3 observations provide baseline context only; **V14 instrumentation equivalence cannot be evaluated without V14 arrays**.

## SHA256 verification

Local V14 adapter/wrapper, V3 parent and document hashes match remote `versions.json` exactly. Raw file hashes are distinct from the ledger's canonical environment-spec hash and were not substituted for that hash.

| Source | SHA256 |
| --- | --- |
| scripts/contact_point_fixture_v14_20261012.py | 87cc0bab8d5dd04158610f334eec19b8dfb52c8aea983c8d60cce8d5f1b1a0ea |
| scripts/run_contact_point_fixture_v14_20261012.py | 9c1e6914afa2209a0c1ffd736100d932bfe6eda3474c008e6a03023d51f75c71 |
| scripts/contact_force_fixture_v3_20261012.py | 291f6c9ca8430a2ace6212ba305946c012986f8926540170b07093d503f85ea8 |
| scripts/run_reset_alignment_checks_v4.py (remote manifest) | 5d0a961b6ce9903fd666fdbae56736a68eddd1402ddf72793fc000ee0dea98d2 |
| configs/env-closd-baseline-20261010.json (remote raw file) | e5f9e6b711b7c6221cfb01e3deab404c5849e521ec08a3d25dca1820a279fcf6 |
| docs/contact_point_fixture_v14_20261012.md | cb24f7e7b4fc8bad36a0e7d567537ad8bcbe353c45d44e9418f1e00c5a243fac |

| Retained remote evidence | SHA256 |
| --- | --- |
| first_attempt.log | 75dc12f37219c38bf1171edf99724b43e7a114ded4f3bd40209dae3758b466f4 |
| state.json | af6e91081cb20e4b72376dc3dad58a40dfd42a801d7db7bb806fdaa9c0229eff |
| budget_ledger.json | 0deccef783d366e81c616b781fa5007f1669b58a7596785d3d063dc7d98d48c6 |
| versions.json | 0f866c856f42ace50f5ae2dd20a79745621c8dfb8ca8a6c89d6eb45483515ffa |
| queue.lock (empty) | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| V3 repeat_a/fixture_arrays.npz | 9d16ec394f9aa5aa5a508638d7da9d35f4066b6b7e580dcdbde81629f5e86ef7 |
| V3 repeat_b/fixture_arrays.npz | 9d16ec394f9aa5aa5a508638d7da9d35f4066b6b7e580dcdbde81629f5e86ef7 |

Independent final status: **CODE_FAILURE_RETAINED; V14 READY/PASS NOT ESTABLISHED**. Any correction must use a new version/output, preserve this failure, and retain its budget charge.
