# V43 independent document-following witness

Verdict: **PASS** for the documented execution, strict provenance, native float32 velocity rule, complete case retention and observed generation/replay consistency. This is a diagnostic witness, not an efficacy experiment or evidence of causality or scientific innovation.

## Actual invocation and terminal outcome

The exact command below was executed once. No relaunch, source edit, package installation or additional simulation batch was made.

~~~~powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_reference_velocity_v43_20261012.py --run
~~~~

The launch was followed through exec session **50565**, remote launcher PID **2836519**, first collector PID **2836542**, to **exit code 0**. Recorded terminal state is **COMPLETE_INTERFACE_DIAGNOSTIC** and summary status is **COMPLETE_REFERENCE_OBSERVATION_DIAGNOSTIC**. Exactly two charged measurement ledger rows have exit code 0. Planned and replay batch elapsed charges are 50.132825454697 and 49.6480085048825 seconds.

## Independent evidence

- Every execution-bound source hash matches versions.json; all frozen V28 anchors and the V36 external input bindings match. Source manifest, configuration and thresholds match their bound hashes. The locally read document matches the execution-bound remote document.
- Both execution-time lineages use bound_request_at_execution; requests are recomputed and equal, all three required output hashes match, complete receipts agree with trajectory hashes, and observation/initial-state archives match metadata hashes. No retrospective certification was performed.
- Each batch retains all **72 ordered cases** from **24 validation sources × 3 offsets**, plus all 72 metric rows. Both preserve three terminated first attempts at lengths 94, 100 and 108; the other 69 retain 120 frames. No failed case is discarded.
- Both traces contain 120 consecutive frames. Position, previous position and velocity shapes are 120×72×24×3; raw task observation shape is 120×72×574. All arrays are finite, reference_frame is exactly 0…119, and trace reference positions equal the applied trajectory reference array.
- Every recorded previous position equals the preceding actually applied position, with frame0 using its current position. Frame0 velocity is exactly zero.
- Native s_dt is 1/30. For every one of **622,080 velocity scalars per batch**, the recorded velocity is bitwise identical to float32(position − previous_position) multiplied by the rounded reciprocal of float32(s_dt): s_dt=0.03333333507180214 and reciprocal=29.999998092651367. This independently reconstructs the recorded native arithmetic with zero residual.
- A direct NumPy float32 division reconstruction is not bitwise identical: 113,395 scalars differ per batch, maximum 9.5367431640625e-7. Against a float64 formula, maximum absolute difference is 1.0132789611816406e-6 and maximum relative difference is 1.7203986515076118e-7. These original checks and differences are preserved in the JSON as arithmetic diagnostics; no simulator acceptance threshold was changed.

## Generation versus replay: every frame

All **120 frames** and **8,582 jointly valid frame-case pairs** were compared, not only the nine frames selected by the launcher summary. Recorded reference positions, previous positions, reference velocities and raw task observations are bitwise identical. Actual positions, reference positions, quaternions, linear/angular velocities and contact forces are also bitwise identical. Valid and terminated masks match.

The entire trace and trajectory arrays also match, including invalid tails. Both trajectory archive SHA256 hashes are **2f2ce526e783ea83fc47b8b1723ddcca34b93e87de84bcdc50823bff539c4cb4**. Both observation trace SHA256 hashes are **9bd7ecfc0c16a169a0015c969796621849796ba353ed7fbe540eaaddf0b02877**.

Thus, the discrepancy is absent in this recorded generation/replay pair. This finite observation does not establish a general guarantee, causal explanation, controller efficacy or innovation. Raw task observations were compared; normalized policy observations were not separately recorded.

## Preservation of the V42 planned reference

V43 planned reference positions are bitwise identical to the complete V42 generation trace and reference trajectory array, all **622,080 scalars**, with maximum difference zero. Ordered cases and original/proposed goals match. Generation prefix and noise hashes match. Historical V42 files were read only.

## Hashes and audit location

The accompanying JSON contains all source/output/input hashes, strict lineage receipts, per-frame comparisons, array shapes/dtypes, retained failure lengths and numerical characterization.

- Local document SHA256: 334de9f51f8a91ee420e86c7af56ed8768e7e3ab4b4d19b8d5dfb620e1f5aa80
- DOCUMENT_WITNESS.json SHA256: 92bfaca03d734699add960a354fff196ff133cc77eb98538102d7e553fd32366
- Native interval source SHA256: dff1ca9bfb15c3058914c326cba83a07f69cd91952cd4e03ecfe1ad3d391e394
- V43 versions.json SHA256: 9cc69c6318892df77657c6dac4e38e8f514d5a85c009addbea65d0edfbce1c5a

Only the local DOCUMENT_WITNESS.md and DOCUMENT_WITNESS.json files were written by this witness. Raw trajectory and observation arrays remain server-local.
