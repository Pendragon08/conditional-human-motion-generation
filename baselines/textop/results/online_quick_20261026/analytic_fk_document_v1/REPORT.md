# Fresh CPU FK document witness

The documented `analytic_fk_document_v1` command ran verbatim exactly once and returned **0**, with empty stderr and no timeout. All 69 independent matrix-audit checks passed. The audit read the existing SDK export and did not import the FK implementation or launch the SDK.

All 258 frames, 30 named tracked bodies and 29 named joints were checked. The independent URDF traversal covered 40 links and 10 fixed joints. Every reference field had its declared shape and finite values; quaternion norms and 50 Hz rate passed.

| Aggregate | Actual documented result | Predeclared upper bound |
|---|---:|---:|
| Position | 3.16915155364e-06 m | 0.00005 m |
| Rotation | 1.78687081519e-06 rad | 0.00002 rad |
| Linear velocity | 0.000126361846924 m/s | 0.005 m/s |
| Angular velocity | 5.51046113921e-05 rad/s | 0.005 rad/s |
| Joint position / velocity | 0 / 0 | Exact equality |

Independent matrix FK reproduced the position and both body-velocity maxima exactly. Its float64 rotation-matrix comparison gave 1.80385227839e-06 rad; the slight difference from the documented quaternion comparison follows the separate representation/float32 rounding path. Both meet the same unchanged bound. Author and document aggregate receipt fields match exactly; elapsed time differs as expected.

The reviewed source uses root WXYZ quaternions, URDF origin `Rz(yaw) Ry(pitch) Rx(roll)`, parent-rotated origin translation, and joint-axis rotation after the origin rotation. Fixed links are included. The actual, hash-bound SDK exporter maps the same named joint order and subtracts environment origins. Its quaternion delta algebra equals `q_next * inverse(q_prev)`, the world-frame convention reproduced by the independent matrix log `R_next R_prev^T`. Quaternion sign equivalence is handled correctly. Linear derivatives use centered differences internally and one-sided endpoints; angular derivatives average adjacent world rotation vectors internally and use one-sided endpoints. A last provisional streaming derivative needs one additional real pose before consumption.

One preflight difference is preserved: the local analytic JSON spec uses CRLF while the server uses LF. A supplemental exact-byte comparison therefore failed before dispatch. Read-only diagnosis established identical normalized text, parsed JSON and canonical hash; all actual server source/spec/input bindings passed. No file was fixed, no invocation retried, and the documented invocation itself had no divergence. All executable sources and the other inspected specs were byte-identical locally and remotely. Post-run source and input hashes were unchanged.

The command took 0.135882 seconds wall time, with 0.010825 seconds in the FK kernel and 37476 KiB maximum child RSS. Preflight load averages were [0.15, 0.12, 0.09]; disk free was 6804168704 bytes. CUDA was empty and affinity was CPUs 0–3. Source review and actual receipts show zero new SDK/GPU calls, optimizer updates and physics steps. No packages, profiles or inputs were modified. Raw motion arrays remained on the server.

Receipt SHA256: `cf7874dd9735d76f50e489df76f0fa4527879a922f3ff2384f85122c3d2721d3`. Actual-exit evidence SHA256: `8e7c11a91c61f516476609bf35a34cc0787f5d33328099514b967fc9b8a73fdb`. Server spec SHA256: `144415c752b85a2e71d0340764f874dad50a70f202a2c9fdb3179d5b7fc5e624`. Canonical spec SHA256: `b5aecec5cfaa054afe7d1c65d7786cb9be1ea8b2747429d3cd4d8bf4b371795d`. Author receipt SHA256: `b9fb434953667bebc0d91d9f4f43bac31ca32eb25b33374073aa6fdaa5381cba`.

An initial local evidence text snapshot failed its hash check because text reading normalized line endings. That failed helper/snapshot is preserved; a separate raw-byte download was SHA-verified for every archived source, URDF and author receipt. This did not rerun the documented command or alter server files.

The actual server author receipt was independently checked. Parent supplied its exit0 status; no separate author exit proof file was found in the inspected analytic output directory or matching receipts. Author was not rerun.

Raw UTF-8 evidence, original failure, exact stdout/stderr/return code, source snapshots and the independent matrix audit are private at `D:\具身智能\TextOp复现_20261024\diagnostics\analytic_fk_document_v1`. This validates only the CPU FK prerequisite against the existing offline SDK export. Online behavior, semantic quality, physics success and full-paper acceptance remain unverified.
