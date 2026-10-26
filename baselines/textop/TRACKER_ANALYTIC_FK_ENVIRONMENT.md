# Online prerequisite: named tracker-URDF CPU FK

Warm generator environment from SERVER_NOTES.md, no installation or shared edits.
This CPU-only profile reads the actual tracker URDF, including fixed links, and
compares all 258 previously captured SDK poses for all 30 tracked rigid bodies.
It does not use the GMR visual model, dispatch SDK, optimize or integrate physics.
The original teacher/generation/SDK/video files and all private inputs are immutable.

Spec: configs/tracker-analytic-fk-spec.json. Bounds were declared before execution:
position 50 micrometers, rotation 20 microradians, linear/angular derivative errors
0.005 m/s and rad/s. Root WXYZ, URDF RzRyRx origin, named29joint order; quaternion
comparison permits q/-q. Last derivative is provisional for streaming: consumers
must have one additional real pose and must never consume unproduced future data.
Parent analytic_fk_author_v1 actually exited0: position3.17e-6m, rotation1.79e-6rad,
linearvelocity1.264e-4m/s, angularvelocity5.51e-5rad/s; joints exactly equal.

Fresh witness must independently execute exactly once on the original server:

```text
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 0-3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/env/bin/python /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/validate_tracker_analytic_fk.py --run analytic_fk_document_v1
```

Use session.py SSH helper from local provider_tools (not a server-side session.py).
Before dispatch check run absent, real CPU/RAM/disk, source/spec/input SHA and
author receipt. Capture actual command stdout/stderr/exit to a new witness folder;
do not overwrite or fix files, rerun author, install anything, or start SDK/GPU.
Verify all258 frames,30 bodies,29 namedjoints, finite metrics/bounds, zero GPU and
model/physics updates. Compare both receipts and actual source bytes. Keep raw
arrays private. Report any document divergence and all failures, including whether
the strict origin/quaternion conventions are consistent. Only REPORT/SUMMARY and
aggregate receipts may be published. This is an FK prerequisite, not online,
semantic, physics-success or full-paper acceptance.
