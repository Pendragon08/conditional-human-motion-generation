# Frozen execution and amended diagnostic provenance

This experiment tests a frozen generator and a frozen human physics tracker.
It is an amended diagnostic study, not an untouched confirmatory or preregistered
experiment. The primary endpoint and numerical limits are not relaxed after test
results. Versioned source, inputs, selections, all attempts and failures are retained.

`results/physics_guided_20261011/versions.json` records the execution snapshot.
Post-run publication is performed by `scripts/publish_physics_guided_final.py` and
recorded separately in `publication_versions.json`. The publication helper does
not alter trajectories, measurements or the frozen execution version.

The queue certifies reused outputs against the exact case list, fixed conditions,
paired noise, references and recomputed metrics. Early outputs without a historical
execution request are explicitly marked as retrospectively verified. Later
input-bound requests are preserved; compatibility after nonexecution amendments
is written separately. Exact reconstruction of an intermediate version file is
labelled post hoc and does not restore an original execution record.

The semantic checkpoint is resolved from the pinned asset snapshot. Its embedding
metrics are learned proxies, not human semantic ground truth. Foot sliding uses
collision geometry and simulator forces as a proxy; episode termination is not
automatically classified as a physical fall.

Historical one-off script `scripts/physics_guided_provenance_correction.py` is
retained only as a record of an already executed correction. **Do not rerun it**:
it is not part of the documented queue and would overwrite journal history.

Licensed original/derived motion arrays, model weights and recordings remain on
the server and in the local report bundle. GitHub stores code, configuration,
numerical records, hashes and the qualified experiment report.
