# Dynamic-filter evidence protocol v2

This is evidence bookkeeping, not a simulated evaluator. No motion has passed
physical filtering merely because this code's synthetic regression tests pass.
The actual entire-motion SDK rollout executor still must be implemented and
validated against the fixed teacher and reference library.

`scripts/apply_tracking_filter.py` now rejects the earlier header-only schema.
No accepted dataset existed under that schema. It reads five SHA-verified
artifacts: trained policy checkpoint, actual training receipt, supervised exit,
relaxed statistics export, strict rollout log. Actual training receipt must be
formal/8192envs/40000completediterations/800000actualAdamupdates; its final
checkpoint hash and completed-count metadata must agree. A witness receipt or
failed supervisor cannot satisfy the protocol despite a40k header elsewhere.

Both logs bind the same checkpoint and reference-order fingerprint. Relaxed
statistics must include every original training reference, positive genuine
attempt count, finite nonnegative EMA counts and consistent p=f/(f+s). Raw
initial resets are excluded by the validated recording runtime. Motions with
p>.05 are rejected before strict evaluation; eligible ids must exactly match
the strict log, without omissions/duplicates. These checks detect internally
inconsistent evidence; hashes alone do not authenticate hand-written data or
prove that simulation occurred. Actual trusted executor provenance is required.

`scripts/strict_tracking_coverage.py` tracks one continuous trial from frame0.
Tframes require T-1control transitions. A trial can end by natural motion
completion or real termination; simultaneous termination and motion end is a
failure. External10s timeout, early natural-end label, still-running trial or
joining two reset episodes is never a strict pass. Each eligible motion needs
the declared number of entire trials. The strict executor must restore pelvis
height0.25m/end-effectorheight0.5m terms, retain other standard terms and record
first failure or whole-motion coverage. Repeat count/reset choices remain
explicit implementation choices where the paper does not fully specify them.

Protocol/schema regression tests: tests/test_filter_evidence.py,17 cases,
localPython and serverPython3.8.10 both exit0. These include synthetic positive
schema fixtures and invalid short-witness, changed-checkpoint, never-attempted,
missing-motion, wrong-fingerprint, incomplete-coverage and failed-supervisor
cases. Fixtures deliberately contain a NOT_A_TRAINED_POLICY marker and are
temporary; they never enter a dataset or acceptance artifact. Server receipt
and full test output are in results/filter_protocol_20261025. No physics, RL
updates, full generator training or accepted data were produced by these tests.
