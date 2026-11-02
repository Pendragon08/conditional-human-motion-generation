# Revision4 executor: bounded validation prepared, actual SDK validation pending

This is a new executor revision, with the same post-exit evidence schemav3.
Earlier9-file candidates and all16 frozen active-training files are untouched.
Current formal training retains the common SDK lock. **Do not run this executor
while training is active.** No SDK run has been performed with revision4.

Files: contact_replay_modes.py, evaluate_contact_filter_v4.py,
supervise_contact_filter_v4.py and configs/contact-filter-v4-spec.json.
Exact five-file bytes, including the new test, are retained in
results/contact_replay_v4_20261026/candidate_runtime_bytes_v4.json.

## Modes and current disabled state

- Both modes require the genuine finished40000-round/800000-update teacher,
  closed writers, actual teacher supervisor exit0 and the private full checkpoint.
  The existing1000-round checkpoint and a public inference export are insufficient.
- validation selects8 complete original trajectories with50..500frames by
  increasing length and stable full-reference index. It ignores their failure
  rate for selection, keeps the original arrays intact and replays them as one
  batch of8. No artificial prefix shortening or forced completion fixture exists.
- Validation has1024 vectorized control calls and600 supervisor seconds as its
  run budget, with separate bounded shutdown/resource-query time on failure.
  If a bound/failure is reached, existing private trials remain; no formal
  completion or acceptance artifact is emitted. Actual control calls and physics
  substeps are recorded separately. Paper strict thresholds remain0.25m/0.5m.
- Formal mode uses all8278 original references, the genuine relaxed failure
  statistics and strict whole-motion coverage of the eligible references. It can
  export evidence inputs only after completion. Actual exit/resource/process
  artifacts are still required by the separate v3 finalizer before any decision.
- Both modes apply the exact validated solid-floor profile,557D privileged
  observations, frozen paper configuration, GPU1/Torch0.65/CPU0–3 and no optimizer
  steps. The SDK's possible auxiliaryGPU0 allocation and native-close fallback
  remain explicit differences.

Three flags are currentlyfalse: validation_launch_enabled,
formal_evaluation_enabled and real_sdk_validation_completed. Current flags refuse
dispatch before opening supplied teacher files or invoking the SDK. Enabling only
the validation flag cannot authorize formal filtering. Validation status is
TEXTOP_CONTACT_REPLAY_VALIDATION_COMPLETED, distinct from formal export status;
it never produces evidence_inputs.private.json, corpus decisions or accepted data.

## Actual CPU preparation evidence

Local and server each passed13 synthetic mode/selection/acceptance checks.
These include incomplete/failed/unclosed teacher refusal, mode flag isolation,
deterministic whole-source selection without mutation, insufficient-source refusal
and rejection of a validation receipt by the actual v3 evidence validator.
Synthetic finished counters are explicitly test fixtures, not trained models.

Server invoked both disabled supervisor modes and verified exit before any SDK
dispatch or output run. With only an in-memory validation flag for a CPU check,
the actual1500-round running receipt was still rejected as incomplete. No actual
configuration flag was enabled. The common SDK lock was observed busy.

Manifest-only planning verified the8278 reference order fingerprint. Eight full
short sources are available; selected lengths range52..64frames. This only proves
that bounded validation inputs can be selected, not that robots track them. The
exact indices/hashes stay in the private server validation_selection_plan file;
public evidence contains counts, range, whole-library fingerprint and private-plan
digest. No motion arrays were reread or retargeted.16 active+9 older candidate
files matched their existing hashes before and after this new-file staging.

## Future actual validation, after the current training exits

1. Confirm the actual completed40000 teacher receipt, closed writers, original
   supervisor exit0 and private checkpoint SHA; back up that checkpoint. Recheck
   live GPU/RAM/disk, SDK workers and mutual exclusion. Do not stop current jobs.
2. Set only validation_launch_enabled=true in the new candidate spec after this
   review, preserving numerical settings/source-byte bindings. Leave the two
   formal-acceptance flags false. This activation has NOT been done.
3. Run one author witness and then one fresh independent document witness, using
   distinct never-existing run names. Example for the author, after prerequisites:

```
python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_contact_filter_v4.py --mode validation --run contact_v4_author_v1 --checkpoint /data/code/ouyanghao/oyh/textop-reproduction-20261024/tracker_validation/filter_policy_paper69_solid_floor_20261026_v1/completed_40000.pt --training-receipt /data/code/ouyanghao/oyh/textop-reproduction-20261024/tracker_validation/filter_policy_paper69_solid_floor_20261026_v1/receipt.json --training-exit-receipt /data/code/ouyanghao/oyh/textop-reproduction-20261024/receipts/filter_policy_filter_policy_paper69_solid_floor_20261026_v1_exit.json
```

   The fresh witness must follow the same source/spec/teacher bytes, with only
   runname contact_v4_document_v1 changed. No retries into existing directories,
   fake counters, parameter repairs or stopping other jobs are part of a witness.
4. Inspect actual child exit0/no timeout, supervision/resource success, closed
   validation receipt,8 recorded whole-source trials, zero optimizer steps and
   false physical_filter_executed/dataset_accepted. Require actual natural-end
   coverage to be exercised (complete_trials>=1), matching journal/log hashes and
   the same teacher/profile/reference order. Retain failed whole-source trials,
   actual SDK resources/native-close flags and private logs. If all sources fail
   or a bound/error triggers, retain evidence and resolve it; do not mark real SDK
   validation complete. A concrete post-exit witness verifier/activation receipt
   still needs to bind these actual outputs before formal enablement.
5. Only after both actual witnesses pass, bind their receipt/exit/source hashes
   and independently review formal resource/acceptance readiness. Then enable
   full formal evaluation with fresh runname; do not reuse diagnostic evidence as
   all-motion coverage. Run existing finalize_contact_filter.py only on the
   completed formal output after its actual supervised exit, not on validation.

This route prepares execution validation on8 short sources. It does not measure
whole-library runtime, final tracking quality, physical acceptance, VAE/LDM
training or generation–execution closure. The current no-duplicate-SDK constraint
takes precedence; author/document actual witness runs remain pending.
