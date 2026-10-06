# Execution feedback: Isaac Lab continuation stage (2026-10-06)

This is a new stage following `reports/21_feedback_world_model_feasibility_20261006.md`. Preserve all previous negative results. The user requests execution-feedback causal validation on Isaac Lab, followed by zero-shot MuJoCo transfer. Extra post-retarget robot IK remains excluded.

## Stage 0: existing generated-output retarget witness

Purpose: close the missing original PyRoki/refinement interface for the nine existing nominal-feedback samples, before attempting physical continuation. This stage cannot establish disturbance recovery or online control. It runs frozen generation outputs and original upstream retarget scripts; no PPO update.

Prerequisites: existing old runtime `/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-retarget/bin/python` and its Python3.10 overlay `envs/dreamcontrol-retarget`, with existing `pyroki/src` and `jaxls/src`. Warm reuse; do not create environments or install packages. GPU0 is reserved for this bounded job. New repository is `/data/code/ouyanghao/oyh/conditional-human-motion-generation`; all outputs stay there.

Fresh agent witness: read this document and the new wrapper, then execute exactly once:

```bash
bash scripts/r3_feedback_v3_original_retarget_isaac_20261006.sh predicted
```

Working directory: new repository. The invocation creates a new isolated `TrajGen/sample_feedback_v3_isaac_20261006_predicted` sibling, copies the pinned official `retarget.py`, `refine_motions.py`, and helpers unchanged, links the existing three generated motions, then performs original retarget and official pause/interpolation refinement. Original relative URDF paths stay valid because directory depth is unchanged. Never run an official collection script that removes sample directories.

Expected evidence: process exit0 and sentinel `FEEDBACK_ORIGINAL_RETARGET_ISAAC_READY`; monitoring `r3_feedback_v3_original_retarget_isaac_20261006_predicted.state/.log`; `Kick_sim1/*.pkl` and `Kick_sim2/*_n.pkl`. Report the accepted count out of3, including rejected inputs. A COMPLETE pipeline with fewer than3 accepted inputs does not pass the future F retarget gate. No native runtime rebuild, policy update, new IK, repeated rerun, or output overwrite is authorized by this witness. Timeout is bounded to1800s retarget and600s refinement. If it fails, preserve logs and return the exact failure.

After this witness the root may execute the same frozen wrapper for `actual` and `shuffled`, sequentially, once each. It must inspect each result. The actual Isaac continuation entry requires a separate documented witness before a physical readiness claim.

## Reviewer-led continuation protocol

1. Existing refs46–48 / one diffusion seed are only an execution-chain smoke; their hints came from nominal MuJoCo execution, not post-disturbance Isaac feedback.
2. New exploratory post-push contexts must come from actual Isaac state up to the branch only. Original future goal, episode clock, reference phase, task and frozen policy remain fixed.
3. Arms: F0 fixed reference, F1 planned-context regeneration, F2 simple root/yaw alignment, F3 actual sparse root/ankle residual, F4 same-task cyclic/time-shift residual. Matched text, future hint, seed, DDPM budget and common physical prefix. Failures remain in denominators.
4. Generation takes about100s per existing batch. The initial causal test is **offline synchronized pause-and-replan**. Physics time pauses equally for regeneration arms; it is not realtime control. Record walltime separately. Do not reset robot state or teleport it to a new reference at the branch.
5. Future-only reference switching uses an identical fixed blending window. Freeze external world goal and event time from the original reference. Primary scores are against that original goal and original tracking reference; changed-reference fit is secondary. Kick has a virtual target: no entity-contact claims.
6. PhysX exposed root/joint/last-action/state matching is not a full contact-solver clone. Validate identical duplicate prefixes and report numerical differences. Disable automatic reset for single-episode scoring; count first fall and penalize the unexecuted interval.
7. True confirmation needs new, locked paired human/G1 reference IDs not previously used for bridge fitting or staged tests. Existing0–65 have been exposed; no independent generalization claim from this smoke.

Confirmatory gate:100% retarget/finite/joint-limit pass; F3 improves recovery AUC by≥10% versus both F1 and F2 with reference-cluster paired bootstrap CI lower bounds>0; better than F4; success loss≤5pp, no fall-rate increase, clean tracking noninferiority5%; switch spikes P95 not worse than F1 by>10%. Three generation and disturbance seeds. Offline success does not pass the separate generation-deadline gate.

## Stage 1: actual Isaac post-push capture witness

Fresh independent agent: read the new `scripts/r3_feedback_isaac_continuation_20261006.py` and shell wrapper, then from the NEW repository invoke exactly once, **only after the original-retarget GPU0 jobs have exited**:

```bash
bash scripts/r3_feedback_isaac_continuation_20261006.sh capture
```

This warm reuses the existing standalone Isaac Sim4.5 and `envs/dreamcontrol-r3/site-packages` overlay declared in `configs/env-dreamcontrol-r3.json`. Do not rebuild or install. GPU0, headless physics, no PPO updates,500 control steps /10 simulated seconds. Eighteen robots comprise six exploratory references49–54 × clean/push/push_duplicate. Observation noise is off. Original TorchScript policy/normalizer is unchanged. Apply world y force80N to torso at control steps50–59, transforming it to the API's local body frame every control step. All reference paths remain original in capture mode. Disable automatic terminations for this single-episode diagnostic and record first fall; this does not change any formal baseline run. A default initial reference pose is set only before the episode, using the official .15m initial height offset. No subsequent state reset, reference switch, IK or training.

Expected: normal exit0 and `ISAAC_FEEDBACK_DOCUMENTED_WITNESS_COMPLETE mode=capture`; state/log prefix `artifacts/monitoring/r3_feedback_isaac_continuation_20261006_capture`; `artifacts/feedback_isaac_20261006/capture/{physical_rollout.npz,summary.json}`. Inspect actual500-step sequence, finite626D observations/29D actions, force timing, original-library identity and duplicate trajectory difference. Sentinel validates the exposed-state interface, not a feedback improvement or a full PhysX solver clone. If failure, preserve logs/output, return exact failure, no autonomous dependency fixes/rerun. No video is requested by this invocation and it cannot satisfy the formal RL local-view Gate.

After capture witness, the root may invoke the identical wrapper with `smoke` once after all three retarget arms have completed. Smoke has fifteen environments: refs46–48 × fixed/fixed_duplicate/planned-regenerated/actual-regenerated/shuffled-regenerated. Before the fixed branch all observations use original references, including lookahead; only reference fields switch after the branch. Same .2s blend for all regenerated arms, no state reset or hidden-clock change, original task goal/offset/event retained. The smoke feedback was previously nominal MuJoCo-derived, not current Isaac post-push feedback. It remains a nonconfirmatory pipeline diagnostic even if scores improve.

### Stage 1 actual finding and Stage 2 revised smoke witness

Capture completed once with exit0,500×18 finite state/action sequences. The duplicate exposed physical trajectories diverged: all-run max q difference0.37763rad; through branch step78 max0.15148rad; pre-push max0.02679rad. This fails any claim that the duplicated environments are an exact common-prefix clone. Original capture output and its frozen source/hash witness stay preserved. The original capture generic rows incorrectly named every row `regenerated_physical_trial_executed=true`; no generation or switching occurred in capture and the independent witness explicitly rejects that interpretation.

Retarget completed once per arm: planned3/3, actual2/3, wrong2/3. Original local source index1 (reference47/human76) is rejected in actual and wrong. The current actual feedback retarget gate therefore FAILS; do not expand this version into confirmatory F or new formal PPO.

The revised smoke-only source, executed once after the capture completed, adds full exposed prefix differences (root/joints/velocities/actions), equal duration/frame/time assertions, accepted-file identity manifest, hard limit audit, active-reference tracking and original-world target error. Candidate translation is `raw_generated_position + original_offset`: subtracting the candidate automatic goal placement does **not** erase its raw feedback displacement; it avoids relocating the external task. Verify candidate foot error against the frozen original goal separately. Retarget-missing rows run original fallback solely for batch safety: mark them ineligible, null their physical benefit metrics, count them as failure, and never average fallback benefit into F. Nominal smoke AUC is named post-branch tracking, not recovery. Full PhysX clone and feedback causal gate remain false.

Fresh independent Stage2 witness: read the revised source/document and invoke once from NEW repository:

```bash
bash scripts/r3_feedback_isaac_continuation_20261006.sh smoke
```

Expected exit0 and `ISAAC_FEEDBACK_DOCUMENTED_WITNESS_COMPLETE mode=smoke`, output `artifacts/feedback_isaac_20261006/smoke/{physical_rollout.npz,summary.json}`,500 steps/15 environments/zero PPO updates. This is only a downstream accepted-sample diagnostic. A finite runtime witness is not evidence the feedback helps. Check missing source index1 in both feedback arms remains failed and all original target/event fields remain unchanged. Do not rerun capture, change source during execution, rebuild env, enable training or retry on failure. Save a separate narrow witness. The delta environment resource spec is unchanged except this invocation's smoke mode and15 environments; no dependencies change.

## Formal RL boundary

The initial work is frozen-policy inference and interface diagnostics. A new formal PPO method run additionally requires locally confirmed continuously updating actual Isaac Lab simulation and a saved gate. Monitor and training remain separate processes. Previous viewer for Open-Door or a checkpoint replay cannot certify this new Kick experiment.


### Stage2 actual result (do not rerun)

Normal exit0,500×15, summary and arrays independently verified. Original fixed goal hits2/3, planned2/3, actual1/3, shuffled0/3; shuffledref48 first fallstep105. Actual/shuffled missingref47 remains excluded from benefit and failed in attempted denominator3. All generated candidate audits finite/hardlimits pass, but retarget2/3 fails required3/3. Prefixroot0.01494m/joint0.18532rad differs beforebranch; no exactpair causal inference. This is nominal non-push execution-chain diagnostic only, not a recovery or formalPPO result. See report23 and continuation_summary.json.
