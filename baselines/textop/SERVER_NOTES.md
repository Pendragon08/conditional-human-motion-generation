# TextOp environment ledger

### env: textop@92b1d1b1
how: isolated venv on nlp03, shared base read only.
tier: CPUs 4, GPU1 A6000, CUDA allocation cap 65%.
status: spec written BEFORE environment build; validation pending.
source: TeleHuman/TextOp ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22
assets: Yochish/TextOp-Data bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9

### env: textop@c5f7e5fa
Spec corrected: isaac_utils is at upstream/deps/isaac_utils, not under TextOpRobotMDAR. Original failed build log retained. Rebuild pending.

### env: textop@9d7f370d
Build metadata fix: pin setuptools68.2.2/wheel0.42.0 before editable packages. Older base setuptools incorrectly labeled RobotMDAR UNKNOWN. Isolated overlay only; base not modified. Validation pending.

validated: spec9d7f370d seeded official MVAE CUDA forward/backward passed; fresh independent doc witness_v2 passed verbatim SSH invocation, exit0, no divergence. No trained result claimed by environment witness.

validated: spec9d7f370d fresh independent doc witness_v3 passed verbatim SSH invocation after corrected training-observability documentation, exit0, no divergence. Shared environment unchanged; attempt1 preserved, attempt2 uses official skip behavior and separate counters.

### env: textop-data-prep@87f577cc
Warm reuse packages from gmr_amass_20261021 (old dependency spec141547bf), no package/environment mutation. New spec records licensed body-model availability and paper69D target. Source and model hash witness pending. Resources CPU4, no CUDA. MVAE57D is SIGSTOP-held pending data preparation.

Validation update 2026-10-25: author and fresh document-following executions both exited0 with TEXTOP_GMR_BODY_WITNESS_PASSED, SMPL-X num_betas16 and finite GMR qpos[36]. Results/data_preparation_20261025 retains both witnesses and explicit canonical-spec hash87f577cc versus file-byte SHA2566e7f21352c2839d62498a068103cfedcd3949f5416b1ca075477f6bdbef94591. The hash formats represent different serialized inputs and are not contradictory. Six complete-motion processing checks passed; initial errors remain recorded. This certifies CPU preprocessing only, not dynamic tracking quality or full paper dataset readiness.

Data scheduling: server batch500 PID3257584; CPU-only, four threads. Continuation wrapper PID3266979 waits for it to finish successfully, then processes143 additionally matched ACCAD inputs; no overlapping retarget batches, RL launch, or generator resumption. Local archive extraction PID69024 selects1,653 indexed BMLmovi/EyesJapan inputs from already licensed SHA-verified D-drive archives into C-drive private storage with6GiB reserve. No source archive or dataset deletion is performed. Do not put these private data directories in Git.

### env: textop-paper69@934d8221
Spec written before numerical validation. Warm reuse textop9d7f370d dependencies, no packages installed. Process-local69D codec/FK/rollout adapter, fixed upstream preserved. GPU1 allocator cap65%, CPU4, batch2 numerical witness only. Validation pending; do not launch training.

Validated textop-paper69@934d8221: seeded author and fresh verbatim document runs both exit0, identical source hashes, TEXTOP_PAPER69_NUMERICAL_WITNESS_PASSED. Core codec/FK/VAE/LDM/CLIP/sampling only; batch2 peak701244928 bytes; optimizer_updates0, dataset_acceptedfalse. Full corpus integration and training remain pending. Failed int32 sampling traces preserved.

### env: textop-tracker@e65409d5
Spec written before dispatch. Warm read-only IsaacSim4.5.0/IsaacLab2.1.0 reuse, pinned TextOp source injected in task PYTHONPATH. Shared assets and packages unchanged; GPU1, CPU4, Torch65% cap. Headless explicitly authorized. Package inventory and actual physics/FK witness pending; do not launch filter training yet.

### env: textop-tracker@893388d9
Inventory confirmed Python3.10.15, Torch2.5.1+cu118, NumPy1.26.0, Gymnasium0.29.1, RSL-RL2.3.1. IsaacLab projectVERSION2.1.0 and distribution0.36.21 refer to different release identifiers. Shared packages unchanged. Spec includes resolved versions before numerical physics dispatch. Author and fresh document witness pending.

### env: textop-tracker@9beeb746
Backend numerical checks passed twice under893388d9, but normal teardown never returned. Both zero-update test children terminated after evidence preservation; GMR and old57D pause state retained. No shared package/source changes. This diagnostic has no Replicator writers, so use documented SimulationApp.close(wait_for_replicator=False). New240s supervisor retains real exit status. Fresh author/document rerun pending.

### env: textop-tracker@b7a3bf0c
Native close remains blocked even without Replicator wait. V1/V2 failures retained. Diagnostic only: disable capture-on-play, use explicit FK forward, stop physics and clear callbacks, attempt native close with20s bound. A fallback process exit is permitted only after complete zero-update numerical receipts; record fallback distinctly from native close success. No camera/writer or uncommitted experiment state. Shared dependencies unchanged; author/document verification pending.

Validation update2026-10-25: backend author_v5 and independent document_v2
both actualexit0, timeoutfalse, numerical pass, matching source/spec/URDF hashes.
The20s fallback was used; stack blocks in SimulationContext.stop/callback/render
before proving native app.close was reached. All native hangs/timeouts and the
optional Replicator import error retained. Full MDP environment_probe_v1 also
passed8env/24controlcalls/557D observations and strict/relaxed config assertions.

Warm diagnostic extension: configs/tracker-ppo-diagnostic.json written before
dispatch; dependencies unchanged. export_author_v1 passed full2 motions+mirrors,
350FK frames. PPO author_v2 passed40 actual Adam updates and exact-value/type/
shape model/Adam/actor+critic normalizer restoration with continued optimization.
Independent ppo_document_v1 direct invocation also exited0, timeoutfalse;
document comparison saved separately. Native shutdown fallback only after
checkpoints/receipts saved and writers closed. This remains8env/2iterations,
not8192env/40k physical-filter training. Author_v1 failed after20 actual steps
on Adam scalar CPU/GPU comparison; stale0 receipt retained with correction.
Full corpus and8192 capacity still pending; do not enable the generator gate.

Storage update2026-10-25: one unusedEngineAI Python environment
/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-engineai-official archived
in a full fileSHA-verified, six-part GitHub Release, then removed with old fixed
caches. Its source trees/base interpreter retained; recover and revalidate
before using that old profile. Current TextOp9d7f370d/GMR87f577cc/paper69934d8221/
Trackerb7a3bf0c environments and shared SDKs unchanged. HOME TextOp assets and
PhyGile preserved assets relocated to personal/data with all19 fileSHA matches
and original paths retained as links. Details in personal_storage_cleanup report.

### capacity diagnostic: textop-tracker@b7a3bf0c / shape@37c64682
Declarative configs/tracker-capacity-spec.json written before dispatch. Warm unchanged dependencies, GPU1/CPU4/Torch65%. Completed643-source library(1286 trajectories,946700 frames),8192envs, one24-step PPO iteration/20 Adam updates only. Verify actual observations, finite gradients/checkpoint and measured memory. Full-corpus capacity,40k formal filtering and generator readiness remain false. Author followed by fresh verbatim document witness required before marking this shape validated.

Validation update2026-10-25: capacity_author_v1 and independent
capacity_document_v1 both exit0/196608transitions/20actualAdamupdates, matching
source/spec and peakTorch10292943872bytes. Initial random policy produced huge
finite losses/contact spikes; these runs certify capacity, not stability or
tracking quality.1286trajectory reference-PD initialization audit author and
document runs also exit0; roots agree within1.1921e-7m,466initial poses clipped
at soft joint limits.20PD steps had contact spikes and37termination events,
not physical acceptance. Native-close fallback remains disclosed.

### recorded filter-policy runtime: warm tracker@b7a3bf0c
configs/filter-training-plan.json written before dispatch. Official PPO/sampler
unchanged; independent raw attempt recording and atomic checkpoint metadata
added. Formal launch remains disabled pending full library/readiness. First
8env diagnostic failed after20updates restoring an upstream inference-mode
EMA tensor; failure/checkpoint retained. Restore now runs in inference mode.
Author_v2 passed40actualupdates and exact model/Adam/normalizer/EMA/raw-count
restoration; fresh FILTER_TRAINING_ENVIRONMENT.md execution still required.
No40k training or filtering or full generator launch has occurred.
