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

Latest recorded-runtime validation: author_v4 and independentdocument_v2 both
actualexit0/no timeout/40Adamupdates, source3929b9d8..., runtimead3bb92f...,
plan10c9ef10.... The restart audit now explicitly resets physics, restores EMA
and raw counts without counting the synthetic reset, and marks new episodes
as real attempts.9completedattempts observed. Model/Adam/normalizers/EMA/counters
match exactly, then optimization continues. Earlierauthor_v3 failed20updates
on inference-mode metric reset (exit1, preserved); reset is scoped to inference
mode, optimizer loading remains outside. Independent REPORT copied by collector
when available. Only this latest source has the resumed-attempt fix validated.
Formal training remains disabled pending complete inputs/full-library checks.

### strict-replay executor diagnostic: warm tracker@b7a3bf0c
configs/strict-replay-spec.json declared before dispatch. No package/shared
source changes, GPU1/CPU4/Torch65%. New evaluator loads the explicitly hashed
two-iteration diagnostic checkpoint and2original references, privileged557D
inference only, no optimizer steps. Deterministic start0, reset perturbations
disabled for strict replay; standard height/orientation terms restored.
Default10stimeout replaced with exactT-1transition motion-end timeout. All
per-motion logs stay private; diagnostic cannot emit accepted evidence. Actual
SDK author and independent documented run required. Formal evaluation disabled
until real40kpolicy and full reference corpus exist. Pure17case coverage/schema
tests alone are not SDK validation or tracking success.

Strict author_v1 exited0 but first-step failures were contaminated by stale/
zero aligned-reference buffers. Preserved sources/strict_replay_author_v1 and
raw receipts/logs. Author_v2+independentdocument_v1 refreshed the pinned exact
alignment transform after reset, no time/EMA/physical-state change:16control
calls,64scene steps,2failed originals,0completions; initial body error.02697m.
Author_v3 adds a distinct2frame prefix control-flow fixture only, not original
motion acceptance. It passed17calls/68steps,2failed originals plus2complete
prefix fixtures,0optimizer steps,actualexit0. Latest source5c8e0bbe..., spec
e5f95521.... Independent STRICT_REPLAY_ENVIRONMENT.md v2 run pending. No full
teacher or real corpus filtering completion claimed. Nativefallback remains.

Latest validation: strict_replay_author_v3 and independentdocument_v2 both
actualexit0/no timeout, source5c8e0bbe.../spece5f95521...,17calls/68scene steps,
0updates,2failed fulloriginals/0fullcompletions. Separate2framefixtures verify
SDK motion-end handler, full_original_motion_passes0, originals unchanged.
Independent44checks passed. This validates replay/accounting paths, not full
motion success or actual40kpolicy filtering. Newest source validated; all
earlier records and historical sources retained.

Reference-only queue: configs/reference-export-queue.json written before
launch, warm unchanged export kernel. Server workerPID3527552, state receipts/
reference_export_queue_v1.json, flock-locked. Waits each exact full GMR cohort,
RAM12GiB/data2GiB/GPU1free20GiB and no own headless SDK before serial exports
1330/BMLrub2139/KIT2513. No retarget, backup duplication, training or deletion.

### candidate loader diagnostic: warm paper69@934d8221
Spec configs/candidate-loader-spec.json written before dispatch, no package/shared-source mutation. All2296completed unfiltered originals+mirrors, private candidate view, train-only float64 stats/std floor1e-6, frozen real CLIP caches. Official dataloader batch512/history2/future8/primitives4, uniform train/val and weightedtrain. Process-local sampler fixes valid-index mapping and inclusive segment bounds. No model gradients/optimizer/accepted data or full training. Author build/kernel and fresh documented loader witness required. Final accepted data must refit statistics/cache after real physical filtering.

Candidate author_v1 failed safely after reading all2296sources on exactcross-
split pose check. Private audit2groups/1train+1valsource/275frames; raw numeric
differences≤9.6e-13,not retargetcollapse. Historicalsource/spec/exit1 retained.
Specv2 before dispatch now quarantines bothTRAINvariants of that source family,
keepsvalidation/allassets, refits trainstats with cross-source boundaries
excluded. Author_v2 actualexit0,2296examined/4590variants,train3440/val1150,
2999realCLIPtexts,3officialbatch512passes,0overlaperror/0missingembeddings,
0optimizerupdates. Source5c6d9a49.../spec703321e4.../samplercf5c1e3f....
This is an unfiltered candidate diagnostic, not accepted data/full model memory
or full8278corpus validation. Fresh documented candidate_loader_document_v1
verification pending. Warm dependencies unchanged; no original data deleted.

Latest candidate-loader validation: author_v2 and independentdocument_v1 both
actualexit0/no timeout;41independent checks, source5c6d9a49.../spec703321e4...,
2296examined/4590variants,train3440/val1150,three[4,512,10,69]loaderpasses,
0overlaperror/0missingtext,exact35frame+validweightedmapchecks. Candidate-only,
0optimizerupdates, not fullmodelmemory or accepted corpus. IndependentpeakGPU0
is expected for cached CPU loading; authorCLIPpeak442733568bytes. Six private
audit/quarantine/feature-statistics/text-cache records backed up to localD with
allSHA matches (12636826bytes); no raw dataset/model/checkpoint deleted or
published. Final corpus must recompute quality exclusions/statistics after real
physical filtering. No package/shared environment changes.

### neural batch512 diagnostic: warm paper69@934d8221
configs/neural-batch512-spec.json declared before dispatch; unchanged packages
and basecodec/sampler. Fixedunfiltered2296view/metaSHA, official VAE9L/LDM8L,
batch512/history2/future8,4primitives with sequential backwards and detached
history, actual cached CLIP and train-only candidate statistics. Official
MVAEManager/DARManager loss bodies reused. No optimizer constructed, no updates
or model learning; Adam/EMA state memory NOT allocated. EightVAE/fourLDM
backwards and5stepCFG5numeric sampling test capacity/stability only. Parameter
hashes unchanged required. Candidate data/randomVAE cannot establish accepted
training readiness, semantic quality or full-state training memory. GPU1/CPU4/
Torch65%; author followed by fresh documented witness required. Do not interrupt
GMR/FK or oldpaused57D. No shared sources/package install or cleanup.

Validated neural batch512 author_v1 and independent document_v1 both actualexit0,
no timeout,42 independent checks, identical source26cbf30c.../spec2a91a7f6...
and candidate viewc4875fa6.... EightVAE/fourLDM backwards have finite nonzero
gradients and unchanged parameters. Two5stepCFG5 numeric segments pass.
Torchallocated peaksVAE3429537280/LDM1681969664bytes. No optimizer/Adam/EMA
states allocated or updates. This validates the zero-update shape only; full
training memory, accepted full corpus, actual learning and quality remain pending.
Documentation NEURAL_BATCH512_ENVIRONMENT.md and report/neural_batch512 evidence.

Live preprocessing profile2026-10-25 18:55 China: CMU process3483711 used
0.999 CPUcores during5s sample despite affinity[0,1,2,3],13threads, lifetime
CPU121%. Completed1235/1330 with0failure,1379092 outputposeframes including
mirrors,3h47elapsed. Main bottleneck is per-frame serial GMR IK, not GPU shortage.
Default NumPy/MuJoCo/Mink+DAQP solve path has no CUDAdevice switch; SMPL-X
Torch body forward could be separately accelerated but speedup is unmeasured.
Do not equate GPU FK/filter use with GPU GMR IK. Existing workers unchanged.

### Persistent preprocessing completion audit2026-10-25 19:40
Read-only local bundled Python/NumPy2.3.5, no ML/GPU environment change. Spec
configs/preprocessing-completion-audit.json beforedispatch, source
a4a9418aa2ee1e6730a4f113f2eecb1955081efe1b5f467147ab99a3d69e076e,
spec1dd2ba137f99bd173b8158dda2d1b8db27ed48ac2bdbd3be0a72e571e90cfa25.
Author_v1 actualexit0/17.5s reread all2296NPZoriginal+mirror payloads from3
persistent private archives; per-motionSHA/source-split/dimensions/times/
finitequaternion/contact/selected69Dfields/mirrorroot/annotationbounds passed.
Selectedfieldmaxerror1.1921e-7. This does NOT recompute everyFK/codec field.
Correct result INCOMPLETE,5982available sources still lack completebackup
verification; do not equate livepartialGMR progress with durablefullcompletion.
No data deletion/copy/export, no worker control or physical acceptance. Fresh
PREPROCESSING_BACKUP_AUDIT.md witness requested, independent result pending.
All6backups+exact8278source/split set required for preprocessor completion
notification; real physical-filter and fulltraining gates unchanged.

Latest backup-audit validation: author_v1 and freshdocument_v1 actualexit0,
22independentchecks/source+spechashes match, no divergence. Both2296verified
from3persistentbackups/5982pending, INCOMPLETE preserved. No physicalfilter/
accepteddata/generatortraining. Independent REPORT collected with aggregate
results. This validates current audit procedure; all6cohorts notyetcomplete.

### Generator full-state diagnostic declared2026-10-25 21:45
Warm paper69@934d8221, no package/shared source changes. The previously draft
generator-state-witness spec now explicitly enables a bounded candidate-only
diagnostic, not formal training. GPU1/CPU4/Torch65%, fixed2296candidate view,
batch512/history2/future8/fourprimitives. Eight actual Adam VAE and eight AdamW
LDM updates, checkpoint at4 per stage; optional EMA enabled only as a diagnostic
state stress fixture (released formal defaults remain false). One validation
batch pending at the checkpoint must survive fresh manager restoration and
complete without optimizer updates. Verify model/Adam/EMA, primitive attempts
versus actual steps, explicit batch generators plus Python/NumPy/Torch/CUDA
RNG and next-batch identity. At most1.1GB additional private checkpoints per
run, at least2GiB free; author then fresh documented witness required. Status
UNVALIDATED until both actual exits/proofs. Existing GMR/backup/FK queues intact.

Validation2026-10-25 21:51: generator_state_author_v1 and freshdocument_v1 both
actualexit0/no timeout/16actualupdates; independent75checks passed, no divergence.
Both8VAEAdam+8LDMAdamW updates,zero skips, actualstates/EMA allocated; fresh
model/optimizer/EMA/RNG/generator/clock/pendingevaluation restoration andnextbatch
identity passed. Pending4validation primitives complete with0updates, thenresume
to8; wrong57Dbinding rejected. Eachrun checkpoints684683326+305124157bytes at
step4 independentlySHAverified; no final8step or formaltrainedcheckpoint claim.
AuthorTorchpeaks4109051904/1979146752bytes. Warmdependenciesunchanged, official
EMAdefaultsfalse, candidate/firststage scopeonly. Afterboth datafree2889515008B
(2.69GiB), above2GiBguard but budget futureformalcheckpoints beforelaunch.
Sourceaa511f44...,state46ef2ade...,specbd3f7a06.... Reportandprivatecheckpoints
preserved; no additionalduplicate diagnostic needed without a new concern.

### Parallel GMR declared2026-10-25, before dispatch
User authorized disjoint-motion parallel preprocessing. Warm GMR87f577cc, no packages changed. Four single-CPU workers, threads1, CPUset0-3, GPUdisabled; same motion-local sequential solver and69D math. Original90db2afd kernel preserved; AST shows only main cpu_threads argument/thread constant and entrypoint dispatch changed. New configs/gmr-parallel-spec.json precedes sample benchmark. Compare8 complete motions against serial4threads and stored previous outputs; contacts/times/labels exact, floating tolerances declared. Canonical output schema and private backup gate preserved. Existing BMLrub upload/KIT waiter continue; scoped WAIT_FOR_VALIDATION prevents unverified production dispatch. Author and fresh documented witness required before enabling gate. No changed algorithm, resource expansion or full physical acceptance claimed.

ParallelGMR author_v1 failed output equivalence atqpos maxdiff0.0001666463840418464, unchangedatol1e-5. serial101.41s/parallel39.40s are failed-profile timing only, not production acceptance. All source/spec/output failures retained. Revised spec beforeauthor_v2 keeps originalTorch/BLAS4threads and allworkers share fixedCPUset0-3; aggregate CPUcap stays4. No tolerance/algorithm change. spec78de590d...,helper9711f166...,witnessa0ae6fc5.... Awaitauthor+freshdocument equivalence beforeactivation.

ParallelGMR author_v2 passed unchanged comparison, serial100.41s/parallel92.83s (1.08x). OriginalTorch/BLAS4 arithmetic preserved, but concurrent tiny BLAS operations oversubscribed; this is limited benefit, not a4x claim. Revision3 declared beforedispatch retainsTorch/MKL4 and changesonlyOpenBLAS to1; same4CPUset and precision gates. Existingauthor_v1failure/v2sources retained. No production activation yet.

ParallelGMR author_v3 preserved exact numerical outputs but serial99.38s/parallel92.83s(1.07x). Revision4 declared beforedispatch changes only idleOpenMP wait behavior toPASSIVE/KMP_BLOCKTIME0, retainingTorch/MKL4 andOpenBLAS1 andsame4CPUset. No numeric tolerance change; previous source/receipts retained. Awaitactual benchmark before enabling.

ParallelGMR finalauthor_v4 and independentdocument_v1 both actualexit0,48 independentchecks pass,112array/144metadata comparisons allzeroerror. Speed2.4086/2.3600x on8motions. Specb797d143...,helper79d532ac...,entrypoint06539b1a... validated andactivated forBMLrub/KIT only. BMLrub stagedrawreverified2139, parent3653970/4workers live, first13processed0failUTC14:55. Oldcoordinators naturally exited on benchmark mutualexclusion guard; failures retained, no raw retransmission. NewWindows61460/38236 authoritative observers; seeCONTINUATION for backup8h and privatecontentaudit3626. Fullphysicalfilter andgenerationtraining remain false.
