<!--tmr100-user-current-start-->
CURRENT 2026-10-29T08:22:19.012365+08:00：最新人类命令覆盖旧500轮：“500轮训练要太久了，改为100轮吧，不要500轮了。”现只选TMR第100轮/101500 AdamW完整状态，后续直接冻结评估与Step3/4，不再等500或507500。原run名tmr69d_evaluator500_train_v1和运行spec/source仍为原来源，冻结不编辑、不重启、不压缩数值/常数LR/RNG。VAE/LDM仍各100000，旧教师停止。
新唯一CPU端点user_tmr100_control_v1已实际派发并ARMED：采样Unix1793233316.210879为72轮/73300AdamW，WAITING_FOR_REAL_TMR_EPOCH100_SAVED_AND_RELOADED/0生产信号。原supervisor416285/start1441791429与worker416293/start1441791443保持原作业；新observer431950/start1443022390、supervisor431945/start1443022380仅身份线索，核真实argv/starttime。active control specSHA2f0498031e7fa9d5630848320dfed944c66c2c36ef45fa6cacbfe1989a02bba8，watcher4856467bd730b1f826a44e96b84faecbde57185f5e1ece20038b1d66d7379a24。控制每秒等真正100轮atomic fsync/rename、model/AdamW/allRNG/sampler/nextbatch生产精确重载及pending=false，再独立全SHA/owner/no-symlink/fingerprint后一次pidfdSIGINT，0模型/真实checkpoint载荷反序列化/GPU；独立metadata锁，不占GPU锁。严禁rearm/第二observer/额外信号。CPU/1GiB/2GiBdata/15秒资源/7天边界不降。
作者v2和fresh user_tmr100_control_document_v2 CPUfixture均实际0/reaped/final，14+4负例/重复intent拒绝，只有各自sleep进程收到1次SIGINT，生产信号/真实checkpoint/model/GPU均0。fresh真实SUMMARY SHA1a380fbc6971654a416f8c0937a498a6f947e31964b996a2e78fad79ec91d38a。prepare本地换行签名拒绝与author_v1测试argv子串错误失败1均保留；无控制/训练信号后已核空，归档精确旧字节才修复新禁用夹具，不改原训练，旧失败不改成功，不重跑已完成夹具。
只有USER_TMR100_EPOCH_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED+控制自身process/sup/outer0+原无worker/模型锁释放才接受端点。原RuntimeError('Owned TMR evaluator interrupted with signal 2')及原训练非零退出保留，post100内存/emergency不选，不重标500轮成功；原100轮immutable state唯一选用。
新100轮CPU导出/独立复核/语义桥已另名部署禁用，详见TMR100_FROZEN_AND_SEMANTIC_ENVIRONMENT.md。12个新prefix计数/RNG/sampler负例远端CPU0、14实际桥渲染Python3.8通过，仅准备；实际100轮出口/导出/语义均尚未接受，不提前pending启用。旧500轮源码/spec全保留但不再用于本目标。请用provider_tools/tmr100_endpoint.py collect --evidence新短ASCII轻量查唯一控制；正式模型live仍原collect_tmr69d_evaluator_training.py。阶段不是停工点，完成后直接新100CPU/fresh/语义→Step3增强与新one-stagePPO。Step3/4新增持久空间仍待用户输入，不能用本次减少epoch假装解决空间。
<!--tmr100-user-current-end-->

<!--user-step34-current-start-->
CURRENT 2026-10-29T07:58:22.863501+08:00：用户新明确授权Step3生成式数据增强和Step4单阶段PPO：冻结已训练VAE/LDM各100000，随机组合TRAIN文本，目标5368条/31.48小时；新Actor431D/Critic557D、未来5帧参考、[2048,1024,512]ELU、50Hz/Isaac Lab/PPO，全新策略/Adam初始化，不采用DAgger/蒸馏、不恢复旧557D特权教师。当前唯一模型任务仍是原TMR500epoch作业，不能重启/额外信号/重叠模型或SDK。
实际metadata run user_augmentation5368_scope_v1已process/sup/outer0/closed、1.66秒/无worker/独立metadata锁释放，0模型/数组/GPU/SDK/新动作/PPO更新；20个新隔离计数/来源/授权负例实际远程CPU0，仅契约检查，不是数值见证。私有文本流17889511B/SHAac49c12afeedc8703cf2c3b8c8e19df4c25fc691c2220df9d927d32d3552cbb4已真正server+D全SHA/owner/fsync/rename/SCP0/reaped接受。5361原TRAIN family/4043文本/34068annotation-chunk抽样，0原VAL；5368条中2208条1055间隔、3160条1056间隔，共5666400个50Hz有效间隔/5671768个真实目标姿态。初始两past features不计入时长，终末姿态必须真实生成，尾padding不计时长。原TRAIN切为5551条<40秒引用，共2835665间隔，尚未构建混合RL数值库。
只部署禁用生成backend/配置和PPO工厂，完整数值生成/混合引用/PPO入口及恢复/独立SDK见证仍须完成，不能称ready。生成spec8fd992a5f098bedc15126f3fc119fc9353c2545b0e950cef9e32a4d68c49ce1c两授权false；PPO spececce7bd48585eb77b64538c8ef10bf1b152283075fa05b1a9304011a0290072c两授权false。当前实际新增强动作0条/0小时，新PPO更新0。
真实容量不足：DATA2390249472B、HOME3004858368B、D复制后3212955648B，仅当时采样。新增canonical含头上界884096064B，生成body30/joint29引用10185795584B，候选3×128MiB策略槽402653184B，再保留2GiB+128MiB日志，至少13754246208B且未计额外原引用/增长；HOME还要容纳原TMR两个201326592B槽+80MiB推理。已向用户请求其有权限至少20GB空闲服务器持久目录及至少5GB本地备份空闲或新备份目录，尚无新输入，不得擅自删旧资产/降低保留线/用SHM冒充持久容量。当前TMR继续健康监测，来源/spec冻结，空间输入只阻塞新大输出。新正面收据覆盖下文旧CURRENT。详见USER_STEP34_ENVIRONMENT.md和reports/user_step34_augmentation_and_ppo_20261029.md。
<!--user-step34-current-end-->

## CURRENT — teacher STOPPED; direct all8278 strict replay RUNNING

Latest user: "直接进行下一步物理筛选，不要再训练跟踪器了。" Teacher is
actually stopped: one original-process pidfd SIGINT after verified15000 backup,
child reaped0, outer1 retained (old40k not completed), common lock released.
No teacher restart/resume, no further stop/15kbackup/release/statistics repeats.
Frozen15000 checkpoint90799682B SHA66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88
and private decisions server+D verified; inference GitHub release digest verified.
Historical EMA:8278total/8278attempted/0<=5%candidates/8278above/0unattempted.
This is NOT8278 actual strict failures. User then asked to relax the gate;
all8278 originals are admitted directly. No numerical physical threshold change.
Paper difference disclosed:15k instead of40k and bypass historical EMA selection.

Current unique fullrun teacher15000_direct_full_v1, active spec SHA
91184fff75f8a0579a759abb418d4a0186d3e42a9a48c176e3d33facc768a1cd.
Author and fresh document smoke both actually8/8 same sources/63controls/
252physics/0updates/closed finite trace/SHA/exit0; do not rerun or count as16.
Source d19e7574ac9899b25cb611bc80b44bb27bb32e5338f1b86ef4414f2f87c7c25a,
protocol9996786f840e5272bf7a6808df243ec8ce551d091bd0bbac218b76369d4dba74,
trace3a84aec7d257c8f41f837cf855d0d934512285b3e84079f802d0b3c03d1086f2,
supervisor191cfb6554b886d90dc55375bb5686055aaf4bd6f59cbaf009a6744ded8c101c.
Do not edit these active sources/spec or22 preserved training bindings.
Driver112822/supervisor112823/bash112830/SDK112837 are historical identity clues;
verify real argv/starttime. Commonlock held by strict replay, no overlapping SDK.
16envs/GPU1/sharedCPU0-3/Torch.65/headless, GPU0aux context disclosed.
All8278 originals sorted by duration/index, one deterministic full trajectory
each; .25m pelvis/.5m ankle+wrist height/orientation rules unchanged.
Full budget1GiB raw36D poses+.5GiB metadata/logs+2GiB reserve; preflight3.5GiB.
Planned max313378controlcalls, safetycap2million/24h; actual end may be earlier.
Every50calls/batchend progress, every15s resources. Incomplete is not success.

Latest observation 2026-10-27T21:52:55.635654+08:00: finished400, passed376,
failed24, pending/inprogress7878; see current.json
for new observations. Raw receipt replayed_original_sources=8278 and
unattempted_original_sources=0 describe selected scope, not live completions.
Use8278-finished_trials for pending; never label untested motions failed.
No final corpus/generator acceptance yet. All raw/failed assets preserved.

Read TEACHER15000_DIRECT_STRICT_ENVIRONMENT.md, reports/teacher15000_direct_strict_20261027.md,
results/teacher15000_direct_strict_20261027/current.json and this run/progress.json.
Read-only collection: provider_tools/collect_teacher15000_strict.py --run
teacher15000_direct_full_v1 --evidence UNIQUE_IDENTIFIER; retain client outcomes.
Then report_teacher15000_direct.py and collect_filter_first_report.py aggregate
the observed evidence. Never relaunch after unknown/pending client outcome.
On actual completion verify all original exits, closed private trials/poses,
source fingerprint/counts, commonlockrelease; final aggregate result notification.
Then accepted nonempty REFERENCE+BABEL corpus, family split/exact leakage,
TRAIN69D stats/fullCLIP/load-gradient-restore/persistent capacity acceptance,
NEW filtered VAE200k→LDM300k automatically. Never resume prefilter8100 or train
empty/unfiltered corpus. If no strict passes, report evidence; do NOT restart
teacher without new human authorization. User has stopped teacher training.
Full LDM entry/semantic TMR/multiseed controls/WBC/augmentation/online/video
remain pending. Preserve current-target vs final-endpoint distinction,
nativeclosefalse/fallbacktrue and exact author collision asset unavailable.
This CURRENT overrides all historical running-teacher/15k-pending/20k instructions.

## Current checkpoint preparation activation2026-10-27

Future recovery CPU backup spec now enabled after freshv4 preparation pass:
6be686cf78d60498b88eedcd71d1a22bc7d9f117ae459cc2275a16a36f1d0e68. This is a control-metadata change only;
no package/runtime/training22source changes and no actual new checkpoint load.
Same live teacher receipt10500, nativeobserved10574, noexit.
V1-v3 review gaps and failed Windowsowner fixture retained. Final local client
journals use direct retained stdout/stderr files; every D write uses current-user
ownership/non-link guards. Four client cases and freshv4 pass; unreaped outcome
branch static only. Positive15k audit/backup/strict screening remain pending.
The environment guide records the exact disabled-spec review commands; current
activation is separate in results/filter_recovery_checkpoint_tools_20261027.

## Current 2026-10-27: filter-first teacher recovery and future checkpoint CPU tools

The pre-filter VAE is safely paused with verified8100 fullstate/private backup;
the only new production job is filter_policy_paper69_resume10000_filter_recovery_20261027_v1.
Read CONTINUATION/current receipt, not the historical VAE-running entries below.
No active22 teacher files or existing environments changed.

FILTER_RECOVERY_CHECKPOINT_ENVIRONMENT.md describes independent future15k..40k
backup/statistics tools. SystemPython3.8 metadata tests and existing env/bin/python
Torch1.13.1 trusted CPU serializer are warm reuse, no package installation.
Local/server metadata-only author fixtures47pass; freshv1 also47pass but blocked
positive audits on destination/journal static gaps. Corrected v2 has real server
filesystem10checks and local synthetic transport4cases passed; freshv2 pending.
V2fresh found residual D ownership/partial/client evidence gaps. V3now uses
native Windows owner/reparse checks (author_v4 ninechecks, originalfailed_v3
retained) and complete client journaling (four updated synthetic cases). Freshv3
completed with two residual findings (D metadata mirror, Windows timeout
reader streams); v4 uses guarded mirror plus direct retained stream files.
Updated four client fixtures passed; freshv4 pending. No positive real new-checkpoint
audit has run. Positive new-checkpoint audit/backup and full
strict physical screening remain unperformed. Never repeat old10k export/audit.

## Actual VAE200k launch2026-10-27
Warm934d8221 unchanged. New3source/spec frozen, actualRUNNING100/200k/100Adam/0skip, noexit. Newcorpusfresh69 andneuralfresh62pre/65postpassed, allwitnessagentsended. Spec ec16b1468543e7a5347595d94ecc915502bb775fe885f97318ff53454ebe3fdc; commonlock held. ReadcurrentCONTINUATION, neverlaunchanotherGPU/SDK orrepeat oldwitness. LDMfullsampleTrue pathwaypending, originalquickfalse preserved.

## Current generator8278 pre-filter runtime — 2026-10-27

Warm paper69@934d8221 reused unchanged, no packages/sharededits. User explicitly
selected8278preprocessed sources beforefullfilter. Separatecorpus author/fresh
actualexit0, fresh69checks pass, same3real512×4loaderchecks; source+exactpose
crosssplit0 after2TRAINfamily quarantine, originalassetsretained. See
GENERATOR8278_CORPUS_ENVIRONMENT.md. NewcorpusRAMcache recoverablefrom6SHA-bound
manifests/privatepreparedbackups; normCLIP/recoverymetadata privateD+server.
NewseparateVAE200k schedule/spec documented GENERATOR8278_VAE_ENVIRONMENT.md,
originalquickruntime unchanged. Actualauthorgradient witness underway, fresh
requirednextbeforeactivation. VAEpersistentbudget5452595200B checked; Dintermediate
backupcapacity limited, finalstagepriority. OriginalSDK/teacher10kpaused; no
44semanticsSDK or40kresume whilegeneratorworkownslock.

## Current 2026-10-27: 10k paired physical evaluation and warm CPU analysis

Resumed teacher is user-paused after verified10kbackup/handoff; original final
receipt11400/outer1 is retained, notanactiveworker or40kcompletion. See current
CONTINUATION and TEACHER10000_PILOT_ENVIRONMENT.md. Same b7a3bf0c trackerSDK,
no environment rebuild/packagechanges; new10k schema onlychangesprovenance.
Author/freshsmoke both8/8actual; fixed64actual46natural18fail closedexit0.
AllSDK/backupworkersended; use actual lock/resource checks for nextjob.

The paired64_v1 comparison has a separate documented CPU invocation in
TEACHER10000_COMPARISON_ENVIRONMENT.md, existing ROOT/env/bin/python+NumPy,
CUDAhidden/CPU0-3/singleBLASthread/noTorch. No environment or dependencies changed.
Spec10bf2854df741534a1c31b63fb54f33068b6e14dff00bdbd513b20bba694024d binds
existing endpointkernel; authoractualexit0. Freshdoc recomputation actualexit0
matchesall64 and remote summaryec1b4c9ae8767351208f97d87a07135918450b262d50a018a5581c8cdba64924.
It recomputesnew10kendpoints and reusesSHA-bound independentlyaudited5kmetrics,
notnewphysics. Originalexecution/source/specbytesnevermodified. Thisledgerentry
was addedafterfreshreview notedmissingcomparison-specific pointer; runtimeand
verbatim documentedinvocation unchanged. Countcomparison42->46 is notfullcorpus
orsemanticacceptance; mixedendpointquality remains. No need rerunwitnessforpointer.

## Current online and full continuation runtime (2026-10-26)
Read QUICK_ONLINE_ENVIRONMENT.md and FULL_TEACHER_RESUME_ENVIRONMENT.md first.
Onlinev11+freshdoc real257timedcalls each passed0deadline/underflow/physicalfailure.
Explicit CPUPhysX single-robot +GPU1neural, aggregate4CPUs/combinedTorch.65; this
runtime difference does not change full8192GPUphysics training. Both closed.
Full teacher continuation uses unchanged frozen SDK env/trainer/plan, separate
--resume launcher with verified5000 fullstate. Newrun and spec are frozen;
inspect results/full_teacher_resume_20261026/progress.json, never relaunch.
Quality review is negative/limited; functional engineering acceptance does not
mean semantic/paper quality passed. Final corpus/fullgenerator gates remain.

### Verified quick offline generation→physical execution→video2026-10-26
VAE2k/LDM3kfinished actualexit0/0skips, allprivatecheckpoints server+correctD
SHAverified. Newtrainedgeneration author/fresh169checks; newSDK2phasesauthor/
fresh416checks, natural257controls/1028substeps one258pose stream. Newmedia
freshactual258frames/50fps/CPUllvmpipe,0newphysics/SDK/optimizer, oldrendercore
unchanged. Allworkers ended/profilesimmutable, no newpackages/sharededits.
Raw/Dprivate evidence retained, onlyaggregatepublic. LDMsamplingHuberworsened,
semantic/realtime/fullpaper remainsnotaccepted. See currentCONTINUATION.

### generated SDK validation declared beforeactualdispatch2026-10-26
Warm trackerb7a3bf0c unchanged, no package/sharedsource changes. New generated
reference schema binds actual VAE2000/LDM3000 and two realneural witnesses,
one258pose stream. Separate SDK FK128slots then fresh SDK1env5kteacher strict
physics; originalURDF/solidfloor/thresholds/preautoreset recorder preserved.
No8278sourceidentity fabrication. Both actualexit proofs, coverage/failures,
resource sampling, nativeclose/fallback required; author/freshdoc pending.
Read QUICK_GENERATED_EXECUTION_ENVIRONMENT.md; fullpaper/realtime/modelquality
acceptancefalse, LDMfixedsamplingHuber worsened .3103→.6075. No SDK dispatched
bythisledger edit. Parent will invoke generated_sdk_author_v1 once.

### quick trained generation declared2026-10-26
Warmpaper69@934d8221 unchanged. New source-bound offline258pose generation
from realVAE2000+LDM3000, realCLIP selfwritten two-text schedule,32x8 futures,
neutralinitial2 thenownpredictionhistory/noGTfuture. Authoractualexit0,0optim/
SDK/physics. Freshdocument required before SDKexport/physical profile gates.
Model quality remainsunaccepted: fixed samplingHuber .3103→.6075. See
QUICK_GENERATION_ENVIRONMENT.md/current spec. No source/data/weight changes.

### quickLDM3000 actualactivation2026-10-26
Warm934d8221 unchanged. Authorv2/freshdocumentv1 realexit0,195independentchecks,
4AdamW/0skip/0VALupdates, exactmemory model/Adam/ALLRNG/nextbatch and full5CFG5
sample finite. Originalv1function-device failure retained; fixed only explicit
CUDAdevice. Parent enabledtrain+boundtwoactualwitnesses, quick_ldm_train_v1
actualRUNNING; freeze3LDMsources/spec, no repeat commands. Driver4048656 only
historical; use actualreceipt/exit. VAE2000 complete/two privatecheckpoints,
corpus42 accepted. See CURRENT QUICK_LDM_ENVIRONMENT/CONTINUATION.

### quickLDM3000 declared beforedispatch2026-10-26; realVAE2000 passed
Warm934d8221, no dependency/sharedsource changes. RealVAE2000 finished2000
Adam updates/0skips, actualexit0, bothfixedVALposteriorreconstruction gates
passed; full1000/2000 file model/Adam/ALL RNG restore and server+D backups.
Separate quickLDM source/spec includes single513523901B HOME frozenVAE cache,
three300MiB state slots+emergency,2GiBreserve+512MiBlogs. Authorv1 failed after
4diagnosticupdates whenfunctionCFG lacked parameters() for inferreddevice;
original failure preserved. ONLY device='cuda' samplerargument changed;
authorv2 actuallypassed4AdamW/0skip/0valupdate and full5stepCFG5 sampling.
Freshdocumentv1 pending; trainingdisabled untilsource-boundactualpass.
See QUICK_LDM_ENVIRONMENT.md. This is quick3000, not paper300000.

### quickVAE2000, declared beforegradient dispatch2026-10-26
Warm934d8221/source-bound real42-reference corpus. New4update memorywitness, officialEMAfalse/math; fullfile helper unchanged and realtraining snapshots reloaded. Author/freshdoc required beforetrain. See QUICK_VAE_ENVIRONMENT.md.

### quick69-retained-reference corpus, declared2026-10-26 beforebuild
Warm generatorpaper69@934d8221, no dependency changes. Bind actual42privateview,
reference/BABEL, refittrain-onlystats and realCLIPcache; new512loader witness.
Author and fresh documented verify required; no optimizer or quicktraining gate.
See QUICK69_CORPUS_ENVIRONMENT.md and configs/quick69-corpus-spec.json.

# TextOp environment ledger

Storage guard2026-10-26: unchanged warm tracker b7a3bf0c and optimization math.
Formal checkpoint I/O now validates personal/persistent/writable destinations
and a conservative remaining-save budget. CPU-only author_v1 and fresh
document_v1 passed11checks with actualexit0 and zero GPU/checkpoint/model work.
This validates the storage checks only, not full-library numerical capacity.
See CHECKPOINT_STORAGE_GUARD.md and results/checkpoint_storage_guard_20261026.

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

### quick budget2026-10-26 — unchanged warm b7a3bf0c
Human requested faster reproduction; effective5000-round budget is external to
the frozen40k run, preserving all16 runtime files and numerical settings. No
package/GPU-kernel rebuild. New stdlibCPU control uses pidfd via installed Linux
x86_64 headers on Python3.8 (pidfd_open434/send_signal424);3 actual CPU-fixture
checks passed, no training signal.11 guard checks and fresh disabled v2 doc
preflight actualexit0/19checks passed. Original PID race review/fix and minor
docs baseline-relative-path clarification retained. CoordinatorPID50680 waits for
checkpoint+private/public backup before a user-authorized early exit. This is not
an already-executed SDK stop or proof of tracking quality; quick pilot still pending.


### rendering: warm GMR CPU / execution-video@dbb4301b20d56abf

2026-10-26: separate visual-pose profile, no package or shared asset mutation.
MuJoCo3.3.7, PyOpenGL3.1.10, imageio2.33.1/imageio-ffmpeg0.4.9,
NumPy1.26.4/Pillow12.3.0; native GL3.1/Mesa21.2.6 llvmpipe.
Spec execution-video-spec.json was written before first execution. Mesa vendor
forced, CUDA empty, CPU0–3, no libnvidia loaded, no Isaac SDK/dynamics integration.
Author_v1 prelaunch checksum helper failure kept. V2/document_v1 had visible
coplanar floors; intermediate_v3 parsed only first of two worldbody sections and
its count1 was not a compiled count. All old media/receipts/source kept. Current
source handles both sections and asserts a single compiled world plane. Actual
author_v4 + fresh document_v2 each exit0/24encoded+decoded50fps frames, recognized
G1/clear synthetic labels/clean ground independently viewed. No actual policy
video or physical filtering claim. Same mediaSHA b082dfd55601956dc22f2684f2edf0d2332f8e71cec01d8d6b987e40f0035845.
Two witness hashes are bound while actual_rendering_enabled remainsfalse. No
16-file training/quick-control changes; current teacher and5000 coordinator keep
running. This profile does not replace Isaac tracking with MuJoCo control.


## CURRENT — 2026-10-27 user restored FILTER FIRST; prefilter VAE safely paused

User explicitly chose safely save/pause prefilter generator and finish all-library
physical filtering before generator. This supersedes every historical generator-
first/LDM auto-start instruction below. Do not resume prefilter VAE/LDM.
Read GENERATOR_PAUSE_MARKER_ENVIRONMENT.md, TEACHER10000_FILTER_RECOVERY_ENVIRONMENT.md,
reports/filter_first_restored_20261027.md and configs/filter-first-priority-plan-20261027.json.

generator8278_vae200k_train_v1 actually saved emergency_8100.pth,8100 attempts/
8100Adam/0skip,513526637B,SHA2b958a5e4efbd74ddfe6bb99b8a5eb8174cec143bb782689b1f6af8934a6c2fe.
CPU audit actualexit0/1004finite tensors/model+Adam exact reload/CPU RNG restored;
CUDA RNG stored+structurally checked, no actualGPUrestore performed. Server+Dprivate
fullstate SHA verified, source retained; original log/metrics/resources/exits Dprivate.
Worker/supervisor/outer exited1, staleRUNNING8000 receipt deliberately preserved.
New receipt ROOT/receipts/generator8278_user_pause_verified_20261027_v1.json
SHAfba3cef55cfb4afd1cc837d73bb926f3d47838fd4162538d143939ef48d64472 explainsuserstop.
No actual signals, no training source changes. Empty owned receipt.partial marker
at known completebatch boundary triggered original fsynced emergency save; original
firsttraceback/exit/gzipclosure/commonlockrelease passed. Marker removedafterexit.
Earlier SIGINT/findmnt approach independently BLOCKED(B1cleanupjournal/B2emergency-
guardambiguity),neverused ontrainer; allv1-v4 failures/review preserved. Marker
author/fresh each CPU0/fixture1/6checks, fresh14additionalchecks passed. Do NOT
restart any pause controller, fixture, backup or oldtraining job.

NEW actual all-source relaxed_scope_v1 CPUexit0: all8278 sources have real attempts,
15887307total;0 satisfy saved EMA failure<=.05. Min.1866086/median.2465156;
raw cumulative min.1333333, also0below.05. No strictcandidate/acceptedcorpus exists.
Private4.7MB decisions SHA0dbe24ed5055e56ce0b5284bad797cdd5440c855780c4a0f391839c047f401b8
server+D verified. Do not repeat export/load ordelete/reject alloriginalassets as
absolutely infeasible. Pinned EMA startsfailed/success1,alpha.001,stablecount;
history/initialization effects exist, no sole-cause claim. Keep original5% gate.

Separate teacher recovery from exact10k checkpoint prepared; new productionrun
filter_policy_paper69_resume10000_filter_recovery_20261027_v1. Check latestactual
results/teacher10000_full_filter_20261027/progress.json and freshdocument before
any operation. Original20sources/8192/PPO/solidfloor unchanged; new supervisor/spec
bind10k, verifiedgeneratorpause, emptyfullscope statistics andactualresources.
Paper40k upper target retained; inspect next real15000 checkpoint first, do not
require40k before nextreview. Do not duplicate dispatch or reuse oldrun-bound
backup/coordinator. Newactualcheckpoint needsnewrun-bound backup/statistics;
preserve allversions. Generator remains paused until realtwo-round filtering,
final motion/text corpus/norm/CLIP/loader/resume/capacity acceptance.

collect_filter_first_report.py takes priority; collect_generator8278_report.py
now defers to savedpause acceptance so oldRUNNING cannot restoreoldroute. Current
generator assets are comparisononly. Fullpaper quality/semantic/WBC/enhancement/
onlineexecution/videos remain incomplete. Allprotection/privacy/resource limits
and finalcomplete notification requirements remain; no repeatedoldcleanup/tests.


## CURRENT UPDATE — actual teacher recovery is running

Fresh teacher10000_filter_recovery_document_v1 dispatched exactly once after
read-only checks. Newrun filter_policy_paper69_resume10000_filter_recovery_20261027_v1
is RUNNING_FORMAL_FILTER_TRAINING/resume_exact_state_verified=true. Actual native
log sampled10007completed; receipt still0 until10100 per original100roundflush,
not trainingfromzero. Noexit. Historicaldriver4190430/supervisor4190431/bash4190443/
SDK4190451 onlyhints; inspect actualargv/starttime. Shared GPU1 withSDK auxiliary
GPU0 context; original20bindings unchanged. Newly active supervisor/spec frozen.
Use collect_teacher_filter_recovery.py only; do not relaunch or signal. Inspect
next stable15000checkpoint with newrun-bound backup/statistics; don't reuse old
hardcoded5k/10ktools. Generator remainspaused with verified8100privatebackup.

