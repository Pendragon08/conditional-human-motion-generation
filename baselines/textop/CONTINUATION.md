# TextOp: data completion → physical filtering → full generator training

User2026-10-25 explicitly requests completing data preparation/processing, then
starting the full generator. This authorizes the continuation; do not ask again
whether to train. It does not authorize skipping validation or licenses.

## Scope and protected state

- Only TextOp continues. PhyGile/ReactiveBFM remain paused; keep their datasets,
  valid checkpoints and records. Do not touch other users' directories.
- Headless/shared GPU is authorized. Use GPU1, CPU4 and Torch cap0.65; inspect
  actual usage, allocations outside Torch and disk before every major job.
- Preserve old TextOp23DoF/57D SIGSTOP-held process/checkpoint evidence. Do not
  resume it as the29DoF/69D reproduction.
- Raw AMASS/SMPL/BABEL and retargeted arrays stay private. GitHub gets code,
  hashes, aggregate metrics/reports and permitted result videos. Do not publish
  private annotation/motion indexes or account credentials.

## Fixed paths and existing workers

Server user ouyanghao, SSH10.147.17.32, keyC:/Users/Pende/.ssh/id_ed25519.
Root /data/code/ouyanghao/oyh/textop-reproduction-20261024.
Local report folder D:/具身智能/TextOp复现_20261024.
Local repository work/phygile_reproduction_20261017/conditional-human-motion-generation,
branch codex/textop-reproduction-20261024, provider_tools/publish.py scopes staging
to baselines/textop and serializes publishers. Never git-add unrelated files.

Check actual command/PID before any process action; the following IDs are history:

- Current1653 GMR cohort: server3319567; Windows continuation61324. RAM working
  directory /dev/shm/ouyanghao_textop_20261025/local1653_v1, output prepared1653_v1.
  It independently archives its completed output to localD. Do not duplicate backup.
- CMU continuation Windows24952, provider_tools/continue_cmu1343.py. Requested
  cohort1343, selected official SMPL-X G archive matches1330 and lacks13. Private
  missing list is datasets/BABEL/cmu_missing_selected_export.private.json onD.
  It stages and verifies1330 raw inputs in server RAM cmu1330_v1, then waits for
  the previous cohort to finish AND its local archive to verify, before CPU-only
  GMR. Output prepared1330_v1; private D archive paper29_cmu1330_20261025.private.tar.gz.
- BMLrub resume Windows30104, KIT14524; terminal-report worker24176. They only
  download and validate originals; retargeting continuation for them still needs
  implementation. Both preserve6GiB volume reserve and keep failed partials.
  BMLrub completes onC work/textop_reproduction_20261024/private_assets/incoming_amass;
  KIT completes onD:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives.

CMU complete archive is onC in incoming_amass, SHA256
8a276bd0ed23fa72202f80f1dcf8fd9a336eb053ad86233cd47a9a2f51210270,
11,572,134,418bytes, full bzip2 CRC/tar verified. Private member inventory is
CMU_archive_validation.private.json. The oldD CMU partial was verified equal to
the original archive prefix, recovery proof published, then removed.

C duplicate raw1653 extraction cache was evicted only after archive hashes,
server-copy verification, a real archive-restore witness and published recovery
code. Complete originals remain onD; private recovery indexes/tools also reside
in local datasets/recovery1653_private. Do not restart the original1653 launcher
against the evicted cache; use the existing running worker or a deliberate resume.

CLoSD original server dataset remains and matches its protected hash. Only its
local temporary restore-verification copy was removed; its four local archive
pieces AND matching GitHub release assets remain. See duplicate_copy_cleanup.json.

## Verified components

- BABELv1.0 ZIP and5 original JSONs saved locally and onserver under
  data_preparation_20261025/babel_v1_0, plus private original-verified pair index.
  All8291 released sequence IDs/splits/paths and50004 segment labels exactly match
  original text/category/time fields after the official outer-directory removal.
- GMR/SMPL-X environment: configs/data-prep-env-spec.json, canonical87f577cc,
  CPU4, no CUDA; author and independent numerical witness passed. Keep it warm.
- Paper69 core: configs/paper69-env-spec.json canonical934d8221; author and fresh
  doc passed codec/FK/VAE/LDM/CLIP/5-step sampling gradients. No optimization or
  dataset acceptance was performed. Full corpus-loader integration is outstanding.
- Tracker backend: configs/tracker-env-spec.json canonicalb7a3bf0c; see current
  TRACKER_ENVIRONMENT.md. Existing IsaacSim4.5.0/Lab project2.1.0 reused read only.
  Distribution0.36.21 is an internal package identifier, not the project release.
  Author_v5 has actual childexit0 and numerical pass. Native SDKclose hangs;
  a20s explicit process-exit fallback after completed diagnostic receipts releases
  its private resources. Preserve all failure/timeout records and disclose fallback.
  Independent document_v2 result must be checked before claiming document validation.
- scripts/paper_filter_config.py contains paper overrides; do not assume tested.
  Actual MDP integration probe environment_probe_v1 was launched afterauthor_v5;
  inspect supervisor exit and numerical receipt. It uses one8-frame private FK
  reference,8 envs,24 calls, an untrained actor,0 optimizer updates. It is not a
  filter-policy training run or a motion success benchmark.

## Necessary next actions (make verified progress, not only polling)

1. Finish available indexed raw staging/GMR and backups. After BMLrub/KIT archive
   verification, reconcile exact source names/members against the private BABEL
   index; record absent/ambiguous sources, never silently fill/drop them. Stream
   selected raw inputs into guarded personal RAM workspaces where useful; complete
   source packages and private indexes remain persistent. Process cohorts serially
   under existing GMR resource limits; archive final arrays toD and hash-check.
2. Diagnose the13 absent CMU sources (all under18_19_rory,20_21_Justin1,
   22_23_justin in the BABEL-indexed source). The selected SMPL-X G archive has
   no matching basenames. Do not invent a numeric-folder alias. A compatible
   SMPL+H model is not available; SMPL-X cannot silently replace it. Record exact
   obtainable public-data coverage and private403-clip unavailability.
3. Finish Tracker-format export using actual robot FK and name-based ordering.
   Isaac importer has29 actuated joints and30 body frames; generator GMR topology
   has38 non-world frames. Never assume their array orders match. Reuse official
   dtype/loader semantics; validate finite poses,50Hz and body/joint layouts.
4. Implement/run the real privileged filter policy:8192envs,40000iterations,
   actor/critic[2048,1024,512],ELU,24steps/env, lr0.001,4minibatches,5epochs,
   clip0.2,200Hz physics/50Hz control. Actor and critic get557D privileged inputs
   including projected gravity. No domain randomization. Use pelvis anchor.
   IMPORTANT: remove anchor_pos and ee_body_pos termination DURING pretraining,
   not only evaluation. Keep orientation/timeout. Paper config overrides released
   defaults for regularization weights and strict EE threshold0.5m (pelvis0.25m).
   Record unpublished EMA/window/repeat conventions as implementation choices.
   Test actual PPO update/resume and8192-env memory shape before declaring ready.
5. Save trained checkpoint plus failure statistics. Run relaxed evaluation and
   remove failure-probability>0.05; then restore standard conditions and reject
   any failure. Cover whole motions/windows deliberately; do not treat a10s
   timeout as completion of a longer motion. Keep all failed episodes/evidence.
6. Build accepted69D generator corpus from actual retained motions and aligned
   BABEL labels. Preserve train/val source disjointness. Train-only normalization,
   complete frozenCLIP cache (including mirrored text), weights/lengths/data
   interfaces must be validated with the real loader before training. Never use
   PRIVATE-DATA normalization or raw23DoF checkpoints for29DoF.
7. On real acceptance, start a NEW full69D VAE200000-step run, then frozen-VAE
   LDM300000-step run, paper batch512/lr1e-4/history2/future8/primitives4 and network
   settings in configs/paper-step2.json. Validate latent normalization from the
   trained VAE, preserve official losses and rollout caps1.0(VAE)/0.8(LDM).
   Count actual optimizer updates/skips, save resumable checkpoints/logs and
   back up milestones. New authorization is unnecessary; evidence is necessary.
8. Evaluate real trained generation and generation→execution, text switches,
   continuity and failures. Label kinematic references separately from executed
   videos. Never portray an untrained/small diagnostic as paper-scale training.

## Reporting and continuation

Keep current code, aggregate results, failure evidence and reports synchronized
to the userGitHub repository and local folder. Refresh README, data, storage and
training reports; status must distinguish downloaded, retargeted, physically
accepted and actually trained. The exact author-private data/asset/geometry-body
configuration may be unavailable; declare scope/differences rather than inventing.
Only notify on a verified milestone, failure requiring action, completion or
actual required user input. Healthy unchanged background progress stays quiet.

## Continuation update2026-10-25 13:15 (supersedes earlier pending items)

- Backend independent document_v2 passed verbatim with real childexit0 and
  timeoutfalse. SDK stack blocks in SimulationContext.stop/callback/render;
  reaching native SimulationApp.close is not established. Explicit watchdog
  fallback remains disclosed; no shared SDK repair is claimed.
- Actual Tracker MDP environment_probe_v1 passed:8envs,24controls,557D actor
  and critic, no DR, relaxed training/strict0.25m pelvis/0.5m EE assertions.
- export_author_v1 passed complete2 sources+mirrors,350frames, name-based29→30
  body FK and50Hz velocities,0 physics steps. Full643-source export supervisor
  historicalPID3436853 is running; output RAM tracker_references/export643_v1.
  Check complete receipt and back up its private derived arrays before eviction.
- Actual PPO author_v2 passed40 Adam steps/384 transitions plus exact checkpoint
  model/Adam/both-normalizer value,dtype,shape restore and further updates.
  author_v1 did20 updates before CPU/GPU comparison error; original stale0
  receipt plus audit_correction are retained. Do not call this a40k filter run.
  Independent ppo_document_v1 direct invocation also exited0; its local witness
  REPORT.md is in work/textop_reproduction_20261024/tracker_ppo_doc_witness_v1.
  configs/tracker-ppo-diagnostic.json was written before these bounded updates;
  unchanged backend dependencies reused. New procedure TRACKER_PPO_ENVIRONMENT.md.
- BMLrub/KIT indexed continuations are now implemented and running as waiting
  Windows workers53968/23184, provider_tools/continue_remaining_indexed.py.
  They wait for CRC/SHA download and previous GMR+private backup success, then
  exact-key reconciliation, guarded RAM staging, full rawSHA verification,
  same CPU4/noCUDA GMR and private D archive. They never overlap retarget jobs.
  State files bmlrub_indexed_queue.json / kit_indexed_queue.json. Inspect actual
  command and state; never launch a duplicate. Missing/ambiguous exports stay
  explicit. CMU still must finish before BMLrub; BMLrub beforeKIT.
- A thread heartbeat automation idtextop now continues hourly. Old automation
  phygile-reactivebfm stays PAUSED. Follow user changes; no other tasks resumed.
- export643_v1 has now completed all643 source sequences /1286 trajectories,
  numerical PASS. Private D archive worker historicalWindowsPID47568 is running,
  provider_tools/backup_tracker643.py, state reference643_backup.json. It verifies
  every sourceSHA and1286 archive members plus gzipCRC, accounts for remaining
  KIT download and6GiB D reserve. Do not duplicate this backup or evict RAM.
- provider_tools/collect_tracker_progress.py copies only whitelisted public
  receipts/logs/stacks, refreshes local mainHTML and tracker report. Run after
  collect_data_preparation.py; refresh download/storage reports too. Keep all
  diagnostics labelled short functional checks; no motion acceptance yet.

Still required:8192-env/full-library capacity, full40k privileged policy,
actual relaxed/strict filtering, accepted69D corpus loader/cache/statistics,
VAE200k→LDM300k and real generation/execution. The user has already authorized
starting full generator training once these prerequisites actually pass.

## Personal-storage cleanup authorized2026-10-25 (check actual phase)

User explicitly requested clearing only their files because both server disks
are tight. provider_tools/finish_personal_storage_cleanup.py currently handles
the task; historical WindowsPID61664, state results/storage_cleanup_20261025/
continuation.json. Do not duplicate this worker. Server archive phase completed
with82,327 fileSHA checks,10,114,610,390 source bytes,5,769,849,788 archive bytes;
6 pieces in /dev/shm/ouyanghao_storage_20261025. Only one idle EngineAI Python
env .venv-engineai-official and fixed derived/download caches are selected.
No processes or other Python environments depended on it at preflight. Current
TextOp/GMR/IsaacSim/IsaacLab runtimes, datasets/checkpoints/records are excluded.

Every backup piece must have a matching uploaded GitHub Release digest and
code/manifest must be committed before any source unlink. Release tag
inactive-engineai-environment-cache-20261025. The worker streams SSH bytes to
official GitHub through the local proxy, no temporary local disk archive copy.
Earlier waiting worker was replaced before deletion, and a requests-import
failure was corrected to standard-library HTTPS; attempt logs/state are kept.

After verified cleanup, the worker moves /home/ouyanghao/textop-assets-20261024
and phygile-preserved-assets-20261023 to personal/data retained-system-assets-
20261025, SHA-verifies every file and preserves each original/home path as a
link. Assets are retained; moving them frees system disk but consumes data
disk. EngineAI source tree/base Python stay intact; its archived environment
must be restored from ordered Release pieces before any future EngineAI job.
Read final cleanup/relocation receipts and local personal-storage HTML report,
do not assume completion from this handoff. Large CLoSD cache actually holds
datasets/pretrained weights; it was intentionally excluded from deletion.

Cleanup transport update: main worker historicalPID60276 now reuses successful
uploaded parts and retries transient GitHub GET failures; tail-only helper57644
uploads parts4-6 independently, state parallel_tail_upload.json. They only
upload software/cache archive parts; main must verify ALL6 digests and commit
code/index before unlinking source. All earlier attempt states/logs retained.
Do not launch new SDK/training jobs that could change selected cache files
while this cleanup's file snapshot is awaiting application. Existing CPU GMR
continues. Check cleanup continuation first; this is user's latest priority.
The local BMLrub old partial has already been removed after full-archive CRC,
SHA/prefix verification and GitHub-backed recovery evidence, freeing3.31GiB.
