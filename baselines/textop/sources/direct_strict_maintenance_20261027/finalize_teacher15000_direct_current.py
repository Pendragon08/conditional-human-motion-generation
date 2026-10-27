import pathlib,json,hashlib,shutil,datetime
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
T=W/'work/textop_reproduction_20261024';O=B/'results/teacher15000_direct_strict_20261027'
c=json.loads((O/'current.json').read_text(encoding='utf-8'));assert c['teacher_stopped'] and c['status']=='RUNNING_TEACHER15000_STRICT'
for source,label in [('teacher15000_control_document_v1','control_document'),('teacher15000_strict_document_v1','strict_document')]:
 d=O/label;d.mkdir(exist_ok=True)
 for name in ['REPORT.md','SUMMARY.json']:
  src=T/source/name;dst=d/name;assert not dst.exists();shutil.copy2(src,dst)
M=B/'sources/direct_strict_maintenance_20261027';M.mkdir(exist_ok=False)
for name in ['prepare_teacher15000_control.py','prepare_teacher15000_strict.py','deploy_teacher15000_strict.py','check_compact_strict_trace.py','activate_teacher15000_direct_strict.py','finalize_teacher15000_direct_current.py']:
 shutil.copy2(T/name,M/name)
(M/'.gitattributes').write_text('*.py -text\n',encoding='utf-8')
(M/'README.md').write_text('Archived maintenance source from work/textop_reproduction_20261024, retained for recovery/audit. Do not execute these snapshots from this folder or repeat their completed stop, deploy, activation or fixture work. Active run instructions are in TEACHER15000_DIRECT_STRICT_ENVIRONMENT.md.\n',encoding='utf-8')
current=f'''## CURRENT — teacher STOPPED; direct all8278 strict replay RUNNING

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

Latest observation {c['sample_beijing']}: finished{c['finished']}, passed{c['passed_complete']},
failed{c['failed']}, pending/inprogress{c['pending_or_in_progress']}; see current.json
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

'''
for name in ['CONTINUATION.md','README.md']:
 p=B/name;old=p.read_text(encoding='utf-8');assert not old.startswith('## CURRENT — teacher STOPPED;');p.write_text(current+old,encoding='utf-8')
p=B/'SERVER_NOTES.md';p.write_text(current+p.read_text(encoding='utf-8'),encoding='utf-8')
p=B/'SYNC_SCOPE_AND_DATA_RECOVERY.md';p.write_text('## 2026-10-27追加：15000轮已真实核验并开始直接严格回放\n\n15000轮完整状态与逐源统计已在服务器和D盘私有核验，公开推理30,208,218字节已上传Release textop-filter-recovery29d-20261027并核验digest。跟踪器停止、两次实际入口核验和全库直接回放代码/聚合结果本次增量同步；真实轨迹和逐源结果仍私有。历史“未来15k”说明已由这些实际收据覆盖。\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'status':'CURRENT_STOPPED_AND_DIRECT_FULL_RUNNING','files_written_without_training_changes':True}))
