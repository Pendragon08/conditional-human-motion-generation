import base64,hashlib,json,pathlib,shutil,subprocess,sys
TASK=pathlib.Path(__file__).resolve().parent;BASE=TASK.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop';D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
sys.path.insert(0,str(BASE/'provider_tools'));from session import remote,SSH
for run,public in [('generator_pause_document_v1','rejected_signal_document_v1'),('generator_pause_marker_document_v1','marker_document_v1')]:
 src=TASK/run;out=BASE/'results/generator_pause_20261027'/public;out.mkdir(exist_ok=False)
 for n in ['REPORT.md','SUMMARY.json']:
  shutil.copy2(src/n,out/('independent_summary.json' if n=='SUMMARY.json' else n))
 dest=D/'diagnostics'/run;assert not dest.exists();shutil.copytree(src,dest)
code=r'''import base64,hashlib,json,pathlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sp=r/'configs/generator8278-marker-pause-spec.json';s=json.loads(sp.read_text())
names=list(s['frozen_sources'])+['scripts/pause_generator8278_with_marker.py','configs/generator8278-marker-pause-spec.json']
runtime={n:{'sha256':hashlib.sha256((r/n).read_bytes()).hexdigest(),'base64':base64.b64encode((r/n).read_bytes()).decode()} for n in names}
prefix='generator8278_vae_generator8278_vae200k_train_v1';run=r/'generator8278_20261027/vae_runs/generator8278_vae200k_train_v1'
files=[run/'metrics.jsonl.gz',r/'logs'/(prefix+'.log'),r/'logs'/(prefix+'_resources.jsonl')]
files += [r/'receipts'/(prefix+x) for x in ['_exit.json','_process_exit.json']]
files += [r/'generator_pause_20261027/generator8278_to_filtered_order_v1'/n for n in ['cpu_audit_exit.json','cpu_audit_stdout.txt','cpu_audit_stderr.txt']]
print(json.dumps({'runtime':runtime,'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}))
'''
j=json.loads(remote(code,timeout=30));(BASE/'results/generator_pause_20261027/exact_runtime_bytes.json').write_text(json.dumps(j['runtime'],indent=2),encoding='utf-8')
dest=D/'diagnostics/generator8278_user_pause_actual_20261027';dest.mkdir(exist_ok=False)
for item in j['files']:
 target=dest/pathlib.Path(item['path']).name;assert not target.exists()
 q=subprocess.run(['scp']+SSH[1:-1]+[SSH[-1]+':'+item['path'],str(target)],capture_output=True,timeout=120)
 assert q.returncode==0,q.stderr
 assert target.stat().st_size==item['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']
(dest/'manifest.json').write_text(json.dumps(j['files'],indent=2),encoding='utf-8')
summary={'files':j['files'],'local_private_dir':str(dest),'all_sha_verified':True,'no_checkpoint_deleted':True}
(BASE/'results/generator_pause_20261027/original_logs_private_backup.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
plan={'schema':'textop.filter_first_priority.v1','user_request':'安全保存并暂停当前生成器，先完成全库筛选','prefilter_generator_paused':True,'prefilter_generator_resume_authorized':False,'prefilter_ldm_autostart_authorized':False,
 'filter_teacher_checkpoint_completed':10000,'whole_library_relaxed_source_count':8278,'current_saved_ema_eligible':0,'strict_replay_completed':False,'filtered_corpus_accepted':False,
 'generator_resume_acceptance_required':['actual_relaxed_strict_filtering','motion_text_source_isolation','TRAIN_only_statistics','complete_real_CLIP_cache','loader_gradient_resume','persistent_storage_capacity'],
 'new_teacher_statistics_review_at':15000,'paper_teacher_reference_target':40000,'require_40000_before_next_review':False,'paper_generator_vae_attempts':200000,'paper_generator_ldm_attempts':300000,
 'retain_all_original_assets_and_checkpoints':True,'complete_paper_effect_notification_required':True}
(BASE/'configs/filter-first-priority-plan-20261027.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
current='''## CURRENT — 2026-10-27 user restored FILTER FIRST; prefilter VAE safely paused

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

'''
for name in ['CONTINUATION.md','README.md']:
 p=BASE/name;p.write_text(current+p.read_text(encoding='utf-8'),encoding='utf-8')
p=BASE/'SERVER_NOTES.md';p.write_text(p.read_text(encoding='utf-8')+'\n\n'+current,encoding='utf-8')
subprocess.run([sys.executable,str(BASE/'provider_tools/collect_filter_first_report.py')],check=True)
print(json.dumps({'private_logs_sha_verified':len(j['files']),'route':'FILTER_FIRST','checkpoint_preserved':True}))
