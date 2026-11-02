"""Create only independent future-checkpoint tooling; never edit active files."""
from pathlib import Path
import hashlib,json
TASK=Path(__file__).resolve().parent
B=TASK.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((B/'configs/full-teacher-backup-spec.json').read_text(encoding='utf-8'))
resume=json.loads((B/'configs/teacher10000-filter-recovery-spec.json').read_text(encoding='utf-8'))
progress=json.loads((B/'results/teacher10000_full_filter_20261027/progress.json').read_text(encoding='utf-8'))
files=['scripts/filter_recovery_backup_protocol.py','scripts/filter_recovery_statistics.py',
       'scripts/inspect_filter_recovery_checkpoint.py','scripts/validate_filter_recovery_backup.py']
spec=dict(schema='textop.filter_recovery_backup.v1',run=resume['run'],
 resume_spec_sha256='f39e2103ff6e9cceb3f7442f83a5c217328cb2cb18bce9cecfabb4fd774a64b4',
 resumed_checkpoint=resume['checkpoint'],training_receipt_bindings=old['training_receipt_bindings'],
 contact_profile=progress['receipt']['contact_profile'],bound_backup_files={n:sha(B/n) for n in files},
 audit_run='filter_recovery_author_v1',backup_enabled=False,
 positive_recovery_checkpoint_audit_completed=False,strict_physical_filter_completed=False,
 next_review_completed=15000,metadata_validation_only_until_real_checkpoint=True,
 validation_scope='New run-bound metadata and saved-counter logic; unchanged trusted CPU serializer. Actual new snapshot load/export/backup requires stable latest receipt and remains pending.',
 local_private_reserve_bytes=2*1024**3)
spec['bound_backup_files'].update({
 'scripts/inspect_formal_filter_checkpoint_v2.py':old['bound_backup_files']['scripts/inspect_formal_filter_checkpoint_v2.py'],
 'upstream/TextOpTracker/source/textop_tracker/textop_tracker/tasks/tracking/mdp/commands_multi.py':'b838ea5d5aa985b7635e11d2d3de85ba5714c57f5cdd3e50dbf925937460a9eb'})
p=B/'configs/filter-recovery-backup-spec.json';assert not p.exists();p.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8',newline='\n')
s=(B/'provider_tools/backup_full_teacher_checkpoint.py').read_text(encoding='utf-8')
replacements={
 'filter_policy_paper69_resume5000_to40000_20261026_v1':resume['run'],
 "TAG = 'textop-filter-policy29d-20261026'":"TAG = 'textop-filter-recovery29d-20261027'",
 'range(10000, 40001, 5000)':'range(15000, 40001, 5000)',
 'results/full_teacher_checkpoints_20261026':'results/filter_recovery_checkpoints_20261027',
 'full_teacher_round':'filter_recovery_round',
 'full-teacher-backup-spec.json':'filter-recovery-backup-spec.json',
 'full_teacher_backup_protocol':'filter_recovery_backup_protocol',
 'inspect_full_teacher_checkpoint.py':'inspect_filter_recovery_checkpoint.py',
 "'full_teacher_checkpoint_cpu_'":"'filter_recovery_checkpoint_cpu_'",
 "assert shutil.disk_usage(dest).free > item['bytes'] + 1024**3":"assert shutil.disk_usage(dest).free > item['bytes'] + spec['local_private_reserve_bytes']",
 'Training target: 40,000 iterations on 8,278 available motion sources.':'Continued from the verified 10,000-round checkpoint on 8,278 sources. The next review is at 15,000 rounds; the 40,000-round paper reference is retained.',
}
for x,y in replacements.items():assert x in s,x;s=s.replace(x,y)
s=s.replace("ap.add_argument('--completed', type=int, required=True)","ap.add_argument('--completed', type=int, required=True)\n    ap.add_argument('--preflight-only', action='store_true')")
needle="    folder = B / 'results/filter_recovery_checkpoints_20261027' / ('round_' + str(a.completed))"
pre='''    # Readiness does not create a backup directory or import Torch.
    spec_path = B / 'configs/filter-recovery-backup-spec.json'
    expected_spec = sha(spec_path)
    code = "import sys,json,pathlib; r=pathlib.Path(%r); sys.path.insert(0,str(r/'scripts')); from filter_recovery_backup_protocol import preflight,sha; p=r/'configs/filter-recovery-backup-spec.json'; assert sha(p)==%r; s=json.loads(p.read_text()); print(json.dumps(preflight(r,s,%d)))" % (ROOT,expected_spec,a.completed)
    ready = json.loads(remote(code,timeout=90))
    if a.preflight_only or not ready['ready']:
        print(json.dumps(ready)); return
    assert json.loads(spec_path.read_text(encoding='utf-8'))['backup_enabled'] is True
'''
assert needle in s;s=s.replace(needle,pre+needle)
needle="ap=json.loads(adapter.read_text())"
assert needle in s;s=s.replace(needle,needle+"\nsummary=json.loads((out/'relaxed_summary.json').read_text())\nassert ap['relaxed_summary_sha256']==sha(out/'relaxed_summary.json')\nassert summary['run']==s['run'] and summary['completed_learning_iterations']==COMPLETED\nassert summary['source_count']==8278 and summary['checkpoint_sha256']==proof['full_checkpoint']['sha256']\nassert summary['dataset_accepted'] is False and summary['strict_evaluation_completed'] is False\nprivate=pathlib.Path(summary['private_decisions']['path']);assert private.parent==out and sha(private)==summary['private_decisions']['sha256']")
needle="'preflight':after}))"
assert needle in s;s=s.replace(needle,"'preflight':after,'relaxed_summary':summary}))")
needle="        proof = audited['audit']"
assert needle in s;s=s.replace(needle,needle+"\n        (folder / 'relaxed_summary.json').write_text(json.dumps(audited['relaxed_summary'], indent=2), encoding='utf-8')\n        proof['private_relaxed_statistics'] = audited['relaxed_summary']['private_decisions']")
needle="for key, category in [('full_checkpoint', 'private'), ('public_inference', 'public')]:"
assert needle in s;s=s.replace(needle,"for key, category in [('full_checkpoint', 'private'), ('public_inference', 'public'), ('private_relaxed_statistics', 'private_statistics')]:")
needle="            dest = D / 'checkpoints' / category / RUN"
assert needle in s;s=s.replace(needle,needle+"\n            if category == 'private_statistics':\n                dest = D / 'datasets/filter_recovery_statistics_private_20261027' / ('round_' + str(a.completed))")
needle="        state.update(status='CPU_AUDITED_DOWNLOAD_PENDING'"
assert needle in s;s=s.replace(needle,"        state.update(status='CPU_AUDITED_DOWNLOAD_PENDING'",1)
needle="                     convergence_claimed=False, full_generator_started=False)"
assert needle in s;s=s.replace(needle,"                     private_relaxed_statistics_two_location_sha_verified=True,\n                     relaxed_counts=audited['relaxed_summary']['counts'], strict_evaluation_completed=False,\n                     dataset_accepted=False, convergence_claimed=False, full_generator_started=False)")
p=B/'provider_tools/backup_filter_recovery_checkpoint.py';assert not p.exists();p.write_text(s,encoding='utf-8',newline='\n')
compile(s,str(p),'exec')
print(json.dumps({'created_files':files+[str(p.relative_to(B)),'configs/filter-recovery-backup-spec.json'],'active_files_modified':0,'backup_enabled':False}))
