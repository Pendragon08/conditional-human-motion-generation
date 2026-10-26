"""New run-bound adapter reusing the unchanged, previously verified CPU auditor."""
import argparse,hashlib,json,os,pathlib,sys
from full_teacher_backup_protocol import RUN,preflight,sha
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--completed',type=int,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sp=ROOT/'configs/full-teacher-backup-spec.json';s=json.loads(sp.read_text());assert sha(__file__)==s['bound_backup_files']['scripts/inspect_full_teacher_checkpoint.py']
result=preflight(ROOT,s,a.completed)
if a.preflight_only or not result['ready']:
    print(json.dumps(result));raise SystemExit(0)
assert s['backup_enabled']
# The saved trusted file will still be re-hashed, CPU-loaded, finite-checked and
# validated by the original auditor before any optimizer-free export is produced.
import inspect_formal_filter_checkpoint_v2 as core
assert sha(core.__file__)==s['bound_backup_files']['scripts/inspect_formal_filter_checkpoint_v2.py']
core.RUN=RUN
sys.argv=[core.__file__,'--completed',str(a.completed),'--audit-run',s['audit_run']]
core.main()
destination=ROOT/'checkpoint_backups'/('round_'+str(a.completed))/s['audit_run']
audit=json.loads((destination/'audit.json').read_text());assert audit['run']==RUN
assert audit['source_sha256']==s['bound_backup_files']['scripts/inspect_formal_filter_checkpoint_v2.py']
proof={'status':'RESUMED_CHECKPOINT_ADAPTER_CPU_AUDIT_COMPLETE','run':RUN,'completed':a.completed,
       'adapter_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'scripts/full_teacher_backup_protocol.py'),
       'spec_sha256':sha(sp),'preflight':result,'audit_sha256':sha(destination/'audit.json'),
       'sdk_runs':0,'optimizer_updates':0,'public_export_omits_optimizer_and_motion_statistics':True}
with (destination/'adapter_proof.json').open('x') as f:json.dump(proof,f,indent=2)
