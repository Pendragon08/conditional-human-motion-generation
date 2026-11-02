import base64,hashlib,json,pathlib,sys
TASK=pathlib.Path(__file__).resolve().parent;BASE=TASK.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(BASE/'provider_tools'));from session import remote
enc=base64.b64encode((BASE/'scripts/supervise_teacher10000_filter_recovery.py').read_bytes()).decode()
code=r'''import base64,hashlib,json,pathlib,subprocess
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
name='scripts/supervise_teacher10000_filter_recovery.py';p=ROOT/name;assert not p.exists();data=base64.b64decode(ENC);compile(data,name,'exec');p.write_bytes(data)
s=json.loads((ROOT/'configs/full-teacher-resume-spec.json').read_text());s.update(schema='textop.teacher10000_filter_recovery.v1',resume_enabled=False,
 run='filter_policy_paper69_resume10000_filter_recovery_20261027_v1',source_sha256=sha(p),
 generator_pause_acceptance_sha256=sha(ROOT/'receipts/generator8278_user_pause_verified_20261027_v1.json'),
 relaxed_scope_sha256=sha(ROOT/'teacher10000_full_filter_20261027/relaxed_scope_v1/summary.json'),
 route='Filter-first restored by explicit user;10k all-source saved relaxed stats yielded0 eligible at unchanged0.05. Continue teacher, re-evaluate actual new checkpoints, do not train prefilter generator.',
 next_statistics_review_completed=15000,paper_target_is_not_required_before_next_review=True,full_generator_launch_authorized_before_filtering=False)
prior=json.loads((ROOT/'receipts/teacher10000_backup_verified.json').read_text())['audit']['full_checkpoint']
s['checkpoint']={'path':prior['path'],'bytes':prior['bytes'],'sha256':prior['sha256'],'completed':10000,'updates':200000}
assert s['checkpoint']['sha256']=='6f949e4edc50d9a763a3634d716162cb5ddecb079bd1142d3720f907f5efab45'
sp=ROOT/'configs/teacher10000-filter-recovery-spec.json';assert not sp.exists();sp.write_text(json.dumps(s,indent=2))
cmd=['taskset','-c','0-3','python3',str(p),'--preflight-only'];q=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
o=ROOT/'teacher10000_full_filter_20261027/recovery_preflight_v1';o.mkdir(exist_ok=False)
(o/'stdout.txt').write_text(q.stdout);(o/'stderr.txt').write_text(q.stderr);(o/'exit.json').write_text(json.dumps({'exit':q.returncode,'command':cmd}))
print(json.dumps({'exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'spec':s,'spec_b64':base64.b64encode(sp.read_bytes()).decode()}))
'''.replace('ENC',repr(enc));compile(code,'remote','exec')
j=json.loads(remote(code,timeout=90));out=BASE/'results/teacher10000_full_filter_20261027';out.mkdir(exist_ok=True)
(out/'recovery_preflight_v1.json').write_text(json.dumps({k:v for k,v in j.items() if k not in ['spec','spec_b64']},indent=2),encoding='utf-8')
(BASE/'configs/teacher10000-filter-recovery-spec.json').write_bytes(base64.b64decode(j['spec_b64']))
assert j['exit']==0,j['stderr'];print(j['stdout'])
