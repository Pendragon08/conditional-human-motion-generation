"""Prepare one disabled continuation, reusing unchanged paper training files."""
import ast,base64,hashlib,json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
name='scripts/supervise_full_teacher_resume.py';raw=(B/name).read_bytes();ast.parse(raw.decode('utf-8'))
bindings=json.loads((B/'configs/formal-contact-activation.json').read_text(encoding='utf-8'))['deployed_files']
checkpoint=json.loads((B/'results/formal_filter_checkpoints_20261026/round_5000/backup.json').read_text(encoding='utf-8'))['audit']['full_checkpoint']
s={'schema':'textop.full_teacher_resume.v1','resume_enabled':False,'run':'filter_policy_paper69_resume5000_to40000_20261026_v1',
   'source_sha256':hashlib.sha256(raw).hexdigest(),'frozen_bindings':bindings,
   'checkpoint':dict(checkpoint,completed=5000,updates=100000),'target_completed_iterations':40000,
   'functional_acceptance':'receipts/quick_functional_acceptance_20261026_v1.json','evidence':[],
   'full_generator_training_enabled':False,'paper_quality_accepted':False,
   'generator_followup':'Only after full available-data physical filtering and final corpus/normalization/CLIP/resume/storage acceptance, new paper VAE200000 and LDM300000 runs; quick compressed clocks are not exact paper resumes.'}
code='''import ast,base64,hashlib,json,pathlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');s=SPEC
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for n,h in s['frozen_bindings'].items():assert sha(r/n)==h,n
p=r/NAME;sp=r/'configs/full-teacher-resume-spec.json';assert not p.exists() and not sp.exists()
raw=base64.b64decode(DATA);ast.parse(raw.decode('utf-8'));p.write_bytes(raw)
sp.write_text(json.dumps(s,indent=2)+'\\n')
proof={'status':'DISABLED_FULL_TEACHER_RESUME_ENTRY_STAGED','source_sha256':sha(p),'spec_sha256':sha(sp),'original_training_files_unchanged':True,'sdk_dispatches':0}
print(json.dumps({'proof':proof,'spec_bytes':base64.b64encode(sp.read_bytes()).decode()}))
'''.replace('SPEC',repr(s)).replace('NAME',repr(name)).replace('DATA',repr(base64.b64encode(raw).decode()))
j=json.loads(remote(code));(B/'configs/full-teacher-resume-spec.json').write_bytes(base64.b64decode(j['spec_bytes']))
o=B/'results/full_teacher_resume_20261026';o.mkdir(exist_ok=False)
(o/'stage_v1.json').write_text(json.dumps(j['proof'],indent=2),encoding='utf-8')
(o/'disabled_spec_v1.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
print(json.dumps(j['proof']))
