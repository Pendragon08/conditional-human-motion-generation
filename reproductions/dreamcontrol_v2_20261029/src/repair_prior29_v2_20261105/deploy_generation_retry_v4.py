from pathlib import Path
import hashlib,json,os,shutil,subprocess
s=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(text,a,b):
    assert text.count(a)==1,a
    return text.replace(a,b)
def launch(cmd,output):
    with output.open('xb') as log:
        return subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=s,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}).pid
old=s/'src/generation_retry_v3'
new=s/'src/generation_retry_v4'
assert json.loads((s/'new_references_v3/status.json').read_text())['status']=='failed_preserved'
raw=s/'new_references_v3/pick/cohort_0000/raw'
v=json.loads((raw/'report.json').read_text())
assert v['generated_candidates']==1 and v['checkpoint_sha256']=='f73da1fdc88d033c5b5fd277aabd9d280c2aa1d4cbe94db09589feab8c491fd6'
plan=json.loads((s/'generation_pipeline_v3_launch/config.json').read_text())
new.mkdir(exist_ok=True)
for f in old.glob('*.py'):
    if not (new/f.name).exists(): shutil.copyfile(f,new/f.name)
controller=new/'pipeline_controller_v2.py'
body=(old/controller.name).read_text()
i=body.index('    runner("sample",')
j=body.index('    rows = producer_records(',i)
body=body[:i]+'''    reuse = plan["reuse_first_pick_raw"]
    if (task, number, offset, count) == ("pick", 0, 0, 1):
        report = bound(reuse)
        require(report == Path(cfg["stage_dir"]) / "new_references_v3/pick/cohort_0000/raw/report.json",
                "unexpected reused sampling receipt")
        producer = report.parent
    else:
'''+''.join('    '+line if line.strip() else line for line in body[i:j].splitlines(keepends=True))+body[j:]
compile(body,str(controller),'exec')
if controller.read_text()!=body:
    assert controller.read_bytes()==(old/controller.name).read_bytes()
    controller.write_text(body)
plan['output_dir']=str(s/'new_references_v4')
plan['reuse_first_pick_raw']={'path':str(raw/'report.json'),'sha256':sha(raw/'report.json')}
plan['postopt_env']['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
plan['postopt_env']['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.20'
for spec in plan['sources'].values():
    p=Path(spec['path'])
    if p.parent==old:
        p=new/p.name
        spec.update(path=str(p),sha256=sha(p))
dest=s/'generation_pipeline_v4_launch'
dest.mkdir()
config=dest/'config.json'
config.write_text(json.dumps(plan,indent=2)+'\n')
pid=launch([plan['sample_python'],'-B','-u',str(controller),'--plan',str(config),'--plan-sha256',sha(config)],dest/'controller.log')
eval_path=s/'src/evaluation_controller_v4.py'
eval_body=(s/'src/evaluation_controller_v3.py').read_text().replace('new_references_v3','new_references_v4').replace('evaluation_controller_v3','evaluation_controller_v4')
compile(eval_body,str(eval_path),'exec')
with eval_path.open('x') as f: f.write(eval_body)
epid=launch(['/usr/bin/python3','-B','-u',str(eval_path)],s/'evaluation_controller_v4.log')
receipt={'generation_pid':pid,'generation_plan_sha256':sha(config),'generation_controller_sha256':sha(controller),'reused_raw_report_sha256':sha(raw/'report.json'),'evaluation_pid':epid,'evaluation_controller_sha256':sha(eval_path)}
(dest/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
