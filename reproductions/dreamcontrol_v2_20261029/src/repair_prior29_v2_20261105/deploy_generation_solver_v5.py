from pathlib import Path
import hashlib,json,os,shutil,signal,subprocess,time
s=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def change(text,a,b):
    assert text.count(a)==1,a
    return text.replace(a,b)
def file(path,body):
    if path.suffix=='.py': compile(body,str(path),'exec')
    if path.exists(): assert path.read_text()==body,path
    else:
        with path.open('x') as f:f.write(body)
def launch(cmd,out,env=None):
    with out.open('xb') as log:
        return subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=s).pid
old=s/'src/generation_retry_v4'
new=s/'src/generation_retry_v5'
dest=s/'generation_pipeline_v5_launch'
done=dest/'launch.json'
if done.exists():
    print(done.read_text())
    raise SystemExit
assert not (s/'new_references_v5').exists()
assert not (s/'new_references_v4/generation_receipt.json').exists()
plan=json.loads((s/'generation_pipeline_v4_launch/config.json').read_text())
log=(s/'new_references_v4/pick/cohort_0000/postopt.log').read_text()
assert 'ATb_norm=nan' in log and 'trust_region_cost' in log
new.mkdir(exist_ok=True)
for p in old.glob('*.py'):
    body=p.read_text()
    if p.name=='postopt_v2.py':
        body=change(body,'jnp.linalg.norm(delta, axis=-1) - .075','jnp.sqrt(jnp.maximum(jnp.sum(delta ** 2, axis=-1), 1e-12)) - .075')
        body=change(body,'jnp.linalg.norm(vals[r].translation() - ra[4:]) - .045','jnp.sqrt(jnp.maximum(jnp.sum((vals[r].translation() - ra[4:]) ** 2), 1e-12)) - .045')
        body=change(body,'static_argnames=("stage", "penalty")','static_argnames=("stage",)')
    if p.name=='pipeline_controller_v2.py':
        start=body.index('    reuse = plan["reuse_first_pick_raw"]')
        end=body.index('    else:',start)
        body=body[:start]+'''    reuse = plan["reused_pick_raw"].get(str(number))
    if task == "pick" and count == 1 and offset == number and reuse is not None:
        report = bound(reuse)
        require(report.is_relative_to(Path(cfg["stage_dir"])) and report.parent.name == "raw",
                "unexpected reused sampling receipt")
        producer = report.parent
'''+body[end:]
    file(new/p.name,body)
cpu={**os.environ,**plan['postopt_env'],'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','PYTHONDONTWRITEBYTECODE':'1'}
test='''import jax,jax.numpy as j
v=j.zeros(3)
before=jax.jacfwd(lambda x:j.maximum(j.linalg.norm(x)-.045,0.))(v)
after=jax.jacfwd(lambda x:j.maximum(j.sqrt(j.maximum(j.sum(x**2),1e-12))-.045,0.))(v)
assert j.isnan(before).all() and j.isfinite(after).all() and j.all(after==0)
print("zero-anchor Jacobian",before,after)
'''
subprocess.run([plan['postopt_python'],'-B','-c',test],env=cpu,check=True,timeout=60)
# Stop the known defective generation after preserving its raw outputs. The handler
# stops its separately owned worker group and writes a durable failure receipt.
pid=1860824
proc=Path('/proc')/str(pid)/'cmdline'
if proc.exists():
    argv=proc.read_bytes().split(b'\x00')
    assert os.stat(proc).st_uid==os.getuid() and str(old/'pipeline_controller_v2.py').encode() in argv
    assert str(s/'generation_pipeline_v4_launch/config.json').encode() in argv
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+45
    while proc.exists() and (old/'pipeline_controller_v2.py').name.encode() in proc.read_bytes():
        assert time.monotonic()<deadline,'Previous controller did not exit'
        time.sleep(1)
assert json.loads((s/'new_references_v4/status.json').read_text())['status']=='failed_preserved'
reused={'0':plan.pop('reuse_first_pick_raw')}
for n in range(1,8):
    p=s/f'new_references_v4/pick/cohort_{n:04d}/raw/report.json'
    if p.exists():
        r=json.loads(p.read_text())
        if r['status']=='sampling_completed_not_postoptimized' and r['generated_candidates']==1:
            reused[str(n)]={'path':str(p),'sha256':sha(p)}
plan['reused_pick_raw']=reused
plan['output_dir']=str(s/'new_references_v5')
for spec in plan['sources'].values():
    p=Path(spec['path'])
    if p.parent==old:
        p=new/p.name
        spec.update(path=str(p),sha256=sha(p))
dest.mkdir(exist_ok=True)
config=dest/'config.json'
file(config,json.dumps(plan,indent=2)+'\n')
sources={}
for name,oldname in [('evaluation_controller','evaluation_controller'),('supermimic_repaired_controller','supermimic_repaired_controller')]:
    oldpath=s/f'src/{oldname}_v4.py'
    newpath=s/f'src/{name}_v5.py'
    body=oldpath.read_text().replace('new_references_v4','new_references_v5').replace('evaluation_controller_v4','evaluation_controller_v5').replace('supermimic_repaired_v4','supermimic_repaired_v5')
    file(newpath,body)
    sources[name]=newpath
vid=s/'src/report_video_sources_v2.py'
body=(s/'src/report_video_sources_v1.py').read_text().replace('new_references_v4','new_references_v5').replace('supermimic_repaired_v4','supermimic_repaired_v5').replace('report_video_sources_v1','report_video_sources_v2')
file(vid,body)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
gen=launch([plan['sample_python'],'-B','-u',str(new/'pipeline_controller_v2.py'),'--plan',str(config),'--plan-sha256',sha(config)],dest/'controller.log',env)
evaluation=launch(['/usr/bin/python3','-B','-u',str(sources['evaluation_controller'])],s/'evaluation_controller_v5.log',env)
sm=launch(['/usr/bin/python3','-B','-u',str(sources['supermimic_repaired_controller'])],s/'supermimic_repaired_v5.log',env)
video=launch([plan['postopt_python'],'-B','-u',str(vid)],s/'report_video_sources_v2.log',cpu)
receipt={'generation_pid':gen,'evaluation_pid':evaluation,'supermimic_pid':sm,'video_pid':video,'plan_sha256':sha(config),'reused_raw_receipts':reused}
file(done,json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
