from pathlib import Path
import hashlib,json,os,subprocess
s=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
p=s/'src/report_video_sources_v1.py'
assert p.is_file()
compile(p.read_bytes(),str(p),'exec')
plan=json.loads((s/'generation_pipeline_v4_launch/config.json').read_text())
env={**os.environ,**plan['postopt_env'],'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','PYTHONDONTWRITEBYTECODE':'1'}
launch=s/'report_video_sources_v1_launch.json'
if launch.exists():
    print(launch.read_text())
else:
    assert not (s/'report_video_sources_v1').exists()
    py=plan['postopt_python']
    subprocess.run([py,'-B','-c','import viser, jaxlie, pyroki, yourdfpy; print(viser.__version__)'],env=env,check=True,timeout=30)
    with (s/'report_video_sources_v1.log').open('xb') as log:
        process=subprocess.Popen([py,'-B','-u',str(p)],env=env,cwd=s,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'pid':process.pid,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    launch.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))
