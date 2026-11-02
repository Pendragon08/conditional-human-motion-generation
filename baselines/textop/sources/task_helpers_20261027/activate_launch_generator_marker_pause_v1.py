"""One explicitly authorized marker request, only after fresh independent approval."""
import base64,hashlib,json,pathlib,sys
TASK=pathlib.Path(__file__).resolve().parent;BASE=TASK.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(BASE/'provider_tools'));from session import remote
review=TASK/'generator_pause_marker_document_v1/SUMMARY.json'
assert review.exists(),'Fresh actual review must exist'
review_sha=hashlib.sha256(review.read_bytes()).hexdigest()
code=r'''import hashlib,json,os,pathlib,subprocess,sys
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sys.path.insert(0,str(ROOT/'scripts'))
from closed_boundary_interrupt import identity,require_identity
sp=ROOT/'configs/generator8278-marker-pause-spec.json';s=json.loads(sp.read_text());assert s['pause_authorized'] is False
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in s['frozen_sources'].items():assert sha(ROOT/n)==h,n
require_identity(identity(s['process_identity']['pid']),s['process_identity'])
w=[]
for run in ['marker_author_v1','marker_document_v1']:
 rp=ROOT/'pause_marker_fixtures'/run/'receipt.json';r=json.loads(rp.read_text())
 assert r['status']=='CPU_CLOSED_BOUNDARY_MARKER_PASSED' and all(r['checks'].values()) and r['gpu_calls']==0
 assert r['source_sha256']['closed_boundary_marker.py']==s['frozen_sources']['scripts/closed_boundary_marker.py']
 w.append({'run':run,'sha256':sha(rp)})
audit=ROOT/'pause_marker_fixtures/production_prepare_v1';(audit/'marker_spec.disabled.json').write_bytes(sp.read_bytes())
s.update(pause_authorized=True,cpu_witnesses=w,independent_review_sha256=REVIEW_SHA)
sp.write_text(json.dumps(s,indent=2))
driver=ROOT/'generator8278_20261027/drivers/user_marker_pause_v1.py';assert not driver.exists()
source="import json,pathlib,subprocess,time\nroot=pathlib.Path("+repr(str(ROOT))+')\n'+"started=time.time()\ncmd=['taskset','-c','0-3','python3',str(root/'scripts/pause_generator8278_with_marker.py'),'--execute']\ncode=subprocess.call(cmd)\n(root/'generator8278_20261027/drivers/user_marker_pause_v1_exit.json').write_text(json.dumps({'returncode':code,'elapsed_seconds':time.time()-started,'command':cmd},indent=2))\nraise SystemExit(code)\n"
compile(source,str(driver),'exec');driver.write_text(source)
log=ROOT/'logs/user_marker_pause_v1.log';assert not log.exists()
with log.open('xb') as handle:
 proc=subprocess.Popen(['taskset','-c','0-3','python3',str(driver)],stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=ROOT)
print(json.dumps({'spec':s,'driver_pid':proc.pid,'spec_sha256':sha(sp),'driver_sha256':sha(driver),'training_signals':0,'one_marker_controller_dispatched':True}))
'''.replace('REVIEW_SHA',repr(review_sha));compile(code,'launch','exec')
j=json.loads(remote(code,timeout=30))
(BASE/'configs/generator8278-marker-pause-spec.json').write_text(json.dumps(j['spec'],indent=2),encoding='utf-8')
(BASE/'results/generator_pause_20261027/marker_activation_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in j.items() if k!='spec'}))
