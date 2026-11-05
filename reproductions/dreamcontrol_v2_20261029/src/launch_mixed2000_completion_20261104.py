import hashlib,json,os,pathlib,subprocess,time
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104'
J=B/'mixed2000_completion_v1';preview=B/'mixed_contract_preview8_v1'
P='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'
assert not J.exists() and not (B/'mixed2000_completion_v1_launch.json').exists()
check=json.loads((preview/'native_loader_check_v1/report.json').read_text())
assert check['status']=='preview_native_library_passed' and check['reference_count']==8 and check['optimizer_updates']==0
sources=[B/'src'/name for name in ['stage3_supported_mixed2000_prepare.py','stage3_supported_native_library_check.py','stage3_supported_mixed_completion.py']]
for path in sources:compile(path.read_text(),str(path),'exec')
assert hashlib.sha256(sources[1].read_bytes()).hexdigest()==check['worker_sha256']
log=(B/'mixed2000_completion_v1.log').open('x')
command=['nice','-n','10',P,'-B','-u',str(sources[2])]
child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=str(B),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},start_new_session=True)
receipt={'pid':child.pid,'command':command,'status_directory':str(J),'preview_report_sha256':hashlib.sha256((preview/'native_loader_check_v1/report.json').read_bytes()).hexdigest(),
 'source_bindings':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'formal_training_started':False,'scope':'CPU-only final dataset preparation after all1500 genuine scene admissions'}
(B/'mixed2000_completion_v1_launch.json').open('x').write(json.dumps(receipt,indent=2))
time.sleep(1);assert child.poll() is None and json.loads((J/'status.json').read_text())['status']=='waiting_all1500_scene_admissions'
print(json.dumps(receipt))
