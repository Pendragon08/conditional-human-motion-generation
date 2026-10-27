import sys,pathlib,json,hashlib,subprocess
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote,REPO
BASE=REPO/'baselines/textop';ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
sp=BASE/'configs/env-spec.json';spec=json.loads(sp.read_text())
if spec['pip_phases'][0]!=['setuptools==68.2.2','wheel==0.42.0']:spec['pip_phases'].insert(0,['setuptools==68.2.2','wheel==0.42.0'])
sp.write_text(json.dumps(spec,indent=2));sh=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
with (BASE/'SERVER_NOTES.md').open('a') as f:f.write(f'\n### env: textop@{sh}\nBuild metadata fix: pin setuptools68.2.2/wheel0.42.0 before editable packages. Older base setuptools incorrectly labeled RobotMDAR UNKNOWN. Isolated overlay only; base not modified. Validation pending.\n')
subprocess.run(['scp','-i','C:/Users/Pende/.ssh/id_ed25519',str(sp),'ouyanghao@10.147.17.32:'+ROOT+'/configs/env-spec.json'],check=True)
print(remote(f'''
import pathlib,subprocess,os,json
r=pathlib.Path('{ROOT}');py=str(r/'env/bin/python')
env=os.environ.copy();env.update(HTTPS_PROXY='http://127.0.0.1:17897',HTTP_PROXY='http://127.0.0.1:17897',PIP_NO_CACHE_DIR='1',CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
cmds=[[py,'-m','pip','install','setuptools==68.2.2','wheel==0.42.0'],[py,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(r/'upstream/TextOpRobotMDAR')]]
with open(r/'logs/env_finalize.log','a') as log:
    for cmd in cmds:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    p=subprocess.run([py,str(r/'scripts/kernel_witness.py')],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
print('WITNESS_EXIT',p.returncode)
print((r/'logs/env_finalize.log').read_text()[-3000:])
''',timeout=120))
