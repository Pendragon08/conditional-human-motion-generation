"""Build only the isolated TextOp overlay; the shared base is read only."""
import json,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=json.loads((ROOT/'configs/env-spec.json').read_text())
env=os.environ.copy();env.update(spec['env'])
env.update(HTTP_PROXY='http://127.0.0.1:17897',HTTPS_PROXY='http://127.0.0.1:17897',PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
base=spec['base']['read_only_base']+'/bin/python'
subprocess.run([base,'-m','venv','--system-site-packages',str(ROOT/'env')],env=env,check=True)
py=str(ROOT/'env/bin/python')
for phase in spec['pip_phases']:
    if phase[0].startswith('-e '):break
    subprocess.run([py,'-m','pip','install']+phase,env=env,check=True)
for package in [ROOT/'upstream/deps/isaac_utils',ROOT/'upstream/TextOpRobotMDAR']:
    subprocess.run([py,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(package)],env=env,check=True)
packages=subprocess.run([py,'-m','pip','freeze'],capture_output=True,text=True,env=env,check=True).stdout
(ROOT/'receipts/pip-freeze.txt').write_text(packages)
(ROOT/'receipts/environment_built.json').write_text(json.dumps({'status':'installed; kernel/document validation pending','spec':spec},indent=2))
print('TEXTOP_ENV_INSTALLED',flush=True)
