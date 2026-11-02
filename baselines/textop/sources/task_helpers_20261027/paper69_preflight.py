import pathlib,json,subprocess,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
print(subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)
code="""import importlib.metadata as m,json,pathlib,sys; import mujoco; r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024'); s=json.loads((r/'configs/paper69-env-spec.json').read_text()); model=mujoco.MjModel.from_xml_path(s['robot_xml']); print(json.dumps({'python':sys.version.split()[0],'versions':{k:m.version(k) for k in s['required_versions']},'clip_weights':[{ 'path':p, 'bytes':pathlib.Path(p).stat().st_size} for p in s['clip_weight_candidates'] if pathlib.Path(p).exists()],'model':{'nq':model.nq,'nv':model.nv,'nbody':model.nbody,'njnt':model.njnt}}))"""
result=subprocess.run([str(r/'env/bin/python'),'-c',code],capture_output=True,text=True);print(result.stdout);print(result.stderr);assert result.returncode==0
print(json.dumps({'data_free':shutil.disk_usage(r).free}))
