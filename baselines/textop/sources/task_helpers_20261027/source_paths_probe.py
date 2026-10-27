import pathlib,json,subprocess
root=pathlib.Path('/data/code/ouyanghao/oyh')
out={}
for rel in ['humanoid-combat-motion/third_party/GMR','humanoid-combat-motion/.venv-retarget','phygile-reproduction-20261017/data/amass/ssm_20261021/environment_extension_v1','paper-faithful-reproduction-20261022']:
    p=root/rel;out[rel]={'exists':p.exists(),'symlink':p.is_symlink(),'listing':[x.name for x in p.iterdir()][:30] if p.is_dir() else []}
    if p.is_dir():out[rel]['notes']=[str(x) for x in p.rglob('*') if x.is_file() and x.name in ['ENVIRONMENT.md','env-spec.json','pyproject.toml','smpl.py']][:15]
print(json.dumps(out))
