import pathlib,subprocess,json,hashlib
root=pathlib.Path('/data/code/ouyanghao/oyh')
paths=[]
for base in [root/'phygile-reproduction-20261017/upstream',root/'phygile-unitree-20261022']:
 for p in base.rglob('main.urdf'):
  paths.append(str(p))
print(json.dumps({'paths':paths[:25],'isaac_wrapper':(root/'humanoid-combat-motion/third_party/isaacsim-4.5.0/python.sh').exists()}))
