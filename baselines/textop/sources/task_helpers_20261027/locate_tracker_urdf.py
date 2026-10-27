import pathlib,os,json
r=pathlib.Path('/data/code/ouyanghao/oyh');found=[]
for base,dirs,files in os.walk(r):
 dirs[:]=[d for d in dirs if d not in ['.git','node_modules','site-packages','models_lockedhead','remaining_20261021','train','test','smplx','smplh']]
 if 'main.urdf' in files:found.append(str(pathlib.Path(base)/'main.urdf'))
print(json.dumps({'main_urdf_paths':found[:30]}))
