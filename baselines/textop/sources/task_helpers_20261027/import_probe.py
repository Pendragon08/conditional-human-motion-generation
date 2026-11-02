import pathlib,subprocess,os,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
cmd=[str(r/'env/bin/python'),'-c',"import robotmdar.dtype.abc; import isaac_utils, torch, numpy, importlib.metadata as m; print('TRAIN_IMPORT_OK',isaac_utils.__file__,torch.__version__,numpy.__version__,m.version('robotmdar'))"]
p=subprocess.run(cmd,env=env,cwd=r/'upstream/TextOpRobotMDAR',capture_output=True,text=True)
(r/'receipts/import_probe.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2));print(p.returncode,p.stdout,p.stderr[-4000:])
