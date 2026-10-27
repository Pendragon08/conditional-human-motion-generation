import pathlib,subprocess,json,os
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');d=r/'data_preparation_20261025'
receipt=r/'receipts/raw_reconciliation_launch.json'
if receipt.exists():raise FileExistsError('Existing reconciliation launch preserved')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=[str(r/'env/bin/python'),str(r/'scripts/reconcile_raw_sources.py'),'--dataset',str(r/'upstream/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps'),'--raw-amass','/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass','--previous-pairs',str(d/'available_raw_pairs.private.json'),'--output',str(d)]
p=subprocess.Popen(cmd,env=env,stdout=open(r/'logs/raw_reconciliation.log','ab'),stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
receipt.write_text(json.dumps({'pid':p.pid,'command':cmd},indent=2));print(json.dumps({'pid':p.pid}))
