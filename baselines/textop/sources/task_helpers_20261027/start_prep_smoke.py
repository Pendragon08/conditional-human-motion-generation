import pathlib,subprocess,os,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=['/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python',str(r/'scripts/prepare_paper_data.py'),'--spec',str(r/'configs/data-prep-env-spec.json'),'--pairs',str(r/'data_preparation_20261025/available_raw_pairs.private.json'),'--output',str(r/'data_preparation_20261025/smoke_v2'),'--limit','6']
p=subprocess.Popen(cmd,env=env,stdout=open(r/'logs/data_prep_smoke_v2.log','ab'),stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
print('SMOKE_PID',p.pid)
