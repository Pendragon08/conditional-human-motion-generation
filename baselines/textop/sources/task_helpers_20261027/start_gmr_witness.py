import pathlib,subprocess,os,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
cmd=['/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python',str(r/'scripts/prepare_paper_data.py'),'--spec',str(r/'configs/data-prep-env-spec.json'),'--pairs',str(r/'data_preparation_20261025/available_raw_pairs.private.json'),'--output',str(r/'data_preparation_20261025/witness_author'),'--witness','--limit','1']
p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120)
log=r/'logs/gmr_witness_author.log'
if log.exists():log.rename(r/'logs'/('gmr_witness_attempt_'+str(__import__('time').time_ns())+'.log'))
log.write_text(p.stdout+'\n'+p.stderr)
print('WITNESS_EXIT',p.returncode);print(p.stdout[-3500:],p.stderr[-2500:])
print(subprocess.run(['df','-hP'],capture_output=True,text=True).stdout)
