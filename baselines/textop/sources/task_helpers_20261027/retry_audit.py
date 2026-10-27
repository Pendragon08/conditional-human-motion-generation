import pathlib,subprocess,os,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
(r/'logs/data_audit.log').rename(r/'logs/data_audit_attempt1_failed.log')
launch=json.loads((r/'receipts/data_audit_launch.json').read_text());env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
p=subprocess.Popen(launch['command'],env=env,stdout=open(r/'logs/data_audit.log','ab'),stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
print('AUDIT_RETRY_PID',p.pid)
g=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR/general_motion_retargeting/utils/smpl.py')
text=g.read_text();pos=text.find('def get_smplx_data_offline_fast');print(text[pos:pos+4800])
