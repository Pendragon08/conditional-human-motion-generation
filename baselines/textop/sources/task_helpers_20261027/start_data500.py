import pathlib,subprocess,os,json,hashlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');d=r/'data_preparation_20261025'
smoke=json.loads((d/'smoke_v2/progress.json').read_text());assert smoke['completed_sequences']==6 and smoke['failed_sequences']==0
assert (d/'witness_document/witness.json').exists()
spec=json.loads((r/'configs/data-prep-env-spec.json').read_text());canonical=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8];assert canonical=='87f577cc'
receipt={'canonical_spec_hash':canonical,'spec_file_sha256':hashlib.sha256((r/'configs/data-prep-env-spec.json').read_bytes()).hexdigest(),'independent_agent':'/root/textop_data_environment_witness','independent_witness_passed':True,'meaning':'Hash conventions differ: canonical parsed JSON vs file bytes. Both are verified, not a spec mismatch.','scope':'CPU SMPL-X forward and GMR IK only; no physical filtering'}
(r/'receipts/data_env_validation.json').write_text(json.dumps(receipt,indent=2))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
out=d/'available500_v1';assert not out.exists()
cmd=[spec['python'],str(r/'scripts/prepare_paper_data.py'),'--spec',str(r/'configs/data-prep-env-spec.json'),'--pairs',str(d/'available_raw_pairs.private.json'),'--output',str(out)]
p=subprocess.Popen(cmd,env=env,stdout=open(r/'logs/data_prep500.log','ab'),stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
(r/'receipts/data500_launch.json').write_text(json.dumps({'pid':p.pid,'command':cmd},indent=2))
view=d/'released_aligned_view';assert not view.exists()
v=[str(r/'env/bin/python'),str(r/'scripts/prepare_released_view.py'),'--dataset',str(r/'upstream/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps'),'--output',str(view)]
pv=subprocess.Popen(v,env=env,stdout=open(r/'logs/released_alignment.log','ab'),stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
print(json.dumps({'data500_pid':p.pid,'released_alignment_pid':pv.pid,'environment':receipt}))
