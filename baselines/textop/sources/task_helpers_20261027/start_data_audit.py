import pathlib,subprocess,os,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
out=r/'data_preparation_20261025';out.mkdir(exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=[str(r/'env/bin/python'),str(r/'scripts/audit_dataset.py'),'--dataset',str(r/'upstream/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps'),'--raw-amass','/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass','--output',str(out)]
log=open(r/'logs/data_audit.log','ab')
p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
(r/'receipts/data_audit_launch.json').write_text(json.dumps({'pid':p.pid,'command':cmd},indent=2));print('AUDIT_PID',p.pid)
source=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR/general_motion_retargeting')
for name in ['__init__.py','motion_retarget.py','utils/smpl.py','params.py']:
    p=source/name
    if p.exists():print('GMR_SOURCE',name,p.read_text()[:2500])
