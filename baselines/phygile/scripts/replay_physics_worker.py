"""Seeded isolated G1 simulation; replay accepted history before a proposal."""
from pathlib import Path
import argparse,hashlib,json,os,sys,traceback
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from isaaclab.app import AppLauncher
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
os.environ['G1_TRAINING_CFG_PATH']=str(ROOT/'configs/moe_pilot.yaml')
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'reimplementation'))
from guarded_bootstrap import activate
activate()
app=AppLauncher(args).app
import numpy as np
import torch
from codec262 import G1Codec
from isaac_prefix_adapter import IsaacPrefixAdapter
from replay_prefix_adapter import load_motion,save_motion


def digest(module):
    h=hashlib.sha256()
    for name,tensor in module.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    request=json.loads(args.request.read_text());out=Path(request['output'])
    import random
    random.seed(request['seed']);np.random.seed(request['seed'])
    torch.manual_seed(request['seed']);torch.cuda.manual_seed_all(request['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    checkpoint=Path(request['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=request['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    runtime=IsaacPrefixAdapter(checkpoint,request['motion_file'],args.device,request['seed'],
        G1Codec(ROOT/'configs/g1_codec_skeleton.json'))
    physical={}
    for name in ('get_masses','get_inertias','get_coms','get_material_properties','get_dof_stiffnesses','get_dof_dampings'):
        physical[name]=getattr(runtime.robot.root_physx_view,name)().cpu().numpy()
    np.savez_compressed(out/'initial_physical_properties.npz',**physical)
    before=digest(runtime.runner.alg.policy);replays=[]
    for history in request['history']:
        candidate=load_motion(history['candidate']);expected=load_motion(history['actual'])
        assert candidate.digest()==history['candidate_sha256'] and expected.digest()==history['actual_sha256']
        execution=runtime.rollout(candidate,history['prefix_frames'])
        if len(execution.continuation)!=len(expected):raise AssertionError('Replay changed termination horizon')
        error=float(np.abs(execution.continuation.states-expected.states).max())
        if error>2e-6:raise AssertionError(('Accepted history replay diverged',error))
        if execution.terminated.any():raise AssertionError('Accepted history contains a termination')
        replays.append({'candidate_sha256':candidate.digest(),'max_state_difference':error})
    current=runtime.current();save_motion(out/'current.npz',current)
    if request['mode']=='rollout':
        candidate=load_motion(request['candidate'])
        execution=runtime.rollout(candidate,request['prefix_frames'])
        save_motion(out/'continuation.npz',execution.continuation)
        raw=runtime.calls[-1]
        np.savez_compressed(out/'raw.npz',**{k:v for k,v in raw.items() if isinstance(v,np.ndarray)},
            actual_start_state=raw['start_state'])
    unchanged=digest(runtime.runner.alg.policy)==before
    if not unchanged:raise AssertionError('Frozen GMT parameters changed')
    result={'status':'COMPLETED','checkpoint_sha256':request['checkpoint_sha256'],
        'request_sha256':hashlib.sha256(args.request.read_bytes()).hexdigest(),
        'mode':request['mode'],'replayed_windows':replays,'gmt_parameters_unchanged':unchanged,
        'gmt_parameter_digest':before,
        'determinism':{'torch_deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'tf32':False,'physx_enhanced_determinism':True,
            'physical_properties_sha256':hashlib.sha256((out/'initial_physical_properties.npz').read_bytes()).hexdigest()},
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ('scripts/replay_physics_worker.py','reimplementation/isaac_prefix_adapter.py',
             'reimplementation/replay_prefix_adapter.py','reimplementation/physics_prefix.py')}}
    (out/'worker_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    runtime.close();print('PHYGILE_REPLAY_WORKER_COMPLETE',flush=True)


try:main()
except BaseException:
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
