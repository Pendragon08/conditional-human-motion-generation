"""Actual diffusion/262D/Unitree MuJoCo execution with transactional feedback."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--unitree-checkout',type=Path,required=True)
p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--contact-profile',choices=['official','matched_sliding_friction'],default='matched_sliding_friction');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled')
sys.path.insert(0,str(a.unitree_checkout));sys.path.insert(0,str(a.parent/'reimplementation'))
import numpy as np
import torch
import mjlab,warp as wp
from codec262 import G1Codec
from physics_prefix import Motion,PrefixRejected,roll_physics_prefix
from diffusion_proposer import DiffusionProposer
from terminal_reference import nominal_standing
from mjlab_prefix_adapter import TransactionalMjlab
from source_provenance import capture,verify
torch.set_num_threads(4);torch.manual_seed(2026);wp.init()
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def digest(module):
 h=hashlib.sha256()
 for n,v in module.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
codec=G1Codec(a.parent/'configs/g1_codec_skeleton.json');checkpoint=a.assets/'parent_checkpoint.pt'
original=sha(checkpoint);started=time.monotonic()
runtime=TransactionalMjlab(a.assets,checkpoint,codec,2026,a.output/'simulator',a.contact_profile)
prefix=runtime.current();snapshot=runtime.snapshot()
provenance=capture(a.output/'executed_sources.json',[Path(__file__).resolve().parents[1],a.parent,a.unitree_checkout,Path(mjlab.__file__).parent],
 [__file__,a.parent/'configs/g1_codec_skeleton.json',a.parent/'configs/nominal_standing_reference.json',a.assets/'normalization.npz',a.assets/'contract.json',a.assets/'encoder_actor_critic.py',checkpoint])
hold=prefix.append(Motion(np.repeat(prefix.states,10,0),np.repeat(prefix.body_positions,10,0),prefix.fps,prefix.state_schema,prefix.body_names))
first=runtime.rollout(hold,len(prefix));runtime.restore(snapshot);second=runtime.rollout(hold,len(prefix));runtime.restore(snapshot)
if len(first.continuation)!=len(second.continuation):raise AssertionError('Rollback horizon changed')
state_delta=float(np.abs(first.continuation.states-second.continuation.states).max())
body_delta=float(np.abs(first.continuation.body_positions-second.continuation.body_positions).max())
if max(state_delta,body_delta)>2e-6:raise AssertionError('Fresh-scene replay changed actual trajectory')
inputs=[]
def held(context,terminal,count,final,attempt):
 inputs.append(context.digest())
 return context.append(Motion(np.repeat(prefix.states,count,0),np.repeat(prefix.body_positions,count,0),context.fps,context.state_schema,context.body_names))
feedback,decisions=roll_physics_prefix(prefix,prefix,held,runtime,total_new_frames=2,chunk_frames=1,max_attempts=1,tolerance_m=.1,terminal_tolerance_m=.1)
assert all(x['accepted'] for x in decisions) and len(decisions)==2 and inputs[0]!=inputs[1]
assert len(runtime.invocations[-1]['replayed_windows'])==1
runtime.restore(snapshot)
suffix,suffix_config=nominal_standing(codec,prefix,a.parent/'configs/nominal_standing_reference.json')
proposer=DiffusionProposer(codec,a.assets/'normalization.npz','cpu',2026,a.output/'proposals')
before=digest(proposer.model)
try:
 committed,attempts=roll_physics_prefix(prefix,suffix,proposer,runtime,total_new_frames=50,chunk_frames=50,max_attempts=2,tolerance_m=.1,terminal_tolerance_m=.1)
 outcome='COMPLETED_ALL_WINDOWS'
except PrefixRejected as e:committed=e.prefix;attempts=e.attempts;outcome='REJECTED_WITHOUT_INVALID_COMMIT'
assert before==digest(proposer.model) and original==sha(checkpoint)
np.testing.assert_allclose(runtime.current().states[-1],committed.states[-1],atol=2e-6,rtol=0)
for i,raw in enumerate(runtime.calls):np.savez_compressed(a.output/('physics_rollout_'+str(i).zfill(3)+'.npz'),**raw)
np.savez_compressed(a.output/'committed_prefix.npz',states=committed.states,body_positions=committed.body_positions)
result={'status':'COMPLETED_ENGINEERING_VALIDATION','outcome':outcome,'actual_mujoco_warp_physics':True,
 'official_unitree_factory':'src.tasks.tracking.config.g1.env_cfgs.unitree_g1_flat_tracking_env_cfg',
 'contact_profile':a.contact_profile,'evaluation_type':'simulation_only','optimizer_updates':0,
 'actual_diffusion_sampling':True,'generator':proposer.provenance,'trained_generator':False,'trained_semantic_generation_claim':False,
 'checkpoint_sha256':original,'checkpoint_unchanged':True,'generator_parameters_unchanged':True,
 'committed_new_frames':len(committed)-len(prefix),'attempts':attempts,'proposals':proposer.calls,
 'rollback_witness':{'method':'fresh_scene_seeded_accepted_history_replay','state_max_abs_delta':state_delta,'body_max_abs_delta_m':body_delta,'frames':len(first.continuation)},
 'actual_feedback_fixture':{'evaluation_type':'self_supervised_proxy','reference':'held initialization pose, not learned generation or dataset truth','accepted_windows':2,'window_frames':1,'input_prefix_sha256':inputs,'decisions':decisions},
 'simulator_invocations':runtime.invocations,'source_provenance':provenance,'executed_sources_unchanged':verify(provenance),
 'elapsed_seconds':time.monotonic()-started,'raw_files':{str(f.relative_to(a.output)):sha(f) for f in a.output.rglob('*.npz')}}
(a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));runtime.close()
