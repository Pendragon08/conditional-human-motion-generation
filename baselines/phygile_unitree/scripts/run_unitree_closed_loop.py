"""Actual diffusion -> 262D decode -> G1 GMT -> pre-reset feedback/rollback.

Without --generator_checkpoint the generator is explicitly UNTRAINED and
unconditional. Success is never required or fabricated for this wiring check.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path(os.environ.get('PHYGILE_PARENT','/data/code/ouyanghao/oyh/phygile-reproduction-20261017'))
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--motion_file',type=Path,required=True)
p.add_argument('--normalization',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--generator_checkpoint',type=Path);p.add_argument('--text_encoder',type=Path)
p.add_argument('--text');p.add_argument('--seed',type=int,default=2026)
p.add_argument('--total_frames',type=int,default=100);p.add_argument('--attempts',type=int,default=2)
p.add_argument('--tolerance_m',type=float,default=.1)
p.add_argument('--terminal_reference',type=Path,default=PARENT/'configs/nominal_standing_reference.json')
p.add_argument('--device',default='cuda:0');p.add_argument('--headless',action='store_true')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
os.environ['G1_TRAINING_CFG_PATH']=str(PARENT/'configs/moe_pilot.yaml')
sys.path.insert(0,str(PARENT/'scripts'));sys.path.insert(0,str(PARENT/'reimplementation'));sys.path.insert(0,str(ROOT/'scripts'))
from guarded_bootstrap import activate
guards=activate()
import numpy as np
import torch
from codec262 import G1Codec
from unitree_replay_adapter import UnitreeReplayPrefixAdapter as ReplayPrefixAdapter
from diffusion_proposer import DiffusionProposer
from physics_prefix import Motion,PrefixRejected,roll_physics_prefix
from terminal_reference import nominal_standing


def parameter_digest(module):
 h=hashlib.sha256()
 for name,value in module.state_dict().items():
  h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def main():
 from source_provenance import capture,verify
 codec=G1Codec(PARENT/'configs/g1_codec_skeleton.json')
 provenance=capture(a.output/'executed_sources.json',[ROOT,PARENT],[__file__,ROOT/'scripts/run_isaac.sh',PARENT/'configs/g1_codec_skeleton.json',a.normalization])
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20,torch.device(a.device))
 torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 runtime=ReplayPrefixAdapter(ROOT,a.checkpoint,a.motion_file,a.device,a.seed,a.output/'simulator')
 print('CLOSED_LOOP_STAGE runtime_ready',flush=True)
 prefix=runtime.current();fixture_suffix=Motion(prefix.states.copy(),prefix.body_positions.copy(),
     prefix.fps,prefix.state_schema,prefix.body_names)
 suffix,terminal_config=nominal_standing(codec,prefix,a.terminal_reference)
 np.savez_compressed(a.output/'nominal_standing_target.npz',states=suffix.states,
                     body_positions=suffix.body_positions)
 # Independently exercise repeated actual physics/control execution after state
 # restoration before allowing transactional proposal retries.
 checkpoint=runtime.snapshot()
 print('CLOSED_LOOP_STAGE snapshot_ready',flush=True)
 hold=prefix.append(Motion(np.repeat(prefix.states,10,0),np.repeat(prefix.body_positions,10,0),
     prefix.fps,prefix.state_schema,prefix.body_names))
 runtime.restore(checkpoint);first=runtime.rollout(hold,len(prefix))
 print('CLOSED_LOOP_STAGE first_rollback_rollout',flush=True)
 runtime.restore(checkpoint);second=runtime.rollout(hold,len(prefix))
 print('CLOSED_LOOP_STAGE second_rollback_rollout',flush=True)
 if len(first.continuation)!=len(second.continuation):raise AssertionError('Rollback changed termination horizon')
 state_delta=float(np.max(np.abs(first.continuation.states-second.continuation.states)))
 body_delta=float(np.max(np.abs(first.continuation.body_positions-second.continuation.body_positions)))
 np.savez_compressed(a.output/'rollback_development_evidence.npz',
     first_states=first.continuation.states,second_states=second.continuation.states,
     first_actions=runtime.calls[0]['actions'],second_actions=runtime.calls[1]['actions'])
 if max(state_delta,body_delta)>5e-4:raise AssertionError(('Exposed-state rollback witness failed',state_delta,body_delta))
 runtime.restore(checkpoint)
 witness={'repeated_rollout_frames':len(first.continuation),'state_max_abs_delta':state_delta,
     'body_max_abs_delta_m':body_delta,'tolerance':5e-4,
     'scope':'same seeded scene replayed in a fresh process; committed histories checked before continuation',
     'method':'fresh_process_replay; no private PhysX state serialization claim'}
 # Two one-frame held-pose windows exercise actual accept -> feedback -> new
 # proposal -> replay. This synthetic initialization fixture is NOT learned
 # motion generation or a tracking benchmark; its targets are labeled proxies.
 feedback_inputs=[]
 def held_pose(context,terminal,count,final,attempt):
  feedback_inputs.append(context.digest())
  return context.append(Motion(np.repeat(fixture_suffix.states,count,0),
      np.repeat(fixture_suffix.body_positions,count,0),context.fps,context.state_schema,context.body_names))
 feedback_prefix,feedback_decisions=roll_physics_prefix(prefix,fixture_suffix,held_pose,runtime,
     total_new_frames=2,chunk_frames=1,max_attempts=1,tolerance_m=.1,terminal_tolerance_m=.1)
 assert len(feedback_decisions)==2 and all(x['accepted'] for x in feedback_decisions)
 assert feedback_inputs[1]!=feedback_inputs[0] and len(feedback_prefix)==3
 replay_proof=runtime.invocations[-1]['result']['replayed_windows']
 assert len(replay_proof)==1,'Second accepted fixture window did not replay its history'
 feedback_witness={'evaluation_type':'self_supervised_proxy',
     'reference':'held initialization pose; not dataset truth or generated semantics',
     'control_frames':2,'window_frames':1,'accepted_windows':2,
     'input_prefix_sha256':feedback_inputs,'committed_prefix_sha256':feedback_prefix.digest(),
     'replayed_history':replay_proof,'decisions':feedback_decisions}
 save_prefix=a.output/'feedback_fixture_committed.npz'
 np.savez_compressed(save_prefix,states=feedback_prefix.states,body_positions=feedback_prefix.body_positions)
 runtime.restore(checkpoint)
 proposer=DiffusionProposer(codec,a.normalization,a.device,a.seed,a.output/'proposals',
     checkpoint=a.generator_checkpoint,text_encoder=a.text_encoder,text=a.text)
 print('CLOSED_LOOP_STAGE proposer_ready',flush=True)
 generator_digest=parameter_digest(proposer.model)
 started=time.time();initial_digest=prefix.digest()
 try:
  committed,attempts=roll_physics_prefix(prefix,suffix,proposer,runtime,
     total_new_frames=a.total_frames,chunk_frames=50,max_attempts=a.attempts,
     tolerance_m=a.tolerance_m,terminal_tolerance_m=a.tolerance_m)
  outcome='COMPLETED_ALL_WINDOWS'
 except PrefixRejected as error:
  committed=error.prefix;attempts=error.attempts;outcome='REJECTED_WITHOUT_INVALID_COMMIT'
 assert generator_digest==parameter_digest(proposer.model),'Frozen generator changed'
 # The adapter's current state must match the final committed boundary even
 # when the last proposed window exhausted retries.
 np.testing.assert_allclose(runtime.current().states[-1],committed.states[-1],atol=2e-6,rtol=0)
 for i,record in enumerate(runtime.calls):
  np.savez_compressed(a.output/f'physics_rollout_{i:03d}.npz',**{
      k:v for k,v in record.items() if isinstance(v,np.ndarray)})
 np.savez_compressed(a.output/'committed_prefix.npz',states=committed.states,
                     body_positions=committed.body_positions)
 result={'status':'COMPLETED_ENGINEERING_VALIDATION','outcome':outcome,
   'evaluation_type':'simulation_only','evaluation_types':['synthetic_proxy','simulation_only'],'trained_generator':proposer.trained,
   'generator':proposer.provenance,'trained_semantic_generation_claim':False,
   'task':'Unitree-G1-PhyGile-v0','actual_diffusion_sampling':True,'actual_isaac_physics':True,'automatic_reset_during_candidates':False,
   'physics_prefix_ppo_adaptation_updates':0,'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
   'initialization_motion_sha256':hashlib.sha256(a.motion_file.read_bytes()).hexdigest(),
   'seed':a.seed,'chunk_frames':50,'requested_new_frames':a.total_frames,
   'initial_prefix_sha256':initial_digest,'committed_new_frames':len(committed)-len(prefix),
   'committed_prefix_sha256':committed.digest(),'accepted_attempts':sum(x['accepted'] for x in attempts),
   'rejected_attempts':sum(not x['accepted'] for x in attempts),'attempts':attempts,
   'proposals':proposer.calls,'rollback_witness':witness,
   'actual_feedback_fixture':feedback_witness,
   'gmt_parameters_unchanged':True,'generator_parameters_unchanged':True,
   'generator_parameter_digest':generator_digest,
   'simulator_invocations':runtime.invocations,'rollback_method':'fresh_process_accepted_history_replay',
   'terminal_target_source':terminal_config['source']+'; explicit nominal G1 standing, not author-verified suffix',
   'terminal_target_sha256':suffix.digest(),
   'terminal_reference_config_sha256':hashlib.sha256(a.terminal_reference.read_bytes()).hexdigest(),
   'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
   'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
   'memory_scope':'PyTorch allocator only; excludes PhysX/render allocations and other processes',
   'elapsed_seconds':time.time()-started,
   'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
      for folder in ('reimplementation','scripts','configs') for f in (ROOT/folder).glob('*')
      if f.suffix in ('.py','.json','.yaml','.sh')},
   'source_provenance':provenance,'executed_sources_unchanged':verify(provenance),
   'raw_files':{str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*.npz')}}
 (a.output/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print('PHYGILE_GENERATED_CLOSED_LOOP_COMPLETE',json.dumps({k:v for k,v in result.items()
    if k not in ('attempts','proposals','source_sha256','raw_files')}),flush=True)
 runtime.close()


try:
 main()
except BaseException:
 # Kit may hang in its stop callback after a simulator error. Preserve the
 # original exception and a nonzero exit rather than hiding it in shutdown.
 import traceback
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
 os._exit(1)
else:
 pass
