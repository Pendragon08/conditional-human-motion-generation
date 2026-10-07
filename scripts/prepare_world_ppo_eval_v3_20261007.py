"""Map manifest IDs to the existing MotionLib file order; preserve v1/v2."""
import hashlib,json,py_compile
from r3_world_ssh_transport import ROOT,sync
old=ROOT/'scripts/r3_world_ppo_eval_isaac_v2_20261007.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='e1699208bdc37c84e7ccad52317db4f64f6b78cd3093bd749ca861d2872e0d5a'
s=old.read_text(encoding='utf-8')
s=s.replace(" reference_dir=Path(manifest['reference_dir'])", " reference_dir=Path(manifest['reference_dir'])\n manifest_files=[row['file'] for row in manifest['selected']]",1)
s=s.replace("  env.motion_ids[ids]=fixedref[ids];env.start_motion_times[ids]=0.", """  loaded_names=[Path(f).name for f in env.motion_lib._motion_data_load]
  assert len(loaded_names)==11 and len(set(loaded_names))==11 and set(loaded_names)==set(manifest_files)
  mapping=torch.as_tensor([loaded_names.index(name) for name in manifest_files],device=a.device)
  env.motion_ids[ids]=mapping[fixedref[ids]];env.start_motion_times[ids]=0.""",1)
s=s.replace(" assert assigned.all() and torch.equal(raw.motion_ids,fixedref)", " assert assigned.all()",1)
s=s.replace(" assert loaded==[row['file'] for row in manifest['selected']], 'Reference order must match fixed manifest'", """ assert len(loaded)==11 and len(set(loaded))==11 and set(loaded)==set(manifest_files)
 manifest_to_library=np.array([loaded.index(name) for name in manifest_files],dtype=np.int64)
 initial_library_id=manifest_to_library[reference]
 assert torch.equal(raw.motion_ids,torch.as_tensor(initial_library_id,device=a.device))
 print('EVAL_INIT_MANIFEST_LIBRARY_MAPPING_PASSED',json.dumps(manifest_to_library.tolist()),flush=True)""",1)
s=s.replace("'reference_file':loaded[r]", "'reference_file':manifest_files[r]")
s=s.replace("reward=rewards,reference=reference,domain=domain,direction=direction,", "reward=rewards,reference=reference,domain=domain,direction=direction,\n    initial_library_id=initial_library_id,manifest_to_library=manifest_to_library,\n    loaded_reference_files=np.array(loaded),manifest_reference_files=np.array(manifest_files),")
s=s.replace("'reference_manifest_sha256':sha(a.reference_manifest),'reference_maxerror':ref_error,", "'reference_manifest_sha256':sha(a.reference_manifest),'reference_maxerror':ref_error,\n    'manifest_reference_files':manifest_files,'loaded_reference_files':loaded,\n    'manifest_to_library':manifest_to_library.tolist(),'initial_assignment_filename_guard_passed':True,")
new=ROOT/'scripts/r3_world_ppo_eval_isaac_v3_20261007.py';assert not new.exists();new.write_text(s,encoding='utf-8',newline='\n');py_compile.compile(str(new),doraise=True)
stage=(ROOT/'scripts/r3_world_ppo_formal_stage_v2_20261007.sh').read_text()
stage=stage.replace('r3_world_ppo_eval_isaac_v2_20261007.py','r3_world_ppo_eval_isaac_v3_20261007.py').replace('eval_interface_v2','eval_interface_v3').replace('PREFIX="${PREFIX}_v2"','PREFIX="${PREFIX}_v3"')
p=ROOT/'scripts/r3_world_ppo_formal_stage_v3_20261007.sh';assert not p.exists();p.write_text(stage,encoding='utf-8',newline='\n')
wrapper=(ROOT/'scripts/r3_world_ppo_eval_interface_v2_20261007.sh').read_text().replace('eval_interface_v2','eval_interface_v3').replace('formal_stage_v2','formal_stage_v3')
p=ROOT/'scripts/r3_world_ppo_eval_interface_v3_20261007.sh';assert not p.exists();p.write_text(wrapper,encoding='utf-8',newline='\n')
launcher=(ROOT/'scripts/launch_world_ppo_eval_interface_v2_20261007.py').read_text().replace('eval_sources_v2','eval_sources_v3').replace('eval_interface_v2','eval_interface_v3').replace('formal_stage_v2','formal_stage_v3')
p=ROOT/'scripts/launch_world_ppo_eval_interface_v3_20261007.py';assert not p.exists();p.write_text(launcher,encoding='utf-8',newline='\n');py_compile.compile(str(p),doraise=True)
paths=['scripts/r3_world_ppo_eval_isaac_v3_20261007.py','scripts/r3_world_ppo_formal_stage_v3_20261007.sh',
 'scripts/r3_world_ppo_eval_interface_v3_20261007.sh','scripts/launch_world_ppo_eval_interface_v3_20261007.py']
locked=paths+['scripts/r3_world_ppo_core.py','scripts/r3_world_ppo_residual_core.py','scripts/r3_world_ppo_training_core_20261007.py','scripts/r3_world_ppo_control_formal_v3_20261007.py']
hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in locked}
(ROOT/'configs/world_ppo_eval_sources_v3_20261007.json').write_text(json.dumps(hashes,indent=2)+'\n')
spec=json.loads((ROOT/'configs/env-world-ppo-eval-interface-v2-20261007.json').read_text());spec.update({'source_hashes':hashes,
 'revision':'v3_manifest_to_library_filename_mapping_only','file_order_adaptation':'Initial cohort and rows use manifest ID; map to unmodified library order by exact filename',
 'original_invocations_preserved':['v1','v2'],'no_scientific_scores_produced_before_fix':True})
(ROOT/'configs/env-world-ppo-eval-interface-v3-20261007.json').write_text(json.dumps(spec,indent=2)+'\n')
print(sync(paths+['configs/world_ppo_eval_sources_v3_20261007.json','configs/env-world-ppo-eval-interface-v3-20261007.json']))
