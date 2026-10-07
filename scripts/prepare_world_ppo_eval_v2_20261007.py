"""Preserve v1 and expose pre-loop exceptions; scientific evaluator unchanged."""
import hashlib,json,py_compile
from r3_world_ssh_transport import ROOT,sync
old=ROOT/'scripts/r3_world_ppo_eval_isaac_20261007.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='7f034b9af36b1063aab9026d430f91d2ad8c522943c726f32d9083c53070c117'
text=old.read_text(encoding='utf-8')
text=text.replace("ROOT=Path(__file__).resolve().parents[1]", "RUNNING_ENV=None\nROOT=Path(__file__).resolve().parents[1]",1)
text=text.replace('def main():\n', 'def main():\n global RUNNING_ENV\n',1)
text=text.replace(" raw=env.unwrapped;robot=raw.scene['robot'];lib=raw.motion_lib", " RUNNING_ENV=env\n print('EVAL_INIT_ENV_CREATED',flush=True)\n raw=env.unwrapped;robot=raw.scene['robot'];lib=raw.motion_lib",1)
text=text.replace(" obs,_=env.reset(seed=a.seed);obs=obs['policy']", " print('EVAL_INIT_BEFORE_RESET',flush=True)\n obs,_=env.reset(seed=a.seed);obs=obs['policy']\n print('EVAL_INIT_RESET_RETURNED',flush=True)",1)
text=text.replace(" loaded=[Path(f).name for f in lib._motion_data_load]", " print('EVAL_INIT_ALLOCATION_PASSED',flush=True)\n loaded=[Path(f).name for f in lib._motion_data_load]",1)
text=text.replace(" assert loaded==[row['file'] for row in manifest['selected']], 'Reference order must match fixed manifest'", " print('EVAL_INIT_REFERENCE_ORDER',json.dumps({'loaded':loaded,'manifest':[row['file'] for row in manifest['selected']]}),flush=True)\n assert loaded==[row['file'] for row in manifest['selected']], 'Reference order must match fixed manifest'",1)
text=text.replace("if __name__=='__main__':\n try:main()\n finally:app.close()", """if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  print('EVAL_ORIGINAL_EXCEPTION_BEFORE_CLEANUP',flush=True)
  traceback.print_exc();sys.stderr.flush()
  if RUNNING_ENV is not None:
   try:RUNNING_ENV.close()
   except BaseException:traceback.print_exc();sys.stderr.flush()
  raise
 finally:app.close()""",1)
new=ROOT/'scripts/r3_world_ppo_eval_isaac_v2_20261007.py';assert not new.exists()
new.write_text(text,encoding='utf-8',newline='\n');py_compile.compile(str(new),doraise=True)
stage=(ROOT/'scripts/r3_world_ppo_formal_stage_20261007.sh').read_text()
stage=stage.replace('r3_world_ppo_eval_isaac_20261007.py','r3_world_ppo_eval_isaac_v2_20261007.py')
stage=stage.replace('eval_interface"; fi','eval_interface_v2"; fi')
stage=stage.replace('OUT="$BASE/${MODE}_${ARM}_seed${SEED}_$VERSION"', 'if [[ "$MODE" == eval-diagnostic ]]; then PREFIX="${PREFIX}_v2"; fi\nOUT="$BASE/${MODE}_${ARM}_seed${SEED}_$VERSION"')
stage=stage.replace('STEPS=120;EVALMODE=diagnostic','STEPS=120;EVALMODE=diagnostic')
# Bounds of diagnostic infrastructure only; original formal limit remains1800s.
stage=stage.replace('timeout --signal=TERM --kill-after=30s 1800', 'LIMIT=1800; if [[ "$MODE" == eval-diagnostic ]]; then LIMIT=300; fi\n timeout --signal=TERM --kill-after=30s "$LIMIT"')
p=ROOT/'scripts/r3_world_ppo_formal_stage_v2_20261007.sh';assert not p.exists();p.write_text(stage,encoding='utf-8',newline='\n')
wrapper=(ROOT/'scripts/r3_world_ppo_eval_interface_20261007.sh').read_text()
wrapper=wrapper.replace('r3_world_ppo_eval_interface_20261007','r3_world_ppo_eval_interface_v2_20261007').replace('r3_world_ppo_formal_stage_20261007.sh','r3_world_ppo_formal_stage_v2_20261007.sh')
p=ROOT/'scripts/r3_world_ppo_eval_interface_v2_20261007.sh';assert not p.exists();p.write_text(wrapper,encoding='utf-8',newline='\n')
launcher=(ROOT/'scripts/launch_world_ppo_eval_interface_20261007.py').read_text()
launcher=launcher.replace('world_ppo_eval_sources_20261007.json','world_ppo_eval_sources_v2_20261007.json').replace('r3_world_ppo_eval_interface_20261007','r3_world_ppo_eval_interface_v2_20261007').replace('eval_interface\'','eval_interface_v2\'').replace('r3_world_ppo_formal_stage_20261007.sh','r3_world_ppo_formal_stage_v2_20261007.sh')
p=ROOT/'scripts/launch_world_ppo_eval_interface_v2_20261007.py';assert not p.exists();p.write_text(launcher,encoding='utf-8',newline='\n');py_compile.compile(str(p),doraise=True)
paths=['scripts/r3_world_ppo_eval_isaac_v2_20261007.py','scripts/r3_world_ppo_formal_stage_v2_20261007.sh',
 'scripts/r3_world_ppo_eval_interface_v2_20261007.sh','scripts/launch_world_ppo_eval_interface_v2_20261007.py']
locked=paths+['scripts/r3_world_ppo_core.py','scripts/r3_world_ppo_residual_core.py','scripts/r3_world_ppo_training_core_20261007.py','scripts/r3_world_ppo_control_formal_v3_20261007.py']
hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in locked}
(ROOT/'configs/world_ppo_eval_sources_v2_20261007.json').write_text(json.dumps(hashes,indent=2)+'\n')
spec=json.loads((ROOT/'configs/env-world-ppo-eval-interface-20261007.json').read_text());spec.update({'source_hashes':hashes,
 'revision':'v2_error_reporting_and_cleanup_only','diagnostic_wall_limit_s':300,'original_invocation_preserved':True})
(ROOT/'configs/env-world-ppo-eval-interface-v2-20261007.json').write_text(json.dumps(spec,indent=2)+'\n')
print(sync(paths+['configs/world_ppo_eval_sources_v2_20261007.json','configs/env-world-ppo-eval-interface-v2-20261007.json']))
