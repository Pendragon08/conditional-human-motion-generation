"""Independent metadata-only preflight and observations for documented v2."""
from pathlib import Path
import argparse, base64, difflib, hashlib, json, shlex, subprocess, time

ROOT=Path(__file__).resolve().parents[3]
RAW=Path(__file__).resolve().parent
SSH=['ssh','-b','10.147.17.250','-i','C:/Users/Pende/.ssh/id_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=15','ouyanghao@10.147.17.32']
CODE=r'''
from pathlib import Path
import fcntl,hashlib,json,shutil,subprocess,time,difflib
r=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
m=r/'artifacts/monitoring';b=r/'artifacts/world_confirmation_20261007';a=b/'confirmation_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 return h.hexdigest()
def run(args):
 p=subprocess.run(args,capture_output=True,text=True);return {'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
oldlock=json.loads((b/'model_lock.json').read_text())
oldstate=json.loads((m/'r3_world_confirmation_20261007.state').read_text())
manifest=json.loads((b/'selection/manifest.json').read_text())
spec=json.loads((r/'configs/env-dreamcontrol-r3.json').read_text())
oldsrc=(r/'scripts/r3_world_ppo_isaac.py').read_text()
v2src=(r/'scripts/r3_world_confirmation_isaac_v2_20261007.py').read_text()
expected=oldsrc.replace("expected_q=np.concatenate([np.asarray(d['joints'])[:200] for d in lib._motion_data_list])", "expected_q=np.stack([np.asarray(d['joints'])[:200] for d in lib._motion_data_list])\n        assert expected_q.shape==tuple(lib.dof_pos.shape)==(11,200,27)")
lockfree=False
with (m/'r3_world_ppo_gpu0.lock').open('r+') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);lockfree=True;fcntl.flock(f,fcntl.LOCK_UN)
 except BlockingIOError:pass
paths=['scripts/r3_world_confirmation_isaac_v2_20261007.py','scripts/r3_world_confirmation_v2_queue_20261007.py','scripts/r3_world_confirmation_v2_stage_20261007.sh','scripts/r3_world_ppo_isaac.py','scripts/r3_world_confirmation_eval_20261007.py','plans/world_residual_confirmation_20261007.md','docs/world_confirmation_isaac_v2_invocation_20261007.md']
refs=[{'file':row['file'],'actual_sha256':sha(Path(manifest['reference_dir'])/row['file']),'expected_sha256':row['sha256'],'bytes_match':(Path(manifest['reference_dir'])/row['file']).stat().st_size==row['bytes']} for row in manifest['selected']]
obs={'observed_unix':time.time(),'session':run(['tmux','has-session','-t','r3_world_confirmation_v2_20261007']),
 'v2_state_exists':(m/'r3_world_confirmation_v2_20261007.state').exists(),'v2_art_exists':a.exists(),
 'v2_stage_state_names':[p.name for p in m.glob('r3_world_confirmation_v2_*_v1.state')],
 'old_queue_state':oldstate,'old_collection_files':[p.name for p in (b/'collect_seed42_v1').glob('*')],
 'old_evaluate_exists':(b/'evaluate_seed42_v1').exists(),'old_lock_file_count':len(oldlock['files']),
 'old_lock_hash_mismatches':[p for p,h in oldlock['files'].items() if sha(p)!=h],
 'old_preserved_hashes':{str(p.relative_to(r)):sha(p) for p in [b/'model_lock.json',m/'r3_world_confirmation_20261007.state',m/'r3_world_confirmation_collect_42_v1.log',m/'r3_world_confirmation_collect_42_v1.state',r/'scripts/r3_world_ppo_isaac.py',r/'scripts/r3_world_confirmation_queue_20261007.py',r/'scripts/r3_world_confirmation_stage_20261007.sh']},
 'gpu0_lock_free':lockfree,'disk_free_GiB':shutil.disk_usage(r).free/1024**3,
 'gpu':run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader']),
 'gpu_processes':run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),
 'processes':run(['ps','-eo','pid,ppid,etimes,args']),
 'canonical_spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'source_hashes':{p:sha(r/p) for p in paths},'normalized_collector_only_exact_guard_change':expected==v2src,
 'normalized_collector_diff':list(difflib.unified_diff(oldsrc.splitlines(),v2src.splitlines(),fromfile='original',tofile='v2')),
 'original_66_branch_unchanged':"official=np.load(ROOT/'artifacts/sim2sim_mujoco/interfaces/Kick_references.npz')\n        identity=float(np.max(np.abs(lib.dof_pos.cpu().numpy()-official['dof_pos'])))" in v2src,
 'selected_reference_count':len(refs),'selected_reference_hashes':refs,'manifest_sha256':sha(b/'selection/manifest.json'),
 'bash_stage_syntax':run(['bash','-n',str(r/'scripts/r3_world_confirmation_v2_stage_20261007.sh')]),
 'stage_no_CR':b'\r' not in (r/'scripts/r3_world_confirmation_v2_stage_20261007.sh').read_bytes(),
 'remote_AGENTS':{str(p):p.read_text() for p in [r/'AGENTS.md',r.parent/'AGENTS.md'] if p.exists()}}
obs['processes']['stdout']='\n'.join(line for line in obs['processes']['stdout'].splitlines() if any(x in line for x in ['r3_world_confirmation','r3_world_ppo_isaac.py','r3_i1_live_training_viewer.py','play_eval.py']) and 'base64' not in line)
print(json.dumps(obs))
'''

OBSERVE=r'''
from pathlib import Path
import base64,hashlib,json,subprocess,time
r=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation');m=r/'artifacts/monitoring';b=r/'artifacts/world_confirmation_20261007';a=b/'confirmation_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(x):
 p=subprocess.run(x,capture_output=True,text=True);return {'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
paths={'queue.state':m/'r3_world_confirmation_v2_20261007.state','model_lock.json':a/'model_lock.json'}
for seed in [42,43,44]:
 for mode in ['collect','evaluate']:
  paths[f'{mode}_seed{seed}_summary.json']=a/f'{mode}_seed{seed}_v1/summary.json'
  paths[f'{mode}_seed{seed}.state']=m/f'r3_world_confirmation_v2_{mode}_{seed}_v1.state'
  paths[f'{mode}_seed{seed}_coverage.json']=a/f'{mode}_seed{seed}_v1/coverage.json'
logs=[m/'r3_world_confirmation_v2_20261007.log']+[m/f'r3_world_confirmation_v2_{mode}_{seed}_v1.log' for seed in [42,43,44] for mode in ['collect','evaluate']]
tail={}
for p in logs:
 if p.exists():
  text=p.read_text(errors='replace');lines=text.splitlines()
  interesting=[s for s in lines if any(x in s for x in ['WORLD_','Traceback','Error','Exception','Timeout ('])]
  tail[p.name]={'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'sha256':sha(p),'sentinels':{s:s in text for s in ['WORLD_PPO_ISAAC_COMPLETE','WORLD_RESIDUAL_NEW_REFERENCE_CONFIRMATION_COMPLETE','WORLD_RESIDUAL_CONFIRMATION_QUEUE_COMPLETE']},'interesting':[(s[:500] if s.startswith('WORLD_PPO_ISAAC_COMPLETE') else s) for s in interesting[-8:]],'tail':text[-2200:] if 'Traceback' in text or 'Timeout (' in text else '\n'.join(lines[-4:])[:1200]}
processes=run(['ps','-eo','pid,ppid,etimes,args'])
processes['stdout']='\n'.join(s for s in processes['stdout'].splitlines() if any(x in s for x in ['r3_world_confirmation','r3_i1_live_training_viewer.py','play_eval.py']) and 'base64' not in s)
out={'observed_unix':time.time(),'session':run(['tmux','has-session','-t','r3_world_confirmation_v2_20261007']), 'receipts':{k:base64.b64encode(p.read_bytes()).decode() for k,p in paths.items() if p.exists()},'log_observation':tail,'processes':processes,'gpu':run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader']),'progress':{str(seed):json.loads((a/f'collect_seed{seed}_v1/progress.json').read_text()) for seed in [42,43,44] if (a/f'collect_seed{seed}_v1/progress.json').exists()}}
if (a/'model_lock.json').exists():
 lock=json.loads((a/'model_lock.json').read_text());old=json.loads((b/'model_lock.json').read_text())
 out['new_lock']={'sha256':sha(a/'model_lock.json'),'file_count':len(lock['files']),'hash_mismatches':[p for p,h in lock['files'].items() if sha(p)!=h],'same_38_common_hashes_as_old':{p:h for p,h in lock['files'].items() if p in old['files']}=={p:h for p,h in old['files'].items() if p in lock['files']},'common_file_count':len(set(old['files'])&set(lock['files'])),'created_unix_before_collection':lock['created_unix_before_collection'],'reference_manifest_sha256':lock['reference_manifest_sha256']}
print(json.dumps(out))
'''

def remote(code):
 encoded=base64.b64encode(code.encode()).decode()
 cmd='/usr/bin/python3 -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')
 p=subprocess.run(SSH+[cmd],capture_output=True,text=True,timeout=120)
 if p.returncode:raise RuntimeError(p.stderr)
 return json.loads(p.stdout)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','observe','final']);args=p.parse_args()
 if args.mode=='preflight':
  obs=remote(CODE)
  local={x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in obs['source_hashes'] if (ROOT/x).exists()}
  obs['local_source_hashes']=local
  obs['local_remote_hashes_match']=local==obs['source_hashes']
  expected={'scripts/r3_world_confirmation_isaac_v2_20261007.py':'bf594e968b07d402c000e3c3194aa45c43997332dba0653cec87e954557ad030','scripts/r3_world_confirmation_v2_queue_20261007.py':'03791432f0765da982438658d0e608f5fba4da2f2bd582837c34b31470d7801b','scripts/r3_world_confirmation_v2_stage_20261007.sh':'92f2c0cf13d29e85031c43fd0c6fdfb3ca5b52d0dbcd86e944582e64cc3e719b','docs/world_confirmation_isaac_v2_invocation_20261007.md':'d72301ebcdb46932b8f12a953b3b1a367e77ae737594811bbd7e933944d5c3d5'}
  checks={'no_existing_v2_session':obs['session']['rc']==1,'no_existing_v2_outputs':not obs['v2_state_exists'] and not obs['v2_art_exists'] and not obs['v2_stage_state_names'],'old_FAILED_empty':obs['old_queue_state']['status']=='FAILED' and not obs['old_queue_state']['completed'] and not obs['old_queue_state']['datasets'] and not obs['old_collection_files'] and not obs['old_evaluate_exists'],'old_39_locked_files_unchanged':obs['old_lock_file_count']==39 and not obs['old_lock_hash_mismatches'],'fixed_11_refs':obs['selected_reference_count']==11 and all(x['actual_sha256']==x['expected_sha256'] and x['bytes_match'] for x in obs['selected_reference_hashes']),'gpu0_lock_free':obs['gpu0_lock_free'],'disk_gt15GiB':obs['disk_free_GiB']>15,'warm_spec':obs['canonical_spec_sha256']=='47f5eb26218254bbe270c3c43b3e7a552cca04c5caa81b6c76b4a6d2d3d2e9d4','expected_v2_hashes':all(obs['source_hashes'][x]==h for x,h in expected.items()),'only_exact_guard_change':obs['normalized_collector_only_exact_guard_change'] and obs['original_66_branch_unchanged'],'stage_LF_syntax':obs['stage_no_CR'] and obs['bash_stage_syntax']['rc']==0,'old_failed_child_gone':'1612527' not in obs['processes']['stdout']}
  obs['checks']=checks;obs['all_checks_passed']=all(checks.values())
  path=RAW/'identity_v2_preflight_doc_witness.json'
  assert not path.exists(),'Preserve prior independent preflight'
  path.write_text(json.dumps(obs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  print(json.dumps({'checks':checks,'gpu':obs['gpu'],'gpu_processes':obs['gpu_processes'],'processes':obs['processes'],'disk_free_GiB':obs['disk_free_GiB'],'local_remote_hashes_match':obs['local_remote_hashes_match'],'diff':obs['normalized_collector_diff']},ensure_ascii=False,indent=2))
  if not obs['all_checks_passed']:raise SystemExit(1)
 elif args.mode=='observe':
  obs=remote(OBSERVE)
  stamp=str(int(obs['observed_unix']*1000));dest=RAW/'identity_v2_metadata';dest.mkdir(exist_ok=True)
  for name,encoded in obs['receipts'].items():
   (dest/name).write_bytes(base64.b64decode(encoded))
  obs['receipt_names']=list(obs.pop('receipts'))
  (dest/f'observation_{stamp}.json').write_text(json.dumps(obs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  state=json.loads((dest/'queue.state').read_text()) if (dest/'queue.state').exists() else {}
  compact=dict(state);compact['completed']=[{k:v for k,v in x.items() if k!='comparisons'} for x in state.get('completed',[])]
  print(json.dumps({'state':compact,'session':obs['session'],'progress':obs['progress'],'new_lock':obs.get('new_lock'),'gpu':obs['gpu'],'logs':{k:{'bytes':v['bytes'],'sentinels':v['sentinels'],'interesting':v['interesting']} for k,v in obs['log_observation'].items()},'receipt_names':obs['receipt_names']},ensure_ascii=False,indent=2))
 else:
  obs=remote(CODE);before=json.loads((RAW/'identity_v2_preflight_doc_witness.json').read_text())
  obs['old_sources_states_logs_and_lock_exactly_preserved']=before['old_preserved_hashes']==obs['old_preserved_hashes']
  obs['all_seven_source_doc_plan_hashes_unchanged']=before['source_hashes']==obs['source_hashes']
  obs['all_selected_references_unchanged']=before['selected_reference_hashes']==obs['selected_reference_hashes']
  (RAW/'identity_v2_final_readonly.json').write_text(json.dumps(obs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  print(json.dumps({k:obs[k] for k in ['old_sources_states_logs_and_lock_exactly_preserved','all_seven_source_doc_plan_hashes_unchanged','all_selected_references_unchanged','gpu0_lock_free','disk_free_GiB','processes','gpu']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
