"""New reference confirmation only. Never starts a formal PPO run."""
import fcntl,hashlib,json,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MON=ROOT/'artifacts/monitoring';ART=ROOT/'artifacts/world_confirmation_20261007'
STATE=MON/'r3_world_confirmation_20261007.state'

def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb') as s:
        for b in iter(lambda:s.read(2**20),b''):h.update(b)
    return h.hexdigest()

def save(s):
    s.update(updated_unix=time.time(),formal_PPO_started=False,auxiliary_updates=0,old_v1_FAIL_preserved=True)
    f=STATE.with_suffix('.tmp');f.write_text(json.dumps(s,indent=2)+'\n');f.replace(STATE)

def main():
    with (MON/'r3_world_confirmation_20261007.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not STATE.exists(),'Preserve prior dispatch'
        manifest=ART/'selection/manifest.json';m=json.loads(manifest.read_text())
        assert m['status']=='COMPLETE' and m['selected_count']==11 and not m['physical_results_used_for_selection']
        assert not (ART/'model_lock.json').exists()
        files={}
        for seed in [42,43,44]:
            fit=ROOT/('artifacts/world_residual_20261007/fit_seed%d_v1'%seed)
            r=json.loads((fit/'summary.json').read_text());assert r['development_triage']=='PROMISING_NEEDS_NEW_CONFIRMATION_REFS'
            for name in ['summary.json','direct.pt','residual_correct.pt','residual_state_only.pt','residual_shuffled.pt','ridge_correct.npz','ridge_state_only.npz']:
                f=fit/name;files[str(f)]=sha(f)
        for row in m['selected']:
            f=Path(m['reference_dir'])/row['file'];assert sha(f)==row['sha256'];files[str(f)]=row['sha256']
        sources=['scripts/r3_world_confirmation_eval_20261007.py','scripts/r3_world_ppo_isaac.py','scripts/r3_world_ppo_core.py',
                 'scripts/r3_world_ppo_residual_core.py','plans/world_residual_confirmation_20261007.md',
                 'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_for_eval.urdf']
        for name in sources:files[str(ROOT/name)]=sha(ROOT/name)
        files[str(manifest)]=sha(manifest)
        (ART/'model_lock.json').write_text(json.dumps({'files':files,'reference_manifest_sha256':sha(manifest),
            'schema_decoder_included':True,'created_unix_before_collection':time.time(),'normalizers_frozen_in_model_artifacts':True},indent=2)+'\n')
        completed=[];data=[]
        for seed in [42,43,44]:
            for mode in ['collect','evaluate']:
                if shutil.disk_usage(ROOT).free<15*1024**3:
                    save({'status':'BLOCKED','stage':mode,'seed':seed,'reason':'disk_under15GiB','completed':completed,'datasets':data});return 1
                save({'status':'RUNNING','stage':mode,'seed':seed,'completed':completed,'datasets':data})
                rc=subprocess.run(['bash',str(ROOT/'scripts/r3_world_confirmation_stage_20261007.sh'),mode,str(seed)]).returncode
                if rc:
                    save({'status':'FAILED','stage':mode,'seed':seed,'exit':rc,'preserve':True,'completed':completed,'datasets':data});return 1
                f=ART/('%s_seed%d_v1/summary.json'%(mode,seed));r=json.loads(f.read_text())
                if mode=='collect':
                    assert r['actual_transitions']==983040 and r['fresh_confirmation_references'] and r['diagnostic_ppo_updates']==0
                    data.append({'seed':seed,'summary_sha256':sha(f),'transitions':r['actual_transitions'],'resets':r['reset_count'],
                                 'postpush_reset_count':r['disturbance_protocol']['reset_60_to_109_pushed_envs']})
                else:
                    assert r['auxiliary_updates']==0 and r['PPO_updates']==0 and r['confirmation_evaluations_per_seed']==1
                    completed.append({'seed':seed,'summary_sha256':sha(f),'domain_gates':r['domain_gates'],
                                      'prediction_gate_passed':r['formal_prediction_gate_passed'],'comparisons':r['comparisons']})
        passed=all(r['prediction_gate_passed'] for r in completed)
        recovery=all(r['domain_gates']['recovery'] for r in completed)
        save({'status':'COMPLETE','stage':'confirmation_only','completed':completed,'datasets':data,
              'new_actual_transitions':sum(r['transitions'] for r in data),'prediction_gate_passed':passed,
              'recovery_domain_all_seed_gate':recovery,'innovation_validated':False,
              'next':'local_actual_Kick_view_and_matched_PPO' if passed else 'stop_current_candidate_no_repeat_confirmation'})
        print('WORLD_RESIDUAL_CONFIRMATION_QUEUE_COMPLETE','PASS' if passed else 'FAIL',flush=True)
        return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        save({'status':'FAILED','stage':'queue','error':repr(e),'preserve':True});raise
