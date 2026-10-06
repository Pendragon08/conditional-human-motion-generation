"""A bounded physics-oracle upper bound, never a learned world-model result."""
import argparse
import json
import time
from pathlib import Path
import mujoco
import numpy as np
import torch
from r3_sim2sim_mujoco import Transfer, rotation, sha

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs/feedback_world_oracle_20261006.json'
OUT=ROOT/'results/feedback_world_model_20261006'
FLAGS=mujoco.mjtState.mjSTATE_INTEGRATION


def physical_state(t):
    s=np.empty(mujoco.mj_stateSize(t.model,FLAGS))
    mujoco.mj_getState(t.model,t.data,s,FLAGS)
    return s


def clone(t,s=None,last=None):
    q=object.__new__(Transfer)
    q.__dict__=t.__dict__.copy()
    q.data=mujoco.MjData(t.model)
    q.targets=t.targets.copy()
    q.last=t.last.copy() if last is None else last.copy()
    mujoco.mj_setState(t.model,q.data,physical_state(t) if s is None else s,FLAGS)
    mujoco.mj_forward(t.model,q.data)
    return q


def candidate_cost(t,i,k,offset,c):
    q=clone(t)
    total=0.0
    for h in range(c['horizon']):
        q.advance(q.action(q.obs(i,(k+h)*.02))+offset)
        ref=q.ref.state(i,(k+h+1)*.02)
        total+=float(np.mean((q.data.qpos[q.qadr]-ref['joints'])**2)
            +np.mean((q.data.qpos[:3]-ref['pos'])**2)
            +.1*np.mean((rotation(q.data.qpos[3:7])[:,2]-rotation(ref['quat'])[:,2])**2))
        if q.fallen():
            total+=4.0*(c['horizon']-h)
            break
    return total/c['horizon']+.05*float(np.mean(offset[:27]**2))


def choose(t,i,k,seed,c):
    rng=np.random.default_rng(seed+i*1009+k*100003)
    a=np.zeros((5,29),dtype=np.float32)
    a[1:3,:27]=np.clip(rng.normal(0,c['candidate_std'],(2,27)),-c['candidate_clip'],c['candidate_clip'])
    a[3:,:27]=-a[1:3,:27]
    costs=np.asarray([candidate_cost(t,i,k,x,c) for x in a])
    return a[int(costs.argmin())],costs


def prefix(t,i,c):
    t.reset(i)
    success=False
    for k in range(60):
        t.data.xfrc_applied[t.robot_body,:3]=[0,80 if 50<=k<60 else 0,0]
        t.advance(t.action(t.obs(i,k*.02)))
        now=(k+1)*.02
        window=t.ref.state(i,now+.1)['closed'] and not t.ref.state(i,now-.1)['closed']
        success|=bool(window and np.linalg.norm(t.point()-t.ref.state(i,now)['goal'])<c['virtual_goal_thresholds_m'][t.task])
        if t.fallen():
            return success,True,k+1
    t.data.xfrc_applied[:]=0
    return success,False,60


def witness(c):
    reports=[]
    for task in c['tasks']:
        t=Transfer(task,'cuda:0')
        assert t.verify()['passed']
        i=c['validation_reference']
        _,fallen,_=prefix(t,i,c)
        assert not fallen,'Validation prefix fell; do not replace with a survivor'
        before,last=physical_state(t),t.last.copy()
        q1,q2=clone(t),clone(t)
        for k in range(60,70):
            q1.advance(q1.action(q1.obs(i,k*.02)))
            q2.advance(q2.action(q2.obs(i,k*.02)))
        assert np.max(np.abs(physical_state(q1)-physical_state(q2)))<1e-8
        selected,costs=choose(t,i,60,42,c)
        assert np.array_equal(before,physical_state(t)) and np.array_equal(last,t.last)
        assert np.isfinite(costs).all() and np.max(np.abs(selected))<=.120001
        reports.append({'task':task,'clone_parity':True,'main_state_untouched':True,
                        'candidate_costs':costs.tolist(),'cost_selected_not_above_base':bool(costs.min()<=costs[0])})
    write('oracle_witness.json',{'status':'PASS','config_sha256':sha(CFG),'reports':reports,'not_test_result':True})
    print('ORACLE_WITNESS_PASS',json.dumps(reports),flush=True)


def write(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def evaluate(c):
    gate=json.loads((OUT/'coverage_audit.json').read_text())
    assert gate['predictive_gate_after_coverage_audit']=='PASS'
    w=json.loads((OUT/'oracle_witness.json').read_text())
    assert w['status']=='PASS' and w['config_sha256']==sha(CFG)
    if (OUT/'oracle_summary.json').exists() or (OUT/'oracle_records_partial.json').exists():
        raise FileExistsError('Existing oracle experiment must not be overwritten or repeated')
    start=time.time()
    records=[]
    for task in c['tasks']:
        t=Transfer(task,'cuda:0')
        for i in range(*c['test_references']):
            initial_success,initial_fall,initial_steps=prefix(t,i,c)
            initial,last=physical_state(t),t.last.copy()
            arms=[('ppo',0),('simple',0)]+[('oracle',seed) for seed in c['candidate_seeds']]
            for arm,seed in arms:
                q=clone(t,initial,last)
                success,fallen,steps=initial_success,initial_fall,initial_steps
                recovery=[]
                changes=[]
                offset=np.zeros(29,dtype=np.float32)
                decision_times=[]
                if not initial_fall:
                    for k in range(60,c['episode_steps']):
                        base=q.action(q.obs(i,k*.02))
                        if k<150:
                            if arm=='oracle' and (k-60)%c['decision_period']==0:
                                tic=time.time()
                                offset,_=choose(q,i,k,seed,c)
                                decision_times.append(time.time()-tic)
                            elif arm=='simple':
                                offset=np.clip(.2*(q.last-base),-.12,.12)
                                offset[27:]=0
                            elif arm=='ppo':
                                offset[:]=0
                        else:
                            offset[:]=0
                        action=base+offset
                        changes.append(float(np.mean(np.abs(action[:27]-q.last[:27]))))
                        q.advance(action)
                        now=(k+1)*.02
                        ref=q.ref.state(i,now)
                        if k<150:
                            recovery.append(float(np.sqrt(np.mean((q.data.qpos[q.qadr]-ref['joints'])**2))))
                        window=q.ref.state(i,now+.1)['closed'] and not q.ref.state(i,now-.1)['closed']
                        success|=bool(window and np.linalg.norm(q.point()-ref['goal'])<c['virtual_goal_thresholds_m'][task])
                        steps=k+1
                        if q.fallen():
                            fallen=True
                            break
                # Never reward early termination with an artificially short AUC.
                recovery+= [c['early_fall_primary_penalty_rad']]*(90-len(recovery))
                row={'task':task,'reference':i,'arm':arm,'candidate_seed':seed,
                    'prefix_fallen':initial_fall,'fallen':fallen,'executed_steps':steps,
                    'recovery_tracking_AUC_rad_s':float(sum(recovery)*.02),
                    'virtual_target_success':success,
                    'mean_abs_action_change':float(np.mean(changes)) if changes else 0.0,
                    'mean_oracle_decision_wall_s':float(np.mean(decision_times)) if decision_times else None}
                records.append(row)
                print('ORACLE_TRIAL',json.dumps(row),flush=True)
            write('oracle_records_partial.json',records)
    keys=[(task,i) for task in c['tasks'] for i in range(*c['test_references'])]
    values={arm:np.asarray([np.mean([r['recovery_tracking_AUC_rad_s'] for r in records
              if r['arm']==arm and (r['task'],r['reference'])==key]) for key in keys]) for arm in ('ppo','simple','oracle')}
    improvement=1-float(values['oracle'].mean()/values['ppo'].mean())
    rng=np.random.default_rng(c['bootstrap_seed'])
    trials=[]
    for _ in range(c['bootstrap_repetitions']):
        ids=np.concatenate([rng.choice(np.arange(j*20,(j+1)*20),20,replace=True) for j in range(2)])
        trials.append(1-float(values['oracle'][ids].mean()/values['ppo'][ids].mean()))
    ci=np.quantile(trials,[.025,.975]).tolist()
    rates={}
    for arm in values:
        rs=[r for r in records if r['arm']==arm]
        rates[arm]={'virtual_target_success_rate':sum(r['virtual_target_success'] for r in rs)/len(rs),
                    'fall_rate':sum(r['fallen'] for r in rs)/len(rs),
                    'mean_recovery_AUC_rad_s':float(values[arm].mean())}
    passed=improvement>=.05 and ci[0]>0 and rates['oracle']['virtual_target_success_rate']>=rates['ppo']['virtual_target_success_rate']-.05 and rates['oracle']['fall_rate']<=rates['ppo']['fall_rate']
    summary={'status':'COMPLETE','oracle_upper_bound_gate':'PASS' if passed else 'FAIL',
        'config_sha256':sha(CFG),'script_sha256':sha(__file__),'records':records,
        'rates':rates,'relative_AUC_improvement_vs_PPO':improvement,'reference_bootstrap_95CI':ci,
        'wall_s':time.time()-start,'reference_clusters':40,
        'candidate_seeds_are_not_independent_policy_training_seeds':True,
        'no_learned_world_model_control':True,'no_PPO_update':True,'no_diffusion_feedback':True,
        'virtual_target_is_not_physical_contact':True,'full_innovation_validated':False}
    write('oracle_summary.json',summary)
    print('ORACLE_UPPER_BOUND_COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['witness','eval'],required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    c=json.loads(CFG.read_text())
    witness(c) if args.stage=='witness' else evaluate(c)
