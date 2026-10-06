"""Frozen learned-dynamics control ablation after the physics-oracle gate."""
import argparse
import json
import time
from pathlib import Path
import mujoco
import numpy as np
import torch
from r3_feedback_world_oracle_20261006 import prefix, clone, physical_state
from r3_feedback_world_model_pilot import Dynamics, state
from r3_sim2sim_mujoco import Transfer, observation, rotation, sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/feedback_world_model_20261006'
ART=ROOT/'artifacts/feedback_world_model_20261006'
CFG=ROOT/'configs/feedback_world_learned_control_20261006.json'
ORCFG=ROOT/'configs/feedback_world_oracle_20261006.json'


def load_model(arm,seed):
    p=ART/'models'/f'{arm}_{seed}.pt'
    manifest=json.loads((OUT/'summary.json').read_text())['checkpoint_hashes']
    assert sha(p)==manifest[p.name]
    ck=torch.load(str(p),map_location='cpu')
    model=Dynamics(ck['norm'],ck['width'],ck['state_only'],ck.get('task_free',False)).cuda().eval()
    model.load_state_dict(ck['state_dict'])
    return model


def choose_model(t,i,k,seed,c,model):
    rng=np.random.default_rng(seed+i*1009+k*100003)
    offsets=np.zeros((5,29),dtype=np.float32)
    offsets[1:3,:27]=np.clip(rng.normal(0,c['candidate_std'],(2,27)),-c['candidate_clip'],c['candidate_clip'])
    offsets[3:,:27]=-offsets[1:3,:27]
    states=np.tile(state(t),(5,1))
    last=np.tile(t.last,(5,1))
    costs=np.zeros(5)
    stopped=np.zeros(5,dtype=bool)
    task=torch.full((5,),0 if t.task=='Kick' else 1,dtype=torch.long,device='cuda')
    for h in range(c['horizon']):
        obs=np.stack([observation(t.ref,i,(k+h)*.02,s[:3],s[3:7],rotation(s[3:7]).T@s[7:10],
            s[10:13],s[13:40],s[40:],a,t.default) for s,a in zip(states,last)])
        with torch.inference_mode():
            actions=t.policy(torch.from_numpy(obs).to('cuda')).cpu().numpy()+offsets
            states=model(torch.from_numpy(states).to('cuda'),torch.from_numpy(actions).to('cuda'),task).cpu().numpy()
        assert np.isfinite(states).all()
        ref=t.ref.state(i,(k+h+1)*.02)
        step_cost=np.mean((states[:,13:40]-ref['joints'])**2,axis=-1)+np.mean((states[:,:3]-ref['pos'])**2,axis=-1)
        up=np.stack([rotation(s[3:7])[:,2] for s in states])
        step_cost+=.1*np.mean((up-rotation(ref['quat'])[:,2])**2,axis=-1)
        costs+=np.where(stopped,0,step_cost)
        new_fall=(states[:,2]<=.3)|(up[:,2]<=.5)
        costs+=np.where(new_fall&~stopped,4.0*(c['horizon']-h),0)
        stopped|=new_fall
        last=actions
    costs=costs/c['horizon']+.05*np.mean(offsets[:,:27]**2,axis=-1)
    return offsets[int(costs.argmin())],costs


def write(name, data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def witness(c,l):
    reports=[]
    i=l['validation_witness_reference']
    for task_name in c['tasks']:
        t=Transfer(task_name,'cuda:0')
        assert t.verify()['passed']
        _,fallen,_=prefix(t,i,c)
        assert not fallen
        before,last=physical_state(t),t.last.copy()
        s=state(t)
        obs=observation(t.ref,i,60*.02,s[:3],s[3:7],rotation(s[3:7]).T@s[7:10],
            s[10:13],s[13:40],s[40:],t.last,t.default)
        assert np.max(np.abs(obs-t.obs(i,60*.02)))<2e-4
        with torch.inference_mode():
            batched=t.policy(torch.from_numpy(np.tile(obs,(5,1))).cuda()).cpu().numpy()
        assert np.max(np.abs(batched-t.action(obs)))<2e-4
        for arm in l['arms']+['state_only_multi5']:
            model=load_model(arm,42)
            selected,costs=choose_model(t,i,60,42,c,model)
            assert np.array_equal(before,physical_state(t)) and np.array_equal(last,t.last)
            assert np.isfinite(costs).all() and np.max(np.abs(selected))<=.120001
            if arm=='state_only_multi5':
                assert np.array_equal(selected,np.zeros(29)), 'Action-free cost must choose zero residual'
            reports.append({'task':task_name,'arm':arm,'finite_costs':costs.tolist(),'actual_state_untouched':True})
    write('learned_control_witness.json',{'status':'PASS','config_sha256':sha(CFG),
        'oracle_config_sha256':sha(ORCFG),'reports':reports,'not_test_result':True})
    print('LEARNED_CONTROL_WITNESS_PASS',json.dumps(reports),flush=True)


def evaluate(c,l):
    diagnostic=json.loads((OUT/'oracle_reviewer_diagnostic.json').read_text())
    assert diagnostic['learned_W_allowed_next']
    w=json.loads((OUT/'learned_control_witness.json').read_text())
    assert w['status']=='PASS' and w['config_sha256']==sha(CFG)
    if (OUT/'learned_control_summary.json').exists() or (OUT/'learned_control_records_partial.json').exists():
        raise FileExistsError('Do not duplicate the learned-control pilot')
    models={(arm,seed):load_model(arm,seed) for arm in l['arms'] for seed in l['model_and_candidate_seeds']}
    rows=[]
    start=time.time()
    for task_name in c['tasks']:
        t=Transfer(task_name,'cuda:0')
        for i in range(*l['evaluation_references']):
            initial_success,initial_fall,initial_steps=prefix(t,i,c)
            initial,last=physical_state(t),t.last.copy()
            for arm in l['arms']:
                for seed in l['model_and_candidate_seeds']:
                    q=clone(t,initial,last)
                    success,fallen,steps=initial_success,initial_fall,initial_steps
                    recovery,changes,decision_times=[],[],[]
                    offset=np.zeros(29,dtype=np.float32)
                    if not fallen:
                        for k in range(60,c['episode_steps']):
                            base=q.action(q.obs(i,k*.02))
                            if k<150 and (k-60)%c['decision_period']==0:
                                tic=time.time()
                                offset,_=choose_model(q,i,k,seed,c,models[(arm,seed)])
                                decision_times.append(time.time()-tic)
                            if k>=150:
                                offset[:]=0
                            action=base+offset
                            changes.append(float(np.mean(np.abs(action[:27]-q.last[:27]))))
                            q.advance(action)
                            now=(k+1)*.02
                            ref=q.ref.state(i,now)
                            if k<150:
                                recovery.append(float(np.sqrt(np.mean((q.data.qpos[q.qadr]-ref['joints'])**2))))
                            window=q.ref.state(i,now+.1)['closed'] and not q.ref.state(i,now-.1)['closed']
                            success|=bool(window and np.linalg.norm(q.point()-ref['goal'])<c['virtual_goal_thresholds_m'][task_name])
                            steps=k+1
                            if q.fallen():
                                fallen=True
                                break
                    recovery+=[c['early_fall_primary_penalty_rad']]*(90-len(recovery))
                    row={'task':task_name,'reference':i,'arm':arm,'seed':seed,'prefix_fallen':initial_fall,
                        'fallen':fallen,'executed_steps':steps,'recovery_tracking_AUC_rad_s':float(sum(recovery)*.02),
                        'virtual_target_success':success,'mean_abs_action_change':float(np.mean(changes)) if changes else 0.0,
                        'mean_decision_wall_s':float(np.mean(decision_times)) if decision_times else None,
                        'max_decision_wall_s':float(max(decision_times)) if decision_times else None,
                        'deadline_overruns':sum(x>l['online_deadline_s'] for x in decision_times),
                        'decisions':len(decision_times)}
                    rows.append(row)
                    print('LEARNED_CONTROL_TRIAL',json.dumps(row),flush=True)
            write('learned_control_records_partial.json',rows)
    oracle=json.loads((OUT/'oracle_summary.json').read_text())
    baseline=oracle['records']
    keys=[(t,i) for t in c['tasks'] for i in range(*l['evaluation_references'])]
    def macro(rs,arm):
        return np.asarray([np.mean([r['recovery_tracking_AUC_rad_s'] for r in rs
            if r['arm']==arm and (r['task'],r['reference'])==key]) for key in keys])
    base,simple=macro(baseline,'ppo'),macro(baseline,'simple')
    primary=macro(rows,'mlp_multi5')
    rng=np.random.default_rng(20261006)
    trials,vs_simple=[],[]
    for _ in range(2000):
        ids=np.concatenate([rng.choice(np.arange(j*20,(j+1)*20),20,replace=True) for j in range(2)])
        trials.append(1-float(primary[ids].mean()/base[ids].mean()))
        vs_simple.append(1-float(primary[ids].mean()/simple[ids].mean()))
    ci=np.quantile(trials,[.025,.975]).tolist()
    simple_ci=np.quantile(vs_simple,[.025,.975]).tolist()
    rates={}
    for arm in l['arms']:
        rs=[r for r in rows if r['arm']==arm]
        rates[arm]={'mean_recovery_AUC_rad_s':float(macro(rows,arm).mean()),
            'virtual_goal_success_rate':float(np.mean([r['virtual_target_success'] for r in rs])),
            'fall_rate':float(np.mean([r['fallen'] for r in rs])),
            'deadline_overrun_rate':sum(r['deadline_overruns'] for r in rs)/max(1,sum(r['decisions'] for r in rs))}
    improvement=1-float(primary.mean()/base.mean())
    passed=(improvement>=.1 and ci[0]>0 and simple_ci[0]>0
        and rates['mlp_multi5']['virtual_goal_success_rate']>=oracle['rates']['ppo']['virtual_target_success_rate']-.05
        and rates['mlp_multi5']['fall_rate']<=oracle['rates']['ppo']['fall_rate'])
    summary={'status':'COMPLETE','learned_control_gate':'PASS' if passed else 'FAIL',
        'config_sha256':sha(CFG),'oracle_config_sha256':sha(ORCFG),'script_sha256':sha(__file__),
        'model_hashes':json.loads((OUT/'summary.json').read_text())['checkpoint_hashes'],
        'relative_AUC_improvement_vs_PPO':improvement,'reference_bootstrap95CI_vs_PPO':ci,
        'relative_AUC_improvement_vs_simple':1-float(primary.mean()/simple.mean()),'reference_bootstrap95CI_vs_simple':simple_ci,
        'rates':rates,'baseline_rates_reused':oracle['rates'],'records':rows,'wall_s':time.time()-start,
        'state_only_control_outcomes_reuse_PPO_as_analytically_identical':True,
        'expected_trials':240,'completed_trials':len(rows),'reference_clusters':40,
        'not_independent_final_confirmation':True,'no_PPO_update':True,'no_extra_IK':True,
        'diffusion_feedback_not_executed':True,'full_innovation_validated':False}
    write('learned_control_summary.json',summary)
    print('LEARNED_CONTROL_COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['witness','eval'],required=True)
    a=p.parse_args()
    c=json.loads(ORCFG.read_text());l=json.loads(CFG.read_text())
    witness(c,l) if a.stage=='witness' else evaluate(c,l)
