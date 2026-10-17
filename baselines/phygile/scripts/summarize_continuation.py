"""Recompute continuation metrics, preserve effective state and package final model."""
from pathlib import Path
import csv,hashlib,json,shutil
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root=Path(__file__).resolve().parents[1];out=root/'results/continuation_2000'
suite=json.loads((out/'evaluation_suite.json').read_text());assert suite['status']=='COMPLETED'
rows=[]
for entry in suite['entries']:
    folder=out/'evaluation'/entry['name'];m=json.loads((folder/'metrics.json').read_text())
    with np.load(folder/'raw_pre_reset.npz',allow_pickle=False) as raw:
        assert all(np.isfinite(raw[k]).all() for k in raw.files)
        metrics={'mpjpe_m_until_termination':np.linalg.norm(raw['reference_body_pos']-raw['robot_body_pos'],axis=-1).mean(axis=-1),
                 'mpjae_rad_until_termination':np.abs(raw['reference_joint_pos']-raw['robot_joint_pos']).mean(axis=-1),
                 'mpjve_rad_s_until_termination':np.abs(raw['reference_joint_vel']-raw['robot_joint_vel']).mean(axis=-1)}
        for key,value in metrics.items():assert abs(float(value.mean())-m['metrics'][key])<1e-6
        lengths=[];successes=0
        for eid in range(m['num_envs']):
            mask=raw['env_id']==eid;frame=raw['frame'][mask]
            assert np.array_equal(frame,np.arange(len(frame)))
            lengths.append(len(frame))
            successes+=int(len(frame)==m['frames'] and not raw['terminated'][mask].any())
        assert lengths==m['episode_lengths'] and successes==m['successes']
        with (folder/'per_step_metrics.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['env_id','command_frame','terminated','timeout']+list(metrics))
            w.writerows(zip(raw['env_id'],raw['frame'],raw['terminated'],raw['timeout'],*metrics.values()))
    rows.append({'name':entry['name'],'updates':2000,'split':entry['split'],'clip':entry['clip_index'],
                 'successes':m['successes'],'trials':m['num_envs'],'success_fraction':m['full_clip_success_fraction'],
                 'mean_duration_s':float(np.mean(m['episode_lengths']))/m['fps'],'steps':m['steps_evaluated'],**m['metrics']})
groups=[]
for split in ['train','test']:
    g=[r for r in rows if r['split']==split]
    groups.append({'updates':2000,'split':split,'clips':len(g),'successes':sum(r['successes'] for r in g),
                   'trials':sum(r['trials'] for r in g),'success_fraction':sum(r['successes'] for r in g)/sum(r['trials'] for r in g),
                   'mean_duration_s_macro':float(np.mean([r['mean_duration_s'] for r in g])),
                   'mpjpe_m_macro_until_failure':float(np.mean([r['mpjpe_m_until_termination'] for r in g]))})
summary={'scope':'Exploratory checkpoint continuation, 2000 total updates, one active expert, one training seed',
         'groups':groups,'rows':rows}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'evaluation_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
run=next((root/'logs/rsl_rl/phygile_moe_pilot').glob('*extended2000_seed42'))
params=out/'effective_configuration';params.mkdir(exist_ok=True)
for name in ['env.yaml','agent.yaml']:shutil.copyfile(run/'params'/name,params/name)
for name in ['resume_provenance.json','reference_validation.json']:shutil.copyfile(run/name,out/name)
p=run/'model_1999.pt';cp=torch.load(p,map_location='cpu',weights_only=False)
assert cp['iter']==1999
state=cp['model_state_dict'];assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
release=root/'release_assets';release.mkdir(exist_ok=True)
name='phygile_initial_expert_2000updates_seed42.pt';shutil.copyfile(p,release/name)
(out/'checkpoint_verification.json').write_text(json.dumps({'file':name,'bytes':p.stat().st_size,
    'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'iteration':1999,'optimizer_updates':2000,
    'all_model_tensors_finite':True,'active_experts':1},indent=2)+'\n')
ea=EventAccumulator(str(run),size_guidance={'scalars':0});ea.Reload()
curves={k:[{'step':e.step,'value':e.value,'wall_time':e.wall_time} for e in ea.Scalars(k)] for k in ea.Tags()['scalars']}
(out/'learning_curves.json').write_text(json.dumps(curves,indent=2)+'\n')
print(json.dumps(summary['groups'],indent=2),flush=True)
