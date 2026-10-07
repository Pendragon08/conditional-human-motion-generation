"""Isaac-domain action-conditioned predictive gate, prior to formal PPO.

Reference-grouped train/validation/test; reset-free windows, fixed budgets.
Correct/state-only/shuffled have identical initialization and batch order.
"""
import argparse
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
import pytorch_kinematics as pk
from r3_world_ppo_core import WorldModel, prediction_loss, decoded
from r3_world_ppo_core import HISTORY, HORIZONS, SCHEMA_HASH

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--device',default='cuda:0')
p.add_argument('--updates',type=int,default=1000)
p.add_argument('--train-windows',type=int,default=65536)
p.add_argument('--eval-windows',type=int,default=16384)
a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1]
URDF=ROOT/'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_for_eval.urdf'
JOINTS=['left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint',
        'left_ankle_pitch_joint','left_ankle_roll_joint','right_hip_pitch_joint','right_hip_roll_joint',
        'right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint',
        'waist_yaw_joint','left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint',
        'left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint',
        'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint',
        'right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint']

def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def ap(y,p):
    order=np.argsort(-p,kind='stable'); yy=y[order]
    return float((np.cumsum(yy)/(np.arange(len(yy))+1)*yy).sum()/yy.sum()) if yy.sum() else None

def main():
    if a.output.exists(): raise RuntimeError('Preserve existing output')
    a.output.mkdir(parents=True)
    started=time.time(); rng=np.random.default_rng(a.seed)
    meta=json.loads((a.data/'summary.json').read_text())
    assert meta['status']=='COMPLETE' and meta['schema_hash']==SCHEMA_HASH
    assert meta['mode']=='collect' and meta['actual_transitions']==3932160
    assert meta['formal_ppo_updates']==meta['diagnostic_ppo_updates']==0
    d={k:np.load(a.data/(k+'.npy'),mmap_mode='r') for k in ['observable','labels','actions','done','reference','episode','phase']}
    permutation=np.random.default_rng(20261006).permutation(66)
    splits={'train':permutation[:44].tolist(),'validation':permutation[44:55].tolist(),
            'test':permutation[55:].tolist()}
    steps,n=d['actions'].shape[:2]
    tt=np.arange(HISTORY-1,steps-10+1)
    valid=d['episode'][tt-HISTORY+1]==d['episode'][tt+10]
    valid&=d['reference'][tt-HISTORY+1]==d['reference'][tt+10]
    time_grid=np.broadcast_to(tt[:,None],valid.shape)
    env_grid=np.broadcast_to(np.arange(n)[None,:],valid.shape)
    bundles={}; counts={}
    for split,refs in splits.items():
        mask=valid&np.isin(d['reference'][tt],refs)
        ts=time_grid[mask]; es=env_grid[mask]
        wanted=a.train_windows if split=='train' else a.eval_windows
        assert len(ts)>=wanted
        idx=rng.choice(len(ts),wanted,replace=False); ts=ts[idx]; es=es[idx]
        history=d['observable'][ts[:,None]-np.arange(HISTORY-1,-1,-1)[None,:],es[:,None]].copy()
        actions=d['actions'][ts[:,None]+np.arange(10)[None,:],es[:,None]].copy()
        current=d['labels'][ts,es].copy()
        targets={k:np.concatenate([d['labels'][ts+k,es,:48]-current[:,:48],
                                   d['labels'][ts+k,es,48:]],-1) for k in HORIZONS}
        reference=d['reference'][ts,es].copy()
        episode=np.stack([es,d['episode'][ts,es]],-1)
        phase=np.floor(d['phase'][ts,es]/.2).astype(int)
        contact=(current[:,48]>0).astype(int)+2*(current[:,49]>0).astype(int)
        # Within phase/contact strata use an exact permutation, preserving all
        # label marginals. Sorted blocks rotated by the largest ref block give
        # a cross-reference derangement iff no reference has >50% of the group.
        donor=np.full(wanted,-1,dtype=int)
        for ph,co in set(zip(phase,contact)):
            group=np.where((phase==ph)&(contact==co))[0]
            order=group[np.argsort(reference[group],kind='stable')]
            _,cnt=np.unique(reference[order],return_counts=True)
            largest=int(cnt.max())
            if largest*2>len(order): continue
            chosen=np.roll(order,largest)
            assert np.all(reference[order]!=reference[chosen])
            assert np.all(np.any(episode[order]!=episode[chosen],axis=1))
            donor[order]=chosen
        keep=donor>=0
        # Same deleted rows for correct/ridge/state-only/shuffled comparisons.
        assert keep.sum()>=wanted*.5
        old_to_new=np.full(wanted,-1); old_to_new[keep]=np.arange(keep.sum())
        mapped=old_to_new[donor[keep]]
        assert np.all(mapped>=0) and np.array_equal(np.sort(mapped),np.arange(keep.sum()))
        bundles[split]={'history':history[keep],'actions':actions[keep],
                        'current':current[keep],'targets':{k:v[keep] for k,v in targets.items()},
                        'reference':reference[keep],'episode':episode[keep],
                        'donor':mapped,'sample_t':ts[keep],'sample_env':es[keep]}
        counts[split]={'eligible_reset_free_windows':len(ts),'sampled_windows':wanted,
                       'matched_strata_windows':int(keep.sum()),'common_removed':int((~keep).sum())}
        np.savez_compressed(a.output/(split+'_windows.npz'),**{k:v for k,v in bundles[split].items() if k!='targets'},
                            **{'target_'+str(k):v[keep] for k,v in targets.items()})
    train=bundles['train']
    torch.manual_seed(a.seed)
    template=WorldModel().to(a.device)
    mean=train['history'].reshape(-1,92).mean(0); std=train['history'].reshape(-1,92).std(0).clip(.01)
    delta=np.concatenate([train['targets'][k][:,:48] for k in HORIZONS])
    with torch.no_grad():
        template.input_mean.copy_(torch.from_numpy(mean).to(a.device))
        template.input_std.copy_(torch.from_numpy(std).to(a.device))
        template.delta_mean.copy_(torch.from_numpy(delta.mean(0)).to(a.device))
        template.delta_std.copy_(torch.from_numpy(delta.std(0).clip(.01)).to(a.device))
    initial={k:v.clone() for k,v in template.state_dict().items()}
    models={}; traces={}; validation_choices={}
    tensor={split:{'history':torch.tensor(v['history'],device=a.device),
                   'actions':torch.tensor(v['actions'],device=a.device),
                   'targets':{k:torch.tensor(y,device=a.device) for k,y in v['targets'].items()}}
            for split,v in bundles.items()}
    # GPU resident windows, total <=2GiB for these sample sizes.
    batchrng=np.random.default_rng(a.seed+1000)
    batches=batchrng.integers(0,len(train['history']),size=(a.updates,1024))
    def evaluate_predictions(model,split,state_only=False):
        predictions={k:[] for k in HORIZONS}
        v=tensor[split]
        with torch.no_grad():
            for begin in range(0,len(v['history']),2048):
                result=model.predict(v['history'][begin:begin+2048],v['actions'][begin:begin+2048],state_only)
                for k in HORIZONS: predictions[k].append(decoded(model,result[k]).cpu().numpy())
        return {k:np.concatenate(v) for k,v in predictions.items()}
    for name in ['correct','state_only','shuffled']:
        model=WorldModel().to(a.device); model.load_state_dict(initial)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        traces[name]=[]; best=float('inf'); best_state=None; best_update=0
        for update,indices in enumerate(batches):
            idx=torch.tensor(indices,device=a.device)
            source=torch.tensor(train['donor'][indices],device=a.device) if name=='shuffled' else idx
            yy={k:tensor['train']['targets'][k][source] for k in HORIZONS}
            loss,_=prediction_loss(model,tensor['train']['history'][idx],tensor['train']['actions'][idx],yy,name=='state_only')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            assert torch.isfinite(loss)
            if (update+1)%200==0:
                val=evaluate_predictions(model,'validation',name=='state_only')
                # Validation joint prediction metric only; no test selection.
                error=float(np.mean((val[10][:,:27]-bundles['validation']['targets'][10][:,:27])**2))
                traces[name].append({'update':update+1,'train_loss':float(loss.detach()),'validation_joint_delta_mse':error})
                if error<best:
                    best=error;best_update=update+1;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                print('WORLD_PPO_FIT_PROGRESS',name,update+1,'validation_joint_mse',error,flush=True)
        model.load_state_dict(best_state);models[name]=model
        validation_choices[name]={'selected_update':best_update,'validation_mse':best,'updates_executed':a.updates}
        torch.save({'state_dict':model.state_dict(),'schema_hash':SCHEMA_HASH,'seed':a.seed,'splits':splits,
                    'validation_choice':validation_choices[name],'mode':name},a.output/(name+'_world.pt'))
    chain=pk.build_chain_from_urdf(URDF.read_text()).to(device=a.device,dtype=torch.float32)
    order=[JOINTS.index(x) for x in chain.get_joint_parameter_names()]
    body_names=[x for x in chain.get_link_names() if x not in ['pelvis','pelvis_contour_link']]
    def fk(q):
        with torch.no_grad():
            answer=chain.forward_kinematics(torch.tensor(q[:,order],device=a.device))
            return torch.stack([answer[x].get_matrix()[:,:3,3] for x in body_names],1).cpu().numpy()
    ridge={k:{} for k in HORIZONS}; ridge_alpha={}
    # Ridge uses same normalized ordered history, same future actions and labels.
    history_train=((train['history']-mean)/std).reshape(len(train['history']),-1).astype(np.float64)
    for k in HORIZONS:
        X=np.concatenate([history_train,train['actions'][:,:k].reshape(len(history_train),-1),np.ones((len(history_train),1))],1)
        Y=train['targets'][k].astype(np.float64)
        xtx=X.T@X; xty=X.T@Y; best=float('inf')
        for alpha in [.1,1.,10.]:
            regular=np.eye(X.shape[1])*alpha;regular[-1,-1]=0
            coef=np.linalg.solve(xtx+regular,xty)
            v=bundles['validation'];VX=np.concatenate([((v['history']-mean)/std).reshape(len(v['history']),-1),v['actions'][:,:k].reshape(len(v['history']),-1),np.ones((len(v['history']),1))],1)
            prediction=VX@coef;error=float(np.mean((prediction[:,:27]-v['targets'][k][:,:27])**2))
            if error<best:best=error;ridge[k]={'coef':coef};ridge_alpha[k]=alpha
        print('WORLD_PPO_RIDGE_LOCKED',k,ridge_alpha[k],flush=True)
    np.savez_compressed(a.output/'ridge.npz',**{'coef_'+str(k):ridge[k]['coef'] for k in HORIZONS})
    predictions={name:evaluate_predictions(model,'test',name=='state_only') for name,model in models.items()}
    test=bundles['test']
    predictions['ridge']={}
    for k in HORIZONS:
        X=np.concatenate([((test['history']-mean)/std).reshape(len(test['history']),-1),test['actions'][:,:k].reshape(len(test['history']),-1),np.ones((len(test['history']),1))],1)
        prediction=X@ridge[k]['coef'];prediction[:,48:]=prediction[:,48:].clip(0,1)
        predictions['ridge'][k]=prediction.astype(np.float32)
    metrics={}; per_ref={}; all_errors={}
    for name,by_h in predictions.items():
        metrics[name]={};per_ref[name]={};all_errors[name]={}
        for k,pred in by_h.items():
            true=test['targets'][k];q_current=test['current'][:,:27]
            pred_fk=fk(q_current+pred[:,:27]);true_fk=fk(q_current+true[:,:27])
            body_error=np.mean(np.square(pred_fk-true_fk),axis=(1,2))
            all_errors[name][k]=body_error
            metrics[name][str(k)]={'body_fk_rmse_m':float(np.sqrt(body_error.mean())),
                'joint_delta_rmse_rad':float(np.sqrt(np.mean((pred[:,:27]-true[:,:27])**2))),
                'gravity_rmse':float(np.sqrt(np.mean((pred[:,27:30]-true[:,27:30])**2))),
                'root_velocity_rmse':float(np.sqrt(np.mean((pred[:,30:36]-true[:,30:36])**2))),
                'foot_slip_velocity_rmse_m_s':float(np.sqrt(np.mean((pred[:,42:46]-true[:,42:46])**2))),
                'contact_brier':float(np.mean((pred[:,48:]-true[:,48:])**2)),
                'contact_average_precision':[ap(true[:,48+i],pred[:,48+i]) for i in range(2)]}
            per_ref[name][str(k)]={str(ref):float(np.sqrt(body_error[test['reference']==ref].mean())) for ref in splits['test']}
    gain={};brng=np.random.default_rng(20261006)
    refids=np.asarray(splits['test'])
    for comparator in ['ridge','state_only']:
        c=np.asarray([per_ref['correct']['10'][str(r)] for r in refids])
        b=np.asarray([per_ref[comparator]['10'][str(r)] for r in refids])
        pairs=brng.integers(0,len(refids),size=(10000,len(refids)))
        effect=1-c.mean()/b.mean();ci=np.percentile(1-c[pairs].mean(1)/b[pairs].mean(1),[2.5,97.5])
        gain[comparator]={'paired_reference_relative_improvement':float(effect),
                          '95pct_ci':ci.tolist(),'n_test_references':len(refids)}
    passed=all(v['paired_reference_relative_improvement']>=.1 and v['95pct_ci'][0]>0 for v in gain.values())
    result={'status':'COMPLETE','prediction_gate':'PASS' if passed else 'FAIL',
            'formal_PPO_started':False,'innovation_validated':False,'seed':a.seed,
            'warmup_transitions':meta['actual_transitions'],'splits':splits,'window_counts':counts,
            'history':10,'latent':128,'horizons':[1,5,10],'schema_hash':SCHEMA_HASH,
            'body_fk_urdf_sha256':sha(URDF),'body_fk_links':body_names,
            'metrics':metrics,'paired_H10_gate':gain,'per_reference_body_fk_rmse_m':per_ref,
            'validation_choices':validation_choices,'ridge_validation_alpha':ridge_alpha,
            'aux_gradient_updates_per_model':a.updates,'aux_batch_size':1024,
            'all_models_same_initial_weights_batch_order':True,
            'shuffled_exact_strata_permutation_cross_reference_episode':True,
            'normalizer_fit_only_train_reference':True,'locked_test_evaluations':1,
            'old_policy_previously_seen_all_66_refs':True,'novel_reference_generalization':False,
            'ground_contact_only_not_entity_strike':True,'wall_s':time.time()-started,
            'traces':traces,'files':{f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in a.output.glob('*.pt')}}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('WORLD_PPO_PREDICTION_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
