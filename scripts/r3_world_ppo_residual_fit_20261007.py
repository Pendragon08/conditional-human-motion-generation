"""Bounded development comparison. Retired test references are never scored.

Four-reference-fold OOF ridge residuals avoid training on ridge's in-sample
errors. Wrong supervision permutes residuals, not good linear predictions.
No formal PPO may be launched from this development-only outcome.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import pytorch_kinematics as pk
from r3_world_ppo_core import WorldModel, decoded, prediction_loss, HORIZONS, HISTORY, SCHEMA_HASH
from r3_world_ppo_residual_core import ResidualWorldModel

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--device',default='cuda:0')
p.add_argument('--updates',type=int,default=1000)
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

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as s:
        for chunk in iter(lambda:s.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()

def ap(y,v):
    order=np.argsort(-v,kind='stable'); y=y[order]
    return float((np.cumsum(y)/(np.arange(len(y))+1)*y).sum()/y.sum()) if y.sum() else None

def main():
    assert not a.output.exists(),'Preserve completed/failed invocation'
    a.output.mkdir(parents=True)
    started=time.time();rng=np.random.default_rng(a.seed)
    meta=json.loads((a.data/'summary.json').read_text())
    assert meta['status']=='COMPLETE' and meta['schema_hash']==SCHEMA_HASH
    assert meta['actual_transitions']==3932160 and meta['diagnostic_ppo_updates']==0
    assert meta['disturbance_protocol']['name']=='lateral80'
    d={k:np.load(a.data/(k+'.npy'),mmap_mode='r') for k in
       ['observable','labels','actions','reference','episode','phase','done','cohort','push_direction','push_episode']}
    perm=np.random.default_rng(20261006).permutation(66)
    splits={'train':perm[:44].tolist(),'validation':perm[44:55].tolist(),
            'retired_test_not_evaluated':perm[55:].tolist()}
    steps,n=d['actions'].shape[:2]
    tt=np.arange(HISTORY-1,steps-10+1)
    valid=(d['episode'][tt-HISTORY+1]==d['episode'][tt+10])
    valid&=d['reference'][tt-HISTORY+1]==d['reference'][tt+10]
    times=np.broadcast_to(tt[:,None],valid.shape)
    envs=np.broadcast_to(np.arange(n)[None,:],valid.shape)
    conditions={'clean':np.broadcast_to(d['cohort'][None,:]==0,valid.shape),
                'recovery':(times>=60)&(times<110)&(d['cohort'][None,:]==1)&
                            (d['episode'][tt]==d['push_episode'][None,:])}
    bundles={};counts={}
    for split in ['train','validation']:
        wanted=32768 if split=='train' else 8192
        parts=[];counts[split]={}
        for condition,conditionmask in conditions.items():
            mask=valid&conditionmask&np.isin(d['reference'][tt],splits[split])
            ts=times[mask];es=envs[mask];eligible=len(ts)
            assert eligible>=wanted,(split,condition,eligible)
            idx=rng.choice(eligible,wanted,replace=False);ts=ts[idx];es=es[idx]
            ref=d['reference'][ts,es].copy()
            ep=np.stack([es,d['episode'][ts,es]],-1)
            phase=np.floor(d['phase'][ts,es]/.2).astype(int)
            cur=d['labels'][ts,es].copy()
            contact=(cur[:,48]>0).astype(int)+2*(cur[:,49]>0).astype(int)
            sign=np.zeros(wanted,dtype=int) if condition=='clean' else d['push_direction'][es]
            recoverybin=np.full(wanted,-1) if condition=='clean' else (ts-60)//10
            donor=np.full(wanted,-1,dtype=np.int64)
            strata=np.stack([phase,contact,sign,recoverybin],1)
            for stratum in np.unique(strata,axis=0):
                group=np.where(np.all(strata==stratum,axis=1))[0]
                order=group[np.argsort(ref[group],kind='stable')]
                _,cnt=np.unique(ref[order],return_counts=True);largest=int(cnt.max())
                if 2*largest>len(order):continue
                chosen=np.roll(order,largest)
                assert np.all(ref[order]!=ref[chosen]) and np.all(np.any(ep[order]!=ep[chosen],axis=1))
                donor[order]=chosen
            keep=donor>=0
            assert keep.sum()>=wanted*.5,(split,condition,'insufficient exact matched strata')
            mapping=np.full(wanted,-1,dtype=np.int64);mapping[keep]=np.arange(keep.sum())
            donor=mapping[donor[keep]]
            assert np.array_equal(np.sort(donor),np.arange(keep.sum()))
            ts=ts[keep];es=es[keep];cur=cur[keep]
            targets={k:np.concatenate([d['labels'][ts+k,es,:48]-cur[:,:48],
                                     d['labels'][ts+k,es,48:]],-1) for k in HORIZONS}
            part={'history':d['observable'][ts[:,None]-np.arange(HISTORY-1,-1,-1)[None,:],es[:,None]].copy(),
                  'actions':d['actions'][ts[:,None]+np.arange(10)[None,:],es[:,None]].copy(),
                  'current':cur,'reference':ref[keep],'episode':ep[keep],
                  'donor':donor,'sample_t':ts,'sample_env':es,'strata':strata[keep],
                  'condition':np.full(len(ts),int(condition=='recovery'),dtype=np.int8),
                  **{'target_'+str(k):y for k,y in targets.items()}}
            if condition=='recovery':
                assert np.all(ts>=60) and np.all(ts+10<120)
                assert np.all(d['episode'][ts,es]==d['push_episode'][es])
            parts.append(part)
            counts[split][condition]={'eligible_reset_free_windows':eligible,'sampled':wanted,
                                      'matched':int(keep.sum()),'common_removed':int((~keep).sum())}
        parts[1]['donor']+=len(parts[0]['history'])
        bundle={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
        assert np.array_equal(np.sort(bundle['donor']),np.arange(len(bundle['donor'])))
        assert np.array_equal(bundle['condition'],bundle['condition'][bundle['donor']])
        bundles[split]=bundle
        np.savez_compressed(a.output/(split+'_windows.npz'),**bundle)
    train=bundles['train'];val=bundles['validation']
    mean=train['history'].reshape(-1,92).mean(0);std=train['history'].reshape(-1,92).std(0).clip(.01)
    deltas=np.concatenate([train['target_'+str(k)][:,:48] for k in HORIZONS])
    delta_mean=deltas.mean(0);delta_std=deltas.std(0).clip(.01)
    def X(b,k,state_only=False):
        actions=np.zeros_like(b['actions'][:,:k]) if state_only else b['actions'][:,:k]
        return np.concatenate([((b['history']-mean)/std).reshape(len(b['history']),-1),
                               actions.reshape(len(actions),-1),np.ones((len(actions),1))],1).astype(np.float64)
    def solve_stats(xtx,xty,alpha):
        reg=np.eye(len(xtx))*alpha;reg[-1,-1]=0
        return np.linalg.solve(xtx+reg,xty)
    ridges={};oof={};choices={};fold_map={int(r):j%4 for j,r in enumerate(splits['train'])}
    folds=np.asarray([fold_map[int(r)] for r in train['reference']])
    for state_only in [False,True]:
        label='state_only' if state_only else 'correct'
        ridges[label]={};oof[label]={};choices[label]={}
        for k in HORIZONS:
            x=X(train,k,state_only);y=train['target_'+str(k)].astype(np.float64);vx=X(val,k,state_only)
            best=float('inf');selected=None;xtx=x.T@x;xty=x.T@y
            for alpha in [.1,1.,10.]:
                coef=solve_stats(xtx,xty,alpha)
                error=float(np.mean((vx@coef-val['target_'+str(k)])[:,:27]**2))
                if error<best:best=error;selected=(alpha,coef)
            alpha,coef=selected;ridges[label][k]=coef;choices[label][k]=alpha
            prior=np.empty_like(y,dtype=np.float32)
            for f in range(4):
                fit=folds!=f;hold=~fit
                foldcoef=solve_stats(x[fit].T@x[fit],x[fit].T@y[fit],alpha)
                prior[hold]=(x[hold]@foldcoef).astype(np.float32)
                np.savez_compressed(a.output/f'ridge_{label}_H{k}_fold{f}.npz',coef=foldcoef,
                                    train_references=np.asarray([r for r in splits['train'] if fold_map[r]!=f]))
            oof[label][k]=prior
            print('WORLD_RESIDUAL_RIDGE_AND_OOF_LOCKED',label,k,alpha,flush=True)
        np.savez_compressed(a.output/f'ridge_{label}.npz',**{'coef_'+str(k):v for k,v in ridges[label].items()},
                            input_mean=mean,input_std=std)
    residual={k:train['target_'+str(k)][:,:48]-oof['correct'][k][:,:48] for k in HORIZONS}
    np.savez_compressed(a.output/'oof_training_residuals.npz',fold=folds,donor=train['donor'],
                        **{'residual_'+str(k):v for k,v in residual.items()},
                        **{'correct_prior_'+str(k):v for k,v in oof['correct'].items()},
                        **{'state_only_prior_'+str(k):v for k,v in oof['state_only'].items()})
    # Same initialization and stochastic minibatch order across learned arms.
    torch.manual_seed(a.seed);template=WorldModel().to(a.device)
    initial={k:v.detach().clone() for k,v in template.state_dict().items()}
    torch.manual_seed(a.seed);res_template=ResidualWorldModel(ridges['correct']).to(a.device)
    residual_init={k:v.detach().clone() for k,v in res_template.state_dict().items()}
    assert all(torch.equal(v,template.encoder.state_dict()[k]) for k,v in res_template.encoder.state_dict().items())
    tensors={s:{k:torch.tensor(b[k],device=a.device) for k in ['history','actions']+
               ['target_'+str(k) for k in HORIZONS]} for s,b in bundles.items()}
    priors={name:{k:torch.tensor(v,device=a.device) for k,v in byh.items()} for name,byh in oof.items()}
    batches=np.random.default_rng(a.seed+1000).integers(0,len(train['history']),size=(a.updates,1024))
    models={};traces={};selection={}
    def predict_all(model,state_only=False):
        predictions={k:[] for k in HORIZONS}
        with torch.no_grad():
            for start in range(0,len(val['history']),2048):
                pred=model.predict(tensors['validation']['history'][start:start+2048],
                                   tensors['validation']['actions'][start:start+2048],state_only)
                for k in HORIZONS:predictions[k].append(decoded(model,pred[k]).cpu().numpy())
        return {k:np.concatenate(v) for k,v in predictions.items()}
    for name in ['direct','residual_correct','residual_state_only','residual_shuffled']:
        state_only=name=='residual_state_only'
        if name=='direct':model=WorldModel().to(a.device);model.load_state_dict(initial)
        else:
            model=ResidualWorldModel(ridges['state_only' if state_only else 'correct']).to(a.device)
            if not state_only:model.load_state_dict(residual_init)
            else:
                init={k:v for k,v in residual_init.items() if not k.startswith('ridge_')}
                model.load_state_dict(init,strict=False)
        with torch.no_grad():
            for attr,value in [('input_mean',mean),('input_std',std),('delta_mean',delta_mean),('delta_std',delta_std)]:
                getattr(model,attr).copy_(torch.tensor(value,device=a.device))
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        initial_param_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()
        prediction=predict_all(model,state_only);recovery=val['condition']==1
        best=float(np.mean((prediction[10][recovery,:27]-val['target_10'][recovery,:27])**2))
        best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        trace=[{'update':0,'validation_recovery_joint_delta_mse':best}];best_update=0
        gradient_audit=[]
        for update,indices in enumerate(batches):
            idx=torch.tensor(indices,device=a.device)
            if name=='direct':
                loss,_=prediction_loss(model,tensors['train']['history'][idx],tensors['train']['actions'][idx],
                    {k:tensors['train']['target_'+str(k)][idx] for k in HORIZONS})
            else:
                group='state_only' if state_only else 'correct'
                baseline={k:priors[group][k][idx] for k in HORIZONS}
                prediction=model.predict(tensors['train']['history'][idx],tensors['train']['actions'][idx],state_only,baseline)
                source=torch.tensor(train['donor'][indices],device=a.device) if name=='residual_shuffled' else idx
                loss=0.
                for k in HORIZONS:
                    if name=='residual_shuffled':
                        delta=baseline[k][:,:48]+tensors['train']['target_'+str(k)][source,:48]-priors['correct'][k][source,:48]
                    else:delta=tensors['train']['target_'+str(k)][source,:48]
                    target=(delta-model.delta_mean)/model.delta_std
                    loss=loss+(prediction[k][:,:48]-target).square().mean()
                    loss=loss+.1*torch.nn.functional.binary_cross_entropy_with_logits(prediction[k][:,48:],
                              tensors['train']['target_'+str(k)][source,48:])
                loss=loss/3
            opt.zero_grad();loss.backward()
            grads=[v.grad for v in model.parameters() if v.grad is not None]
            assert grads and all(v.is_cuda and torch.isfinite(v).all() for v in grads)
            grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            gradient_audit.append({'update':update+1,'device':str(grads[0].device),
                'finite':True,'grad_norm_before_clip':grad_norm,'nonzero_parameter_tensors':sum(int(torch.count_nonzero(v)>0) for v in grads)})
            opt.step()
            assert torch.isfinite(loss)
            if (update+1)%200==0:
                prediction=predict_all(model,state_only)
                recovery=val['condition']==1
                error=float(np.mean((prediction[10][recovery,:27]-val['target_10'][recovery,:27])**2))
                trace.append({'update':update+1,'train_loss':float(loss.detach()),'validation_recovery_joint_delta_mse':error})
                if error<best:best=error;best_update=update+1;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                print('WORLD_RESIDUAL_FIT_PROGRESS',name,update+1,error,flush=True)
        final_param_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()
        torch.save(opt.state_dict(),a.output/(name+'_final_optimizer.pt'))
        (a.output/(name+'_gradient_audit.json')).write_text(json.dumps(gradient_audit)+'\n')
        model.load_state_dict(best_state);models[name]=model;traces[name]=trace
        selection[name]={'update':best_update,'validation_recovery_joint_delta_mse':best,'updates_executed':a.updates,
                         'initial_parameter_hash':initial_param_hash,'final_training_parameter_hash':final_param_hash,
                         'optimizer_step_count':min(int(v['step']) for v in opt.state.values())}
        torch.save({'state_dict':model.state_dict(),'schema_hash':SCHEMA_HASH,'development_only':True,
                    'mode':name,'splits':splits,'seed':a.seed,'selection':selection[name]},a.output/(name+'.pt'))
    predictions={name:predict_all(model,name=='residual_state_only') for name,model in models.items()}
    for name in ['correct','state_only']:
        predictions['ridge_'+name]={k:(X(val,k,name=='state_only')@ridges[name][k]).astype(np.float32) for k in HORIZONS}
        for v in predictions['ridge_'+name].values():v[:,48:]=v[:,48:].clip(0,1)
    chain=pk.build_chain_from_urdf(URDF.read_text()).to(device=a.device,dtype=torch.float32)
    order=[JOINTS.index(x) for x in chain.get_joint_parameter_names()]
    links=[x for x in chain.get_link_names() if x not in ['pelvis','pelvis_contour_link']]
    def fk(q):
        with torch.no_grad():
            out=chain.forward_kinematics(torch.tensor(q[:,order],device=a.device))
            return torch.stack([out[x].get_matrix()[:,:3,3] for x in links],1).cpu().numpy()
    metrics={};perref={}
    for name,byh in predictions.items():
        metrics[name]={};perref[name]={}
        for condition in ['clean','recovery']:
            mask=val['condition']==int(condition=='recovery');metrics[name][condition]={};perref[name][condition]={}
            for k,pred in byh.items():
                target=val['target_'+str(k)]
                error=np.mean((fk(val['current'][:,:27]+pred[:,:27])-fk(val['current'][:,:27]+target[:,:27]))**2,axis=(1,2))
                perref[name][condition][str(k)]={str(r):float(np.sqrt(error[mask&(val['reference']==r)].mean())) for r in splits['validation']}
                metrics[name][condition][str(k)]={'body_fk_rmse_m':float(np.sqrt(error[mask].mean())),
                    'joint_delta_rmse_rad':float(np.sqrt(((pred[mask,:27]-target[mask,:27])**2).mean())),
                    'root_linear_velocity_rmse_m_s':float(np.sqrt(((pred[mask,30:33]-target[mask,30:33])**2).mean())),
                    'root_angular_velocity_rmse_rad_s':float(np.sqrt(((pred[mask,33:36]-target[mask,33:36])**2).mean())),
                    'gravity_rmse':float(np.sqrt(((pred[mask,27:30]-target[mask,27:30])**2).mean())),
                    'root_height_rmse_m':float(np.sqrt(((pred[mask,46]-target[mask,46])**2).mean())),
                    'foot_slip_rmse_m_s':float(np.sqrt(((pred[mask,42:46]-target[mask,42:46])**2).mean())),
                    'contact_brier':float(((pred[mask,48:]-target[mask,48:])**2).mean()),
                    'contact_average_precision':[ap(target[mask,48+i],pred[mask,48+i]) for i in range(2)],
                    'contact_prevalence':target[mask,48:].mean(0).tolist()}
    comparisons={};brng=np.random.default_rng(20261007)
    for comparator in ['ridge_correct','direct','residual_state_only','residual_shuffled']:
        c=np.asarray(list(perref['residual_correct']['recovery']['10'].values()))
        b=np.asarray(list(perref[comparator]['recovery']['10'].values()))
        pairs=brng.integers(0,len(c),size=(10000,len(c)))
        comparisons[comparator]={'development_relative_gain':float(1-c.mean()/b.mean()),
            'development_reference_bootstrap_95pct_ci':np.percentile(1-c[pairs].mean(1)/b[pairs].mean(1),[2.5,97.5]).tolist()}
    # Development triage only; selected checkpoint/architecture share validation.
    promising=all(comparisons[c]['development_relative_gain']>=.1 and comparisons[c]['development_reference_bootstrap_95pct_ci'][0]>0
                  for c in ['ridge_correct','residual_state_only'])
    cleanratio=metrics['residual_correct']['clean']['10']['body_fk_rmse_m']/metrics['ridge_correct']['clean']['10']['body_fk_rmse_m']
    directratio=metrics['residual_correct']['recovery']['10']['body_fk_rmse_m']/metrics['direct']['recovery']['10']['body_fk_rmse_m']
    promising=promising and cleanratio<=1.05 and directratio<=1.05 and comparisons['residual_shuffled']['development_relative_gain']>0
    summary={'status':'COMPLETE','development_only':True,'innovation_validated':False,
        'formal_PPO_started':False,'formal_prediction_gate_passed':False,
        'development_triage':'PROMISING_NEEDS_NEW_CONFIRMATION_REFS' if promising else 'FAIL_STOP_CANDIDATE',
        'seed':a.seed,'fresh_actual_transitions':meta['actual_transitions'],'actual_reset_count':meta['reset_count'],
        'disturbance_protocol':meta['disturbance_protocol'],'splits':splits,'retired_test_evaluations':0,
        'window_counts':counts,'metrics':metrics,'per_reference_body_fk_rmse_m':perref,
        'development_comparisons':comparisons,'clean_body_fk_ratio_vs_ridge':cleanratio,
        'recovery_body_fk_ratio_vs_direct':directratio,
        'selection':selection,'ridge_validation_alpha':choices,'OOF_reference_folds':fold_map,
        'OOF_excluded_labels_per_reference_fold':True,'shuffled_OOF_continuous_residual_exact_permutation':True,
        'contact_labels_permuted_separately_with_original_marginals':True,
        'future_action_zero_in_ridge_and_residual_branches':True,'shared_initial_encoders_and_batch_order':True,
        'aux_updates_per_model':a.updates,'aux_batch_size':1024,'learned_model_count':4,
        'recovery_only_after_force_ended_no_future_force_in_H10':True,
        'ground_contact_only_no_entity_strike':True,'excluded_reset_windows_not_control_success':True,
        'wall_s':time.time()-started,'traces':traces,'schema_hash':SCHEMA_HASH,'URDF_sha256':sha(URDF),
        'source_collect_summary_sha256':sha(a.data/'summary.json'),
        'files':{f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in a.output.iterdir() if f.is_file()}}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('WORLD_RESIDUAL_PREDICTION_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
