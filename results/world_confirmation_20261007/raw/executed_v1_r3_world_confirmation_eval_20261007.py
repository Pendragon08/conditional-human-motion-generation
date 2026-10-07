"""One held-new-reference inference-only confirmation; no model selection."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import pytorch_kinematics as pk
from r3_world_ppo_core import WorldModel,decoded,HISTORY,HORIZONS,SCHEMA_HASH
from r3_world_ppo_residual_core import ResidualWorldModel

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
p.add_argument('--model-lock',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1]
JOINTS=['left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint',
 'right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint',
 'waist_yaw_joint','left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint',
 'left_wrist_pitch_joint','left_wrist_yaw_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint',
 'right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint']

def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb') as s:
        for b in iter(lambda:s.read(2**20),b''):h.update(b)
    return h.hexdigest()

def ap(y,v):
    order=np.argsort(-v,kind='stable');y=y[order]
    return float((np.cumsum(y)/(np.arange(len(y))+1)*y).sum()/y.sum()) if y.sum() else None

def main():
    assert not a.output.exists(),'Preserve single confirmation invocation'
    a.output.mkdir(parents=True);started=time.time()
    lock=json.loads(a.model_lock.read_text())
    for path,digest in lock['files'].items():assert sha(path)==digest,path
    meta=json.loads((a.data/'summary.json').read_text())
    assert meta['status']=='COMPLETE' and meta['actual_transitions']==983040 and meta['diagnostic_ppo_updates']==0
    assert meta['fresh_confirmation_references'] and meta['confirmation_data_not_for_training']
    assert meta['schema_hash']==SCHEMA_HASH and meta['reference_manifest_sha256']==lock['reference_manifest_sha256']
    assert meta['disturbance_protocol']['name']=='lateral80'
    d={k:np.load(a.data/(k+'.npy'),mmap_mode='r') for k in ['observable','labels','actions','reference','episode','phase','done','cohort','push_episode']}
    steps,n=d['actions'].shape[:2];assert (steps,n)==(480,2048)
    tt=np.arange(HISTORY-1,steps-10+1)
    valid=(d['episode'][tt-HISTORY+1]==d['episode'][tt+10])&(d['reference'][tt-HISTORY+1]==d['reference'][tt+10])
    times=np.broadcast_to(tt[:,None],valid.shape);envs=np.broadcast_to(np.arange(n)[None,:],valid.shape)
    refs=list(range(11));rng=np.random.default_rng(20261007+a.seed);parts=[];counts={}
    for condition in ['clean','recovery']:
        cond=np.broadcast_to(d['cohort'][None,:]==0,valid.shape) if condition=='clean' else ((times>=60)&(times<110)&(d['cohort'][None,:]==1)&(d['episode'][tt]==d['push_episode'][None,:]))
        counts[condition]={}
        for ref in refs:
            mask=valid&cond&(d['reference'][tt]==ref);ts=times[mask];es=envs[mask]
            eligible=len(ts);counts[condition][str(ref)]={'eligible_reset_free_windows':eligible,'scored':min(1024,eligible)}
            if eligible<256:
                (a.output/'coverage.json').write_text(json.dumps(counts,indent=2)+'\n')
                raise RuntimeError('BLOCKED insufficient coverage ref=%d condition=%s count=%d'%(ref,condition,eligible))
            ix=rng.choice(eligible,min(1024,eligible),replace=False);ts=ts[ix];es=es[ix]
            cur=d['labels'][ts,es].copy()
            part={'history':d['observable'][ts[:,None]-np.arange(HISTORY-1,-1,-1)[None,:],es[:,None]].copy(),
                  'actions':d['actions'][ts[:,None]+np.arange(10)[None,:],es[:,None]].copy(),
                  'current':cur,'reference':np.full(len(ts),ref,dtype=np.int16),
                  'condition':np.full(len(ts),condition=='recovery',dtype=np.int8),'sample_t':ts,'sample_env':es,
                  'episode':d['episode'][ts,es].copy(),
                  **{'target_'+str(k):np.concatenate([d['labels'][ts+k,es,:48]-cur[:,:48],d['labels'][ts+k,es,48:]],-1) for k in HORIZONS}}
            parts.append(part)
    b={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
    np.savez_compressed(a.output/'confirmation_windows.npz',**b)
    fit=ROOT/f'artifacts/world_residual_20261007/fit_seed{a.seed}_v1'
    fs=json.loads((fit/'summary.json').read_text());ridges={};models={};predictions={}
    for name in ['correct','state_only']:
        with np.load(fit/('ridge_'+name+'.npz')) as r:ridges[name]={k:r['coef_'+str(k)].copy() for k in HORIZONS}
    for name in ['direct','residual_correct','residual_state_only','residual_shuffled']:
        model=WorldModel() if name=='direct' else ResidualWorldModel(ridges['state_only' if name=='residual_state_only' else 'correct'])
        checkpoint=torch.load(fit/(name+'.pt'),map_location=a.device,weights_only=False)
        assert checkpoint['schema_hash']==SCHEMA_HASH and checkpoint['seed']==a.seed
        model=model.to(a.device);model.load_state_dict(checkpoint['state_dict']);model.eval();models[name]=model
        byh={k:[] for k in HORIZONS}
        with torch.no_grad():
            for start in range(0,len(b['history']),2048):
                hist=torch.tensor(b['history'][start:start+2048],device=a.device)
                acts=torch.tensor(b['actions'][start:start+2048],device=a.device)
                pred=model.predict(hist,acts,name=='residual_state_only')
                if start==0 and name=='residual_state_only':
                    altered=model.predict(hist,acts+37,True)
                    assert all(torch.equal(pred[k],altered[k]) for k in HORIZONS)
                for k in HORIZONS:byh[k].append(decoded(model,pred[k]).cpu().numpy())
        predictions[name]={k:np.concatenate(v) for k,v in byh.items()}
    mean=models['residual_correct'].input_mean.cpu().numpy();std=models['residual_correct'].input_std.cpu().numpy()
    for name in ['correct','state_only']:
        byh={}
        for k in HORIZONS:
            actions=np.zeros_like(b['actions'][:,:k]) if name=='state_only' else b['actions'][:,:k]
            x=np.concatenate([((b['history']-mean)/std).reshape(len(b['history']),-1),actions.reshape(len(actions),-1),np.ones((len(actions),1))],1).astype(np.float64)
            pred=(x@ridges[name][k]).astype(np.float32);pred[:,48:]=pred[:,48:].clip(0,1);byh[k]=pred
        predictions['ridge_'+name]=byh
    assert all(np.isfinite(v).all() for pred in predictions.values() for v in pred.values())
    if fs['selection']['residual_shuffled']['update']==0:
        assert all(np.max(np.abs(predictions['residual_shuffled'][k][:,:48]-predictions['ridge_correct'][k][:,:48]))<1e-4 for k in HORIZONS)
    urdf=ROOT/'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_for_eval.urdf'
    chain=pk.build_chain_from_urdf(urdf.read_text()).to(device=a.device,dtype=torch.float32)
    order=[JOINTS.index(x) for x in chain.get_joint_parameter_names()]
    links=[x for x in chain.get_link_names() if x not in ['pelvis','pelvis_contour_link']]
    def fk(q):
        out=[]
        with torch.no_grad():
            for start in range(0,len(q),2048):
                frames=chain.forward_kinematics(torch.tensor(q[start:start+2048,order],device=a.device))
                out.append(torch.stack([frames[x].get_matrix()[:,:3,3] for x in links],1).cpu().numpy())
        return np.concatenate(out)
    truth={k:fk(b['current'][:,:27]+b['target_'+str(k)][:,:27]) for k in HORIZONS}
    metrics={};perref={}
    for name,byh in predictions.items():
        metrics[name]={};perref[name]={}
        for condition in ['clean','recovery']:
            mask=b['condition']==int(condition=='recovery');metrics[name][condition]={};perref[name][condition]={}
            for k,pred in byh.items():
                target=b['target_'+str(k)];error=np.mean((fk(b['current'][:,:27]+pred[:,:27])-truth[k])**2,axis=(1,2))
                perref[name][condition][str(k)]={str(ref):float(np.sqrt(error[mask&(b['reference']==ref)].mean())) for ref in refs}
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
    comparisons={};gates={};brng=np.random.default_rng(20261007)
    for condition in ['clean','recovery']:
        comparisons[condition]={};c=np.asarray([perref['residual_correct'][condition]['10'][str(ref)] for ref in refs])
        pairs=brng.integers(0,11,size=(10000,11))
        for comparator in ['ridge_correct','ridge_state_only','direct','residual_state_only','residual_shuffled']:
            v=np.asarray([perref[comparator][condition]['10'][str(ref)] for ref in refs]);ratio=c.mean()/v.mean()
            ratios=c[pairs].mean(1)/v[pairs].mean(1);ci=np.percentile(ratios,[2.5,97.5])
            comparisons[condition][comparator]={'relative_gain':float(1-ratio),'relative_gain_95pct_CI':[float(1-ci[1]),float(1-ci[0])],
                                               'paired_error_ratio_95pct_CI':ci.tolist()}
        gates[condition]=all(comparisons[condition][x]['relative_gain']>=.1 and comparisons[condition][x]['relative_gain_95pct_CI'][0]>0
                           for x in ['ridge_correct','ridge_state_only','residual_state_only']) and comparisons[condition]['direct']['paired_error_ratio_95pct_CI'][1]<=1.05 and comparisons[condition]['residual_shuffled']['relative_gain_95pct_CI'][0]>0
    # Lock and source hashes must still be identical at the end of inference.
    for path,digest in lock['files'].items():assert sha(path)==digest
    summary={'status':'COMPLETE','seed':a.seed,'confirmation_only':True,'new_reference_count':11,
        'confirmation_evaluations_per_seed':1,'formal_PPO_started':False,'innovation_validated':False,
        'formal_prediction_gate_passed':all(gates.values()),'domain_gates':gates,'comparisons':comparisons,
        'metrics':metrics,'per_reference_body_fk_rmse_m':perref,'window_counts':counts,
        'reference_manifest_sha256':meta['reference_manifest_sha256'],'model_lock_sha256':sha(a.model_lock),
        'actual_transitions':meta['actual_transitions'],'reset_count':meta['reset_count'],
        'disturbance_protocol':meta['disturbance_protocol'],'PPO_updates':0,'auxiliary_updates':0,
        'no_checkpoint_or_norm_selection':True,'schema_hash':SCHEMA_HASH,'wall_s':time.time()-started,
        'files':{f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in a.output.iterdir() if f.is_file()}}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('WORLD_RESIDUAL_NEW_REFERENCE_CONFIRMATION_COMPLETE',a.seed,'PASS' if all(gates.values()) else 'FAIL',flush=True)

if __name__=='__main__':main()
