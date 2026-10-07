"""PPO optimizer restoration and auditable online residual supervision."""
import copy
import hashlib
import numpy as np
import torch
from r3_world_ppo_core import HORIZONS, HISTORY, INPUT, CONTINUOUS

def tensor_hash(state):
    h=hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def restore_original_adam(algorithm, original, checkpoint):
    """Match moments by named original parameters; zero new input columns."""
    saved=checkpoint['optimizer_state_dict']
    assert len(saved['param_groups'])==1
    names=list(dict(original.named_parameters()))
    old_ids=saved['param_groups'][0]['params']
    assert len(names)==len(old_ids)
    target=list(algorithm.policy.named_parameters())
    assert [n for n,_ in target[:len(names)]]==names
    states={}; maxerror=0.
    for name, oldid in zip(names,old_ids):
        newidx=next(i for i,(n,_) in enumerate(target) if n==name)
        parameter=target[newidx][1]; state={}
        for key,value in saved['state'][oldid].items():
            if not torch.is_tensor(value): state[key]=copy.deepcopy(value); continue
            if value.ndim and value.shape!=parameter.shape:
                assert name in ['actor.0.weight','critic.0.weight'] and value.shape[1]==626 and parameter.shape[1]==754
                expanded=torch.zeros_like(parameter); expanded[:,:626]=value.to(parameter.device)
                assert torch.count_nonzero(expanded[:,626:])==0
                maxerror=max(maxerror,float((expanded[:,:626]-value.to(parameter.device)).abs().max()))
                state[key]=expanded
            else:
                state[key]=value.clone()
        states[newidx]=state
    group=copy.deepcopy(saved['param_groups'][0]); group['params']=list(range(len(target)))
    algorithm.optimizer.load_state_dict({'state':states,'param_groups':[group]})
    algorithm.learning_rate=float(group['lr'])
    for name,oldid in zip(names,old_ids):
        parameter=dict(target)[name]
        for key,value in saved['state'][oldid].items():
            if torch.is_tensor(value):
                actual=algorithm.optimizer.state[parameter][key]
                if actual.ndim and actual.shape!=value.shape: actual=actual[:,:626]
                assert torch.equal(actual.cpu(),value.cpu())
    assert all(p not in algorithm.optimizer.state for _,p in target[len(names):])
    return {'matched_original_parameters':len(names),'fresh_new_parameters':len(target)-len(names),
            'moments_max_abs_error':maxerror,'restored_learning_rate':algorithm.learning_rate,
            'restored_min_adam_step':min(float(s['step']) for s in saved['state'].values()),
            'new_columns_zero_moments':True}

def make_windows(observable, actions, labels, episodes, references, phases, global_start):
    """Complete causal histories and H10 futures wholly inside one rollout."""
    n=observable.shape[1]; device=observable.device
    tt=torch.arange(HISTORY-1,actions.shape[0]-10+1,device=device)
    valid=(episodes[tt-HISTORY+1]==episodes[tt+10]) & (references[tt-HISTORY+1]==references[tt+10])
    t,e=torch.where(valid); t=tt[t]
    history=observable[t[:,None]-torch.arange(9,-1,-1,device=device),e[:,None]]
    future=actions[t[:,None]+torch.arange(10,device=device),e[:,None]]
    current=labels[t,e]
    targets={k:torch.cat([labels[t+k,e,:48]-current[:,:48],labels[t+k,e,48:]],-1) for k in HORIZONS}
    ref=references[t,e].cpu().numpy(); ep=episodes[t,e].cpu().numpy(); env=e.cpu().numpy()
    phase=(phases[t,e]/.2).floor().long().cpu().numpy()
    contact=(current[:,48]>0).long()+2*(current[:,49]>0).long()
    # Formal training uses the unchanged clean Kick task. Pushed recovery is
    # reserved for locked evaluation, so condition/sign are always 0 here.
    strata=np.stack([np.zeros(len(ref),int),phase,contact.cpu().numpy(),np.zeros(len(ref),int)],1)
    donor=np.full(len(ref),-1,dtype=np.int64)
    for row in np.unique(strata,axis=0):
        group=np.where(np.all(strata==row,axis=1))[0]
        order=group[np.argsort(ref[group],kind='stable')]
        _,counts=np.unique(ref[order],return_counts=True)
        largest=int(counts.max())
        if 2*largest>len(order):continue
        shifted=np.roll(order,largest)
        assert np.all(ref[order]!=ref[shifted])
        assert np.all((env[order]!=env[shifted])|(ep[order]!=ep[shifted]))
        donor[order]=shifted
    keep=np.where(donor>=0)[0]
    mapping=np.full(len(ref),-1,dtype=np.int64);mapping[keep]=np.arange(len(keep))
    matched_donor=mapping[donor[keep]]
    assert np.array_equal(np.sort(matched_donor),np.arange(len(keep)))
    idx=torch.as_tensor(keep,device=device)
    bundle={'history':history[idx],'actions':future[idx],
            'target':{k:v[idx] for k,v in targets.items()},
            'donor':torch.as_tensor(matched_donor,device=device)}
    audit={'sample_t':t[idx].cpu().numpy()+global_start,'sample_env':env[keep],
           'reference':ref[keep],'episode':ep[keep],'strata':strata[keep],
           'donor':matched_donor,'eligible_count':len(ref),'matched_count':len(keep),
           'common_aux_excluded':len(ref)-len(keep),'control_excluded_count':0}
    return bundle,audit

def auxiliary_step(model, optimizer, bundle, indices, shuffled):
    idx=indices; donor=bundle['donor'][idx] if shuffled else idx
    predictions=model.predict(bundle['history'][idx],bundle['actions'][idx])
    with torch.no_grad():
        normalized=(bundle['history']-model.input_mean)/model.input_std
        priors={k:torch.cat([normalized.flatten(1),bundle['actions'][:,:k].flatten(1),
                           torch.ones(len(normalized),1,device=normalized.device)],-1) @ getattr(model,'ridge_'+str(k))
                for k in HORIZONS}
    loss=0.
    for k in HORIZONS:
        delta=bundle['target'][k][donor,:48]
        if shuffled:delta=priors[k][idx,:48]+delta-priors[k][donor,:48]
        target=(delta-model.delta_mean)/model.delta_std
        loss=loss+(predictions[k][:,:48]-target).square().mean()
        loss=loss+.1*torch.nn.functional.binary_cross_entropy_with_logits(
            predictions[k][:,48:],bundle['target'][k][donor,48:])
    loss=loss/3
    optimizer.zero_grad();loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(v).all() for v in grads) and torch.isfinite(loss)
    gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    optimizer.step()
    return {'prediction_loss':float(loss.detach()),'gradient_norm':gn}

def feature_observation(arm, norm, obs, history, model):
    base=norm(obs)
    if arm=='P0':return base
    if arm=='P-cap':return torch.cat([base,obs[:,:INPUT]],-1)
    if arm=='P-history':return torch.cat([base,history.flatten(1)],-1)
    with torch.no_grad():latent=model.encode(history).detach()
    return torch.cat([base,latent],-1)

def gaussian_kl(old_mean,new_mean,std):
    # Encoder handoff changes only the mean; the same actor/std are frozen.
    return ((new_mean-old_mean).square()/(2*std.square())).sum(-1).mean()
