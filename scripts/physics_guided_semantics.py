"""Independent released HumanML3D text-motion evaluator, not a paid VLM.

This is a learned matching/diversity proxy. No human semantic truth is claimed.
Failed trajectories remain as last-pose-padded examples in the denominator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from closd.utils.rep_util import RepresentationHandler
from closd.diffusion_planner.data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from closd.diffusion_planner.data_loaders.humanml.networks import evaluator_wrapper
from closd.diffusion_planner.data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from physics_guided_lineage import file_hash, object_hash

ROOT=Path(__file__).resolve().parents[1]


def main(out):
    start=time.time(); manifest=json.loads((out/'source_manifest.json').read_text())
    snapshot=Path(manifest['snapshot']); source=ROOT/'third_party/CLoSD/closd/diffusion_planner'
    mean=torch.as_tensor(np.load(manifest['mean']),dtype=torch.float32)
    std=torch.as_tensor(np.load(manifest['std']),dtype=torch.float32)
    eval_mean=torch.as_tensor(np.load(source/'dataset/t2m_mean.npy'),dtype=torch.float32)
    eval_std=torch.as_tensor(np.load(source/'dataset/t2m_std.npy'),dtype=torch.float32)
    assert torch.all(eval_std>0)
    vectorizer=WordVectorizer(str(snapshot/'glove'),'our_vab')
    # Upstream resolves an unpinned HF "latest" snapshot internally. Override
    # only its dependency resolver during construction, without modifying
    # upstream files, so the checkpoint actually loaded is the recorded one.
    evaluator_path=snapshot/'evaluators/t2m/text_mot_match/model/finest.tar'
    frozen_versions=json.loads((out/'versions.json').read_text())
    assert file_hash(evaluator_path)==frozen_versions['text_motion_evaluator_weights_sha256']
    resolved_dependencies=[]
    original_resolver=evaluator_wrapper.hf_handler.get_dependencies
    def frozen_dependencies():
        assert snapshot.is_dir()
        resolved_dependencies.append(str(snapshot.resolve()))
        return str(snapshot.resolve())
    evaluator_wrapper.hf_handler.get_dependencies=frozen_dependencies
    try:
        evaluator=EvaluatorMDMWrapper('humanml',torch.device('cuda'))
    finally:
        evaluator_wrapper.hf_handler.get_dependencies=original_resolver
    assert resolved_dependencies==[str(snapshot.resolve())]
    rep=RepresentationHandler(mean,std,None,time_prints=False)
    arms=json.loads((out/'arms.json').read_text()); results={}; vectors={}; case_order=None
    for key,path in arms.items():
        arm,seed=key.split(':'); directory=Path(path)
        meta=json.loads((directory/'metadata.json').read_text())
        if case_order is None: case_order=object_hash(meta['cases'])
        else: assert object_hash(meta['cases'])==case_order, 'Semantic arm case/text order changed'
        arrays=np.load(directory/'trajectories.npz',allow_pickle=False)
        pose=arrays['actual'].transpose(1,0,2,3).copy(); lengths=arrays['valid'].sum(0)
        for i,length in enumerate(lengths):
            assert length>0
            pose[i,int(length):]=pose[i,int(length)-1]
        assert len(pose)==144 and pose.shape[1:]==(120,24,3)
        embeddings=[]; distances=[]
        for begin in range(0,len(pose),32):
            subset=meta['cases'][begin:begin+32]; n=len(subset)
            motion,_,_=rep.pose_to_hml(torch.as_tensor(pose[begin:begin+n],device='cuda'),fix_ik_bug=True)
            motion=((motion.cpu()*std+mean)-eval_mean)/eval_std
            assert motion.shape[1:]==(80,263) and torch.isfinite(motion).all()
            words=[]; pos=[]; cap_lens=[]
            for case in subset:
                tokens=case['tokens'].split('_')[:20]
                tokens=['sos/OTHER']+tokens+['eos/OTHER']; cap_lens.append(len(tokens))
                tokens+=['unk/OTHER']*(22-len(tokens))
                pairs=[vectorizer[token] for token in tokens]
                words.append(np.stack([pair[0] for pair in pairs])); pos.append(np.stack([pair[1] for pair in pairs]))
            # Text GRU packs sorted captions. Explicitly restore both sorting operations.
            order=np.argsort(cap_lens)[::-1].copy(); inverse=np.argsort(order)
            m_lens=torch.full((n,),80,dtype=torch.long)
            with torch.no_grad():
                text,motion_emb=evaluator.get_co_embeddings(torch.as_tensor(np.array(words)[order]),
                    torch.as_tensor(np.array(pos)[order]),torch.as_tensor(np.array(cap_lens)[order]),
                    motion[order],m_lens)
            motion_order=np.argsort(m_lens.numpy())[::-1].copy(); undo_motion=np.argsort(motion_order)
            text=text[undo_motion][inverse].cpu().numpy(); motion_emb=motion_emb[undo_motion][inverse].cpu().numpy()
            embeddings.append(motion_emb); distances.extend(np.linalg.norm(text-motion_emb,axis=1).tolist())
        embeddings=np.concatenate(embeddings)
        vectors[key]=embeddings
        rng=np.random.RandomState(20261011); diversity=[]
        for family in ('jump','dance'):
            indices=[i for i,c in enumerate(meta['cases']) if c['family']==family]
            left=rng.choice(indices,1000); right=rng.choice(indices,1000)
            distinct=np.array([meta['cases'][i]['source_id']!=meta['cases'][j]['source_id'] for i,j in zip(left,right)])
            diversity.extend(np.linalg.norm(embeddings[left[distinct]]-embeddings[right[distinct]],axis=1).tolist())
        results[key]=dict(matching_distance=float(np.mean(distances)),diversity=float(np.mean(diversity)),
                          attempts=len(pose),completed=int((lengths==120).sum()),
                          evaluation='released text-motion embedding proxy, all attempts with failures padded')
        arrays.close()
    combined={}
    for arm in {key.split(':')[0] for key in results}:
        combined[arm]={metric:float(np.mean([results[arm+':'+str(s)][metric] for s in (11,12,13)]))
                       for metric in ('matching_distance','diversity')}
        stacked=np.stack([vectors[arm+':'+str(s)] for s in (11,12,13)])
        combined[arm]['multimodality']=float(np.mean([np.linalg.norm(stacked[i]-stacked[j],axis=1).mean()
                                                     for i,j in ((0,1),(0,2),(1,2))]))
    pre=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None
    baseline=pre['gate']['strongest_equal_budget_baseline'] if pre else 'native_best'
    config=json.loads((ROOT/'configs/physics_guided_20261011.json').read_text())['base_thresholds']
    checks=dict(matching=combined['proposed']['matching_distance']<=combined[baseline]['matching_distance']*(1+config['max_semantic_regression_fraction']),
                diversity=combined['proposed']['diversity']>=combined[baseline]['diversity']*(1-config['max_diversity_regression_fraction']),
                multimodality=combined['proposed']['multimodality']>=combined[baseline]['multimodality']*(1-config['max_diversity_regression_fraction']))
    result=dict(status='pass' if all(checks.values()) else 'fail',checks=checks,baseline=baseline,
                per_arm=combined,per_seed=results,evaluation_type='independent learned semantic/diversity proxy',
                elapsed_seconds=time.time()-start,weights_frozen=True,no_paid_api=True)
    result['input_hashes']={key:{name:file_hash(Path(path)/name) for name in ('trajectories.npz','metadata.json','metrics.json')}
                            for key,path in arms.items()}
    result['arms_sha256']=file_hash(out/'arms.json')
    result['evaluator_weights_sha256']=file_hash(evaluator_path)
    result['actually_loaded_evaluator_path']=str(evaluator_path.resolve())
    result['resolved_dependency_snapshots']=resolved_dependencies
    assert result['evaluator_weights_sha256']==frozen_versions['text_motion_evaluator_weights_sha256']
    result['semantic_asset_hashes']=json.loads((out/'versions.json').read_text())['external_semantic_inputs']
    for name,digest in result['semantic_asset_hashes'].items(): assert file_hash(name)==digest
    (out/'semantic_diversity_gate.json').write_text(json.dumps(result,indent=2))
    print('PHYSICS_GUIDED_SEMANTIC_PROXY_COMPLETE',result['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
    main(Path(a.output).resolve())
