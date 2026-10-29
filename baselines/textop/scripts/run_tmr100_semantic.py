"""Future trained TMR mean embedding witnesses/all-seed diagnostic, no learning.

The policy remains disabled until a real100-epoch fresh CPU binding exists.
Original generated outputs are read only. Numerical/model failures abort with
all partials retained; no trial, physical failure, seed or annotation is dropped.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
import traceback
import zipfile
from tmr100_trained_acceptance import ROOT,PARENT,HOME,owned,read,sha,fingerprint
from tmr100_semantic_inputs import (accepted_frozen_verifier,accepted_diagnostic_metadata,
    accepted_semantic_witnesses,runtime_signature,validate_semantic_policy)
from ldm100k_generation_protocol import SEEDS,MODES


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);args=parser.parse_args()
    sp=owned(args.request);request=read(sp)
    pp=owned(request['policy_path']);assert sha(pp)==request['policy_sha256']
    s=validate_semantic_policy(read(pp))
    assert sha(__file__)==s['executor_sha256'] and os.getuid()==1013
    assert request['mode'] in ['witness','diagnostic'] and request['role'] in ['author','document','diagnostic']
    assert (request['mode']=='witness')==(request['role'] in ['author','document'])
    assert request['run']=={'author':'tmr100_semantic_author_v1','document':'tmr100_semantic_document_v1',
        'diagnostic':'tmr100_semantic32_v1'}[request['role']]
    assert request['runtime_binding_sha256']==runtime_signature(s)
    assert s['witnesses_authorized'] is True if request['mode']=='witness' else s['metrics_authorized'] is True
    if request['mode']=='diagnostic':accepted_semantic_witnesses(s['witness_bindings'],s)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and set(os.sched_getaffinity(0))=={0,1,2,3}
    for name,value in s['bound_sources'].items():assert sha(owned(ROOT/name))==value
    frozen=accepted_frozen_verifier(s['frozen_model_binding'])
    data=accepted_diagnostic_metadata(s['diagnostic_binding'])
    train=frozen['export']['training']
    assert train['binding']['input_signature']['normalization_sha256']==s['normalization_sha256']
    assert data['normalization_sha256']==s['normalization_sha256']
    out=owned(PARENT)/request['run'];out.mkdir(mode=0o700,exist_ok=False)
    started=time.monotonic()
    rec=dict(run=request['run'],mode=request['mode'],role=request['role'],source_sha256=sha(__file__),
        request_sha256=sha(sp),policy_sha256=sha(pp),runtime_binding_sha256=request['runtime_binding_sha256'],
        status='PREPARING_TRAINED_ROBOT_TMR100_SEMANTIC_INFERENCE',output_writers_closed=False,
        actual_completed_trials=0,training_optimizer_updates=0,generator_teacher_new_updates=0,
        formal_semantic_scores_computed=False,full_validation_claimed=False,continuous_transitions_claimed=False)

    def save():
        rec['elapsed_seconds']=time.monotonic()-started
        p=out/'receipt.partial'
        with p.open('w') as stream:
            json.dump(rec,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        p.replace(out/'receipt.json')

    save()
    try:
        item=train['spec']['inputs']['text_receipt'];tp=owned(item['path']);assert sha(tp)==item['sha256']
        text_receipt=read(tp);sys.path.insert(0,str(owned(text_receipt['runtime_dependency_dir'])))
        import numpy as np
        import torch
        from transformers import AutoModel,AutoTokenizer
        from tmr69d_core import make_model
        from generator_training_state import module_sha,equal
        from tmr69d_semantic_math import paired_seed_metrics,retrieval32
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65,0)
        assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>=20*1024**3
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(20261104);torch.cuda.manual_seed_all(20261104)
        infer=frozen['receipt']['inference'];infer_before=fingerprint(infer['path'],home=True)
        assert infer_before['sha256']==infer['sha256'] and infer_before['bytes']==infer['bytes']
        payload=torch.load(owned(infer['path'],HOME),map_location='cpu')
        assert payload['schema']=='textop.user_tmr100_inference.v1'
        assert payload['training_binding']==train['binding'] and payload['encoding']['sample_mean'] is True
        assert payload['encoding']['per_sample_L2_normalization'] is False
        tmr,unused_optimizer,_=make_model(owned(train['spec']['upstream_model_dir']),train['spec']['upstream_model_sha256'],'cpu')
        assert not unused_optimizer.state
        del unused_optimizer
        tmr.load_state_dict(payload['model'],strict=True)
        assert all(torch.isfinite(x).all() for x in tmr.state_dict().values())
        assert module_sha(tmr)==frozen['receipt']['trained_model_sha256']==payload['model_sha256']
        tmr=tmr.eval().requires_grad_(False).to('cuda:0')
        ip=train['spec']['inputs']['import_receipt'];ipath=owned(ip['path']);assert sha(ipath)==ip['sha256']
        imported=read(ipath)
        public_before={item['path']:fingerprint(item['path']) for item in imported['files']}
        assert sum(x['bytes'] for x in public_before.values())==707323471
        for item in imported['files']:
            assert public_before[item['path']]['sha256']==item['sha256']
        directory=owned(text_receipt['model_directories'][0])
        tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True,use_fast=True)
        hf=AutoModel.from_pretrained(str(directory),local_files_only=True,use_safetensors=True).eval().requires_grad_(False).to('cuda:0')
        assert all(torch.isfinite(x).all() for x in hf.state_dict().values())
        hf_before=module_sha(hf);tmr_before=module_sha(tmr)
        saved_rng=(torch.get_rng_state().clone(),[x.clone() for x in torch.cuda.get_rng_state_all()])
        segments=sorted(data['scope']['segments'],key=lambda x:x['segment_id'])
        if request['mode']=='witness':segments=segments[:1]
        seeds=SEEDS if request['mode']=='diagnostic' else SEEDS[:1]
        count=len(segments);assert count==(188 if request['mode']=='diagnostic' else 1)
        text_features=np.empty((count,256),dtype=np.float32)
        reference_features=np.empty((count,256),dtype=np.float32)
        generated=np.empty((2,len(seeds),count,256),dtype=np.float32)
        text_cache={};reference_hashes={};all_npz={};array_proofs=[];max_tokens=0

        def encode(value,modality):
            x=torch.from_numpy(np.array(value,dtype=np.float32,order='C',copy=True)).unsqueeze(0).to('cuda:0')
            mask=torch.ones((1,x.shape[1]),dtype=torch.bool,device='cuda:0')
            with torch.inference_mode():z=tmr.encode(dict(x=x,mask=mask),modality=modality,sample_mean=True)
            assert z.shape==(1,256) and torch.isfinite(z).all()
            return z[0].float().cpu().contiguous().numpy()

        for seg_index,seg in enumerate(segments):
            text=seg['text'];n=seg['future_frames'];assert 0<n<=200
            if text not in text_cache:
                tokens=tokenizer([text],padding=True,truncation=False,return_tensors='pt')
                length=int(tokens['attention_mask'].sum())
                assert length==tokens['input_ids'].shape[1]<=min(tokenizer.model_max_length,hf.config.max_position_embeddings)
                max_tokens=max(max_tokens,length)
                with torch.inference_mode():t=hf(**tokens.to('cuda:0')).last_hidden_state
                assert t.shape==(1,length,768) and torch.isfinite(t).all()
                text_cache[text]=encode(t[0].float().cpu().contiguous().numpy(),'text')
            text_features[seg_index]=text_cache[text]
            for seed_index,seed in enumerate(seeds):
                pair=data['finished'][(seg['segment_id'],seed)]
                assert pair['future_frames']==n and set(pair['mode_artifacts'])==set(MODES)
                for mode_index,mode in enumerate(MODES):
                    artifact=pair['mode_artifacts'][mode];path=owned(artifact['path'])
                    assert path.parent.parent==data['origin'] and path.name==mode+'.private.npz'
                    before=fingerprint(path);assert before['sha256']==artifact['sha256'] and before['bytes']==artifact['bytes']
                    full=((n+7)//8)*8
                    shapes=dict(generated_normalized_future=[full,69],reference_normalized_future=[n,69],
                        generated_qpos=[n,36],generated_contact=[n,2],initial_reference_history=[2,69])
                    assert artifact['array_shapes']==shapes
                    with zipfile.ZipFile(path) as archive:
                        assert archive.testzip() is None and set(archive.namelist())=={k+'.npy' for k in shapes}
                    with np.load(path,allow_pickle=False) as archive:
                        arrays={name:archive[name] for name in shapes}
                    for name,array in arrays.items():
                        assert list(array.shape)==shapes[name] and array.dtype==np.float32 and array.flags.c_contiguous and np.isfinite(array).all()
                        assert hashlib.sha256(array.tobytes()).hexdigest()==artifact['array_sha256'][name]
                    reference=arrays['reference_normalized_future']
                    rh=artifact['array_sha256']['reference_normalized_future']
                    if seg['segment_id'] in reference_hashes:assert reference_hashes[seg['segment_id']]==rh
                    else:
                        reference_features[seg_index]=encode(reference,'motion');reference_hashes[seg['segment_id']]=rh
                    # Do not include generated padding-to8 tail frames in semantic features.
                    generated[mode_index,seed_index,seg_index]=encode(arrays['generated_normalized_future'][:n],'motion')
                    assert fingerprint(path)==before
                    all_npz[str(path)]=before
                    array_proofs.append(dict(segment_id=seg['segment_id'],seed=seed,mode=mode,source_npz_sha256=artifact['sha256'],
                        scored_frames=n,generated_tail_frames_excluded=full-n))
                    rec['actual_completed_trials']+=1
                assert shutil.disk_usage(out).free>=s['reserve_bytes']+s['log_allowance_bytes']+s['output_upper_bytes']
            rec.update(status='RUNNING_TRAINED_TMR100_SEMANTIC_INFERENCE',actual_completed_segments=seg_index+1)
            save()
        assert rec['actual_completed_trials']==count*len(seeds)*2
        for value in [text_features,reference_features,generated]:assert np.isfinite(value).all()
        array_values=dict(text_mean_latents=text_features,reference_motion_mean_latents=reference_features,
                          generated_motion_mean_latents=generated)
        artifact=out/'raw_mean_latents.private.npz'
        with artifact.open('xb') as stream:
            os.chmod(artifact,0o600);np.savez(stream,**array_values);stream.flush();os.fsync(stream.fileno())
        assert artifact.stat().st_size<=s['output_upper_bytes']
        private=out/'source_join.private.json'
        with private.open('x') as stream:
            os.chmod(private,0o600);json.dump(dict(segment_ids=[x['segment_id'] for x in segments],seeds=seeds,modes=MODES,
                array_sources=array_proofs),stream,indent=2);stream.flush();os.fsync(stream.fileno())
        assert sum(p.stat().st_size for p in [artifact,private])<=s['output_upper_bytes']
        assert shutil.disk_usage(out).free>=s['reserve_bytes']+s['log_allowance_bytes']
        metrics=None;summary=None;baseline=None
        if request['mode']=='diagnostic':
            ids=[x['segment_id'] for x in segments]
            metrics={mode:{str(seed):paired_seed_metrics(text_features,reference_features,generated[mi,si],ids)
                for si,seed in enumerate(seeds)} for mi,mode in enumerate(MODES)}
            baseline=retrieval32(text_features,reference_features,ids)
            summary={}
            for mode in MODES:
                rows=metrics[mode]
                scalars={name:[rows[str(seed)]['retrieval'][name] for seed in seeds] for name in ['Rprecision_top1','Rprecision_top2','Rprecision_top3','MMdist']}
                scalars['FID']=[rows[str(seed)]['frechet']['FID'] for seed in seeds]
                summary[mode]={name:dict(equal_seed_mean=float(np.mean(values)),descriptive_seed_population_std=float(np.std(values,ddof=0)),
                    all_five_seed_values=values) for name,values in scalars.items()}
        assert module_sha(tmr)==tmr_before and module_sha(hf)==hf_before
        assert equal(saved_rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all())), 'Mean inference must not consume RNG'
        assert fingerprint(infer['path'],home=True)==infer_before
        for path,before in public_before.items():assert fingerprint(path)==before
        for path,before in all_npz.items():assert fingerprint(path)==before
        assert sha(data['journal'])==s['diagnostic_binding']['private_journal_sha256']
        rec.update(status='TMR100_SEMANTIC_DIAGNOSTIC_FINISHED' if request['mode']=='diagnostic' else 'TMR100_REAL_SEMANTIC_EMBEDDING_WITNESS_FINISHED',
            output_writers_closed=True,formal_semantic_scores_computed=request['mode']=='diagnostic',per_seed_metrics=metrics,
            equal_seed_summary=summary,reference_motion_retrieval_baseline=baseline,
            all_seed_mode_and_physics_failure_outputs_retained=True,raw_latent_dimensions=256,sample_mean=True,
            per_sample_L2_normalization=False,new_model_loads=2,new_gpu_contexts=1,
            source_model_and_arrays_unchanged=True,maximum_actual_text_tokens=max_tokens,text_truncation_performed=False,
            stochastic_sampling_or_dropout_used=False,inference_rng_unchanged=True,
            mean_latent_arrays={name:dict(shape=list(value.shape),dtype='float32',sha256=hashlib.sha256(value.tobytes()).hexdigest()) for name,value in array_values.items()},
            private_artifacts={p.name:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in [artifact,private]},
            actual_segments=count,actual_seeds=seeds,actual_modes=MODES,actual_original_generated_npz_bytes_checked=sum(x['bytes'] for x in all_npz.values()),
            repeated_exact_text_segment_count=count-len(text_cache),independent_segment_protocol=True,
            reference_feature_tail_policy='all and only original scored future frames, no seed/history frames',
            semantic_quality_or_paper_complete_claimed=False)
        save()
    except BaseException as error:
        rec.update(status='FAILED_TMR100_SEMANTIC_ALL_OUTPUTS_RETAINED',error=repr(error),traceback=traceback.format_exc())
        save();raise


if __name__=='__main__':main()
