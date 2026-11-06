"""Build CPU-only, source-bound three-stage caches for NEW repaired generation.

Uses the same original HumanML matching networks, fixed weights/statistics,
project586 test pool, sign-fixed feature encoder and pinned geometric metrics.
Does not regenerate motion, evaluate policies, or silently re-score the old library.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='2'
import numpy as np
import eval_protocol_v2 as protocol

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
UPSTREAM=ROOT.parent/'conditional-human-motion-generation/third_party/DreamControl/TrajGen'
GROUND_TRUTH=ROOT/'grab_omni_29dof_weak_baseline_v1_20261103/manifest.json'
OLD_GENERATION_SHA='6d35b617780968b45ef9d05c2014df17708d6b45bb7158438a13ec6c49df36cf'
OLD_MODEL_SHA='109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9'
ENCODER_SOURCES={'feature263.py':'b66b6779f02e84b1d6e3834ccdd18ea0119cc4f9ab1096fea21363524544c53e',
                 'feature263_signfix.py':'fe85e96dde630ebf2cdf8d621ee129648b954c760c042002941d02b568b8f203'}
UPSTREAM_SOURCES={
 'data_loaders/humanml/networks/modules.py':'ba8511bb338e563de28df3e9423fca71763903dd0e8f9d12d915f4983b6cac57',
 'data_loaders/humanml/scripts/motion_process.py':'4909a03c388e965a67b2304b5f2d95d9c994a41677fb206f5c2632d36da80eda',
 'data_loaders/humanml/common/skeleton.py':'5b047d93ca883a84d972a8e743d33a68089a990e3553f18eb7752b3ce4000776',
 'data_loaders/humanml/common/quaternion.py':'7ec13fc8497de6249f4203ed4cd53aaa8f6c6992bc166b066d5561eaa0639802',
 'data_loaders/humanml/utils/paramUtil.py':'fe0265f3dcd0d466690d816eae973fcece145f64551cef7e46fed7ef3de80d34',
 'data_loaders/humanml/utils/word_vectorizer.py':'64dfc2eeaebb04f0c0f9a9b6526d5416edee248b4e1d10f617bcb07305188803',
 'data_loaders/humanml/networks/evaluator_wrapper.py':protocol.ENCODER_FIXED['wrapper_sha256'],
 'data_loaders/humanml/utils/metrics.py':protocol.METRICS_SHA}
BATCH=32
CAP=protocol.ENCODER_FIXED['max_feature_frames']

def require(test,message):
    if not test:raise ValueError(message)

def sha(path):return protocol.sha_file(path)

def checked_json(path,expected=None):
    raw=Path(path).read_bytes();value=hashlib.sha256(raw).hexdigest()
    if expected is not None:require(value==expected,'JSON content SHA mismatch: '+str(path))
    return json.loads(raw),value

def checked_npz(path,expected):
    protocol.require_sha(expected,'NPZ artifact')
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'NPZ content SHA mismatch: '+str(path))
    # Decode precisely the verified immutable bytes, rather than reopening a path.
    return np.load(io.BytesIO(raw),allow_pickle=False)

def write_json(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def source_path(value,base):
    p=Path(value);return (base/p).resolve(strict=True) if not p.is_absolute() else p.resolve(strict=True)

def unpack_raw(row,base):
    path=source_path(row['source_sample'],base)
    with checked_npz(path,row['source_sample_sha256']) as data:
        meta=json.loads(data['metadata_utf8'].tobytes())
        frames=int(meta['frames']);features=np.asarray(data['features'])
        points=np.asarray(data['canonical_points']);target=np.asarray(data['target_points']);mask=np.asarray(data['target_mask'])
        require(features.shape in ((1,frames,263),(frames,263)),'Unexpected raw feature shape')
        require(points.shape in ((1,frames,22,3),(frames,22,3)),'Unexpected raw point shape')
        features=features.reshape(frames,263).copy();points=points.reshape(frames,22,3).copy();target=target.copy();mask=mask.copy()
    require(frames>=5 and target.shape==(frames,22,3) and mask.shape==(frames,22) and mask.dtype==np.bool_ and mask.any(),'Invalid raw targets/mask/length')
    require(np.isfinite(features).all() and np.isfinite(points).all() and np.isfinite(target).all(),'Nonfinite raw artifact')
    require(meta['task']==row['task'] and str(meta['candidate_id'])==str(row['candidate_id']),'Raw task/candidate ID mismatch')
    require(meta['caption']==row['caption'],'Evaluation caption differs from actual generation condition')
    require(float(meta['fps'])==20 and float(row.get('source_fps',20))==20,'Only fixed 20Hz source supported')
    require(int(row.get('source_frames',frames))==frames,'Generation/frame-count mismatch')
    feature_hash=protocol.array_sha(features)
    require(feature_hash==meta['feature_content_sha256'],'Actual raw feature content SHA mismatch')
    require(row.get('feature_content_sha256',feature_hash)==feature_hash,'Manifest/raw feature SHA mismatch')
    noise_hash=meta['initial_noise_sha256'];protocol.require_sha(noise_hash,'raw metadata initial noise')
    require(row.get('initial_noise_sha256',noise_hash)==noise_hash,'Manifest/raw noise provenance mismatch')
    model_hash=meta['checkpoint_sha256'];protocol.require_sha(model_hash,'generation checkpoint')
    require(model_hash!=OLD_MODEL_SHA,'This builder is for the newly trained repaired prior, not the old completed library')
    rotation=np.asarray(meta['canonical_to_world_rotation'],dtype=np.float64);translation=np.asarray(meta['canonical_to_world_translation'],dtype=np.float64)
    require(rotation.shape==(3,3) and translation.shape==(3,) and np.isfinite(rotation).all() and np.isfinite(translation).all(),'Invalid canonical/world transform')
    require(np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6,rtol=0) and np.linalg.det(rotation)>0 and np.allclose(rotation[:,1],[0,0,1],atol=1e-6,rtol=0),'Expected proper canonical Y-up to world Z-up transform')
    world=points@rotation.T+translation
    require(np.allclose((world-translation)@rotation,points,atol=1e-6,rtol=0),'Coordinate round-trip mismatch')
    return {'path':path,'metadata':meta,'frames':frames,'features':features,'points':points,'world_points':world,'target':target,'mask':mask,'rotation':rotation,'translation':translation,'noise_sha':noise_hash,'feature_sha':feature_hash,'model_sha':model_hash}

def geometry_values(points,target,mask,source_sha,metrics,torch):
    error=np.asarray(metrics.control_l2(points[None],target[None],mask[None,...,None]))[0]
    require(error.shape==mask.shape and np.allclose(error[mask],np.linalg.norm(points[mask]-target[mask],axis=-1),atol=1e-6,rtol=1e-5),'Original Control L2/mask mismatch')
    skating,_=metrics.calculate_skating_ratio(torch.from_numpy(np.ascontiguousarray(points.transpose(1,2,0)[None])))
    value=float(skating[0]);require(np.isfinite(value) and 0<=value<=1,'Invalid original skating ratio')
    return {'source_artifact_sha256':source_sha,'control_l2_m':float(error[mask].mean()),'control_length_unit':'m','skating_ratio':value,'skating_scale':'fraction_0_1'}

def load_evaluator(assets,encoder_dir):
    for rel,value in UPSTREAM_SOURCES.items():require(sha(UPSTREAM/rel)==value,'Pinned upstream source changed: '+rel)
    for rel,value in ENCODER_SOURCES.items():require(sha(encoder_dir/rel)==value,'Pinned feature encoder source changed: '+rel)
    sys.path.insert(0,str(UPSTREAM));sys.path.insert(0,str(encoder_dir))
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    from safetensors.torch import load_file
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.networks.modules import MovementConvEncoder,MotionEncoderBiGRUCo,TextEncoderBiGRUCo
    from data_loaders.humanml.utils.word_vectorizer import POS_enumerator,WordVectorizer
    from data_loaders.humanml.utils import metrics
    import feature263_signfix
    require(Path(feature263_signfix.__file__).resolve()==(encoder_dir/'feature263_signfix.py').resolve(),'Wrong feature encoder import path')
    require(sha(assets/'model.safetensors')==protocol.ENCODER_FIXED['weights_sha256'],'Matching-network weights changed')
    receipt,_=checked_json(assets/'source_assets.json')
    for entry in receipt:
        p=(assets/entry['member']).resolve(strict=True);require(assets in p.parents,'Asset receipt escapes evaluator directory')
        require(p.stat().st_size==entry['size'] and sha(p)==entry['sha256'],'Evaluator source asset changed: '+entry['member'])
    mean,std=[np.load(assets/'stats'/name,allow_pickle=False) for name in ('mean.npy','std.npy')]
    for name,expected in [('mean',protocol.ENCODER_FIXED['mean_sha256']),('std',protocol.ENCODER_FIXED['std_sha256'])]:
        require(sha(assets/'stats'/(name+'.npy'))==expected and sha(UPSTREAM/'dataset'/('t2m_'+name+'.npy'))==expected,'Original matching-network normalization changed')
    require(mean.shape==std.shape==(263,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),'Invalid normalization arrays')
    state=load_file(assets/'model.safetensors',device='cpu');evaluator=EvaluatorMDMWrapper.__new__(EvaluatorMDMWrapper);evaluator.device='cpu';evaluator.opt={'unit_length':4}
    evaluator.movement_encoder=MovementConvEncoder(259,512,512)
    evaluator.motion_encoder=MotionEncoderBiGRUCo(input_size=512,hidden_size=1024,output_size=512,device='cpu')
    evaluator.text_encoder=TextEncoderBiGRUCo(word_size=300,pos_size=len(POS_enumerator),hidden_size=512,output_size=512,device='cpu')
    used=set()
    for name in ('movement_encoder','motion_encoder','text_encoder'):
        prefix=name+'.';subset={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)};module=getattr(evaluator,name);module.load_state_dict(subset,strict=True);module.eval().requires_grad_(False);used.update(prefix+key for key in subset)
    require(used==set(state) and len(used)==42,'Original matching-network state coverage mismatch');del state
    vectorizer=WordVectorizer(str(assets/'glove'),'our_vab')
    return evaluator,mean,std,vectorizer,feature263_signfix.encode_signfixed,metrics,torch,receipt

def embed_batch(batch,evaluator,mean,std,torch):
    require(bool(batch),'Empty embedding batch')
    lengths=np.asarray([min(len(x),CAP) for x in batch],dtype=np.int64)
    require(lengths.min()>=4,'Encoder requires at least four feature frames')
    samples=np.zeros((len(batch),CAP,263),dtype=np.float32)
    for i,features in enumerate(batch):
        require(features.ndim==2 and features.shape[1]==263 and np.isfinite(features).all(),'Invalid 263D encoder input')
        samples[i,:lengths[i]]=(features[:lengths[i]]-mean)/std
    order=np.argsort(lengths)[::-1].copy()
    with torch.inference_mode():values=evaluator.get_motion_embeddings(torch.from_numpy(samples),torch.from_numpy(lengths)).cpu().numpy()[np.argsort(order)]
    require(values.shape==(len(batch),512) and np.isfinite(values).all(),'Invalid original unscaled motion embeddings')
    return np.asarray(values,dtype=np.float32)

def embed_texts(rows,tokens,evaluator,vectorizer,torch):
    captions=list(dict.fromkeys(row['caption'] for row in rows));word_pos=[];lengths=[]
    for caption in captions:
        require(caption in tokens,'Actual conditioning caption missing from supplied token file')
        raw=tokens[caption];require(isinstance(raw,list) and all(isinstance(t,str) and '/' in t for t in raw),'Expected original token/POS strings')
        seq=['sos/OTHER',*raw[:20],'eos/OTHER'];lengths.append(len(seq));word_pos.append([vectorizer[t] for t in seq+['unk/OTHER']*(22-len(seq))])
    order=np.argsort(np.asarray(lengths))[::-1].copy();words=torch.tensor(np.asarray([[x[0] for x in word_pos[i]] for i in order]),dtype=torch.float32);poses=torch.tensor(np.asarray([[x[1] for x in word_pos[i]] for i in order]),dtype=torch.float32)
    with torch.inference_mode():unique=evaluator.text_encoder(words,poses,torch.tensor(np.asarray(lengths)[order])).cpu().numpy()[np.argsort(order)]
    require(unique.shape==(len(captions),512) and np.isfinite(unique).all(),'Invalid original text embeddings')
    lookup=dict(zip(captions,unique));return np.asarray([lookup[row['caption']] for row in rows],dtype=np.float32)

def build(args):
    require(args.generation_sha256!=OLD_GENERATION_SHA,'Do not re-evaluate the old completed library')
    generation,generation_sha=checked_json(args.generation_manifest,args.generation_sha256);rows=generation['records'];base=args.generation_manifest.parent
    require(bool(rows),'New generation manifest has no candidates');ids=[]
    for row in rows:
        require(row['task'] in protocol.TASKS and isinstance(row['admitted'],bool) and isinstance(row['selected_reference'],bool),'Invalid task/admission/selection fields')
        require(not row['selected_reference'] or row['admitted'],'Selected reference lacks actual admission')
        require(isinstance(row['caption'],str) and bool(row['caption'].strip()),'Actual generation caption required')
        row.setdefault('id',row['task']+':'+str(row['candidate_id']));ids.append(row['id'])
        require(row.get('retrieval_group_id',row['task'])==row['task'],'Do not fabricate semantic groups')
        if row['admitted']:require(bool(row.get('source')) and bool(row.get('source_sha256')),'Admitted candidate missing final robot artifact')
    require(len(ids)==len(set(ids)) and all(isinstance(i,str) and i for i in ids),'Duplicate/invalid candidate IDs')
    tokens,tokens_sha=checked_json(args.tokens);require(isinstance(tokens.get('tokens'),dict),'Expected token/POS mapping')
    ground,_=checked_json(GROUND_TRUTH,protocol.POOL_SHA);test=[r for r in ground['records'] if r['split']=='test'];require(len(test)==586,'Held-out pool count changed')
    for key in ('group_id','source_id','feature_sha256'):
        require(not ({r[key] for r in test}&{r[key] for r in ground['records'] if r['split']!='test'}),'Held-out source split overlap: '+key)
    evaluator,mean,std,vectorizer,encode,metrics,torch,asset_receipt=load_evaluator(args.assets.resolve(strict=True),args.encoder_dir.resolve(strict=True))
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();context={'stage':'reference_pool','candidate_id':None}
    try:
        reference_chunks=[];batch=[];reference_lengths=[]
        for row in test:
            with checked_npz(source_path(row['feature_path'],ROOT),row['feature_sha256']) as item:feat=item['features'].copy()
            reference_lengths.append(len(feat));batch.append(feat)
            if len(batch)==BATCH:reference_chunks.append(embed_batch(batch,evaluator,mean,std,torch));batch=[]
        if batch:reference_chunks.append(embed_batch(batch,evaluator,mean,std,torch))
        arrays={'reference':np.concatenate(reference_chunks),'text':embed_texts(rows,tokens['tokens'],evaluator,vectorizer,torch)}
        queued={s:[] for s in protocol.STAGES};chunks={s:[] for s in protocol.STAGES};stage_ids={s:[] for s in protocol.STAGES};stage_sources={s:[] for s in protocol.STAGES};stage_lengths={s:[] for s in protocol.STAGES};records=[];geo_rows=[];models=set()
        def append(stage,row,feature,source_hash):
            queued[stage].append(feature);stage_ids[stage].append(row['id']);stage_sources[stage].append(source_hash);stage_lengths[stage].append(len(feature))
            if len(queued[stage])==BATCH:chunks[stage].append(embed_batch(queued[stage],evaluator,mean,std,torch));queued[stage]=[]
        for index,row in enumerate(rows):
            context.update(stage='raw_and_roundtrip',candidate_id=row['id']);raw=unpack_raw(row,base);models.add(raw['model_sha'])
            sample_sha=row['source_sample_sha256'];append('raw_diffusion',row,raw['features'],sample_sha);roundtrip=encode(raw['world_points'],foot_velocity_squared_threshold=.002);append('raw_reencoded',row,roundtrip,sample_sha)
            values=geometry_values(raw['points'],raw['target'],raw['mask'],sample_sha,metrics,torch)
            geo={'id':row['id'],'source_sample_sha256':sample_sha,'stages':{'raw_diffusion':values,'raw_reencoded':dict(values)}}
            if row.get('source'):
                context['stage']='postoptimized';require(bool(row.get('source_sha256')),'Postoptimized source lacks SHA')
                with checked_npz(source_path(row['source'],base),row['source_sha256']) as final:
                    world=final['keypoints'].copy();require(float(final['fps'])==20,'Postoptimized timebase changed')
                require(world.shape==(raw['frames'],22,3) and np.isfinite(world).all(),'Postoptimized point shape/nonfinite mismatch')
                append('postoptimized',row,encode(world,foot_velocity_squared_threshold=.002),row['source_sha256'])
                canonical=(world-raw['translation'])@raw['rotation'];geo['stages']['postoptimized']=geometry_values(canonical,raw['target'],raw['mask'],row['source_sha256'],metrics,torch)
            records.append({'id':row['id'],'task':row['task'],'candidate_id':row['candidate_id'],'admitted':row['admitted'],'selected_reference':row['selected_reference'],'caption':row['caption'],'caption_role':'actual_generation_condition','retrieval_group_id':row['task'],'source_sample':str(raw['path']),'source_sample_sha256':sample_sha,'initial_noise_sha256':raw['noise_sha'],'feature_content_sha256':raw['feature_sha'],'generation_checkpoint_sha256':raw['model_sha'],'source_frames':raw['frames'],'source_fps':20,'postoptimized_source':row.get('source'),'postoptimized_source_sha256':row.get('source_sha256')});geo_rows.append(geo)
            if (index+1)%32==0:print(json.dumps({'status':'building_new_generation_caches','candidates_completed':index+1,'total':len(rows),'CPU_threads':2}),flush=True)
        require(len(models)==1,'New generation manifest mixes different diffusion checkpoints')
        for stage in protocol.STAGES:
            if queued[stage]:chunks[stage].append(embed_batch(queued[stage],evaluator,mean,std,torch))
            arrays[stage]=np.concatenate(chunks[stage]) if chunks[stage] else np.empty((0,512),dtype=np.float32)
        encoder={**protocol.ENCODER_FIXED,'feature_encoder_sources':ENCODER_SOURCES,'upstream_source_sha256':UPSTREAM_SOURCES,'foot_velocity_squared_threshold':.002};fingerprint=protocol.content_sha(encoder)
        def spec(key):return {'embedding_key':key,'embedding_content_sha256':protocol.array_sha(arrays[key]),'encoder_fingerprint_sha256':fingerprint}
        manifest={'schema':'generated_reference_eval_inputs_v2','retrieval_group_definition':'authorized_four_task_ids','generation_manifest_sha256':generation_sha,'generation_checkpoint_sha256':next(iter(models)),'encoder':encoder,'encoder_fingerprint_sha256':fingerprint,'reference_pool':{'name':'project586','split':'test','count':586,'source_manifest_sha256':protocol.POOL_SHA,**spec('reference')},'records':records,'matched_text':{'ids':ids,**spec('text')},'stages':{stage:{'ids':stage_ids[stage],'source_artifact_sha256':stage_sources[stage],**spec(stage)} for stage in protocol.STAGES},'legacy_retrieval':None}
        geometry={'schema':'generated_reference_geometry_cache_v2','generation_manifest_sha256':generation_sha,'upstream_metrics_sha256':protocol.METRICS_SHA,'records':geo_rows,'units':{'Control_L2':'metres','Skating_Ratio':'fraction_0_1'},'raw_reencoded_geometry':'identical raw physical points/targets; only embedding representation is reencoded'}
        protocol.validate_inputs(manifest,arrays,geometry)
        require(sha(args.generation_manifest)==generation_sha and sha(args.tokens)==tokens_sha,'Generation manifest or token file changed during evaluation')
        np.savez_compressed(args.output/'embeddings.npz',**arrays);write_json(args.output/'manifest.json',manifest);write_json(args.output/'geometry.json',geometry)
        provenance={'status':'new_generation_caches_completed_report_pending','generation_manifest':str(args.generation_manifest.resolve()),'generation_manifest_sha256':generation_sha,'builder_source_sha256':sha(__file__),'protocol_source_sha256':sha(Path(protocol.__file__)),'tokens_sha256':tokens_sha,'tokenizer':tokens.get('tokenizer'),'asset_receipt':asset_receipt,'CPU_threads':2,'GPU_used':False,'all_generated_count':len(records),'admitted_count':sum(r['admitted'] for r in records),'selected_reference_count':sum(r['selected_reference'] for r in records),'task_counts':dict(Counter(r['task'] for r in records)),'selected_task_counts':dict(Counter(r['task'] for r in records if r['selected_reference'])),'stage_available_counts':{s:len(stage_ids[s]) for s in protocol.STAGES},'stage_feature_lengths':stage_lengths,'reference_feature_lengths':reference_lengths,'truncated_at_196':{s:sum(n>CAP for n in stage_lengths[s]) for s in protocol.STAGES},'reference_truncated_at_196':sum(n>CAP for n in reference_lengths),'input_files':{name:{'path':str(args.output/name),'sha256':sha(args.output/name)} for name in ('manifest.json','embeddings.npz','geometry.json')},'legacy_project_r32':'N/A: no exact prior plan invented','paired_batch_r32':'N/A for only four actual task semantic groups','r_at_4':'separate four-task-text diagnostic when actual task captions are fixed','paper_fullmix_comparable':False,'elapsed_seconds':time.monotonic()-started}
        write_json(args.output/'cache_build_receipt.json',provenance)
        command=[sys.executable,str(Path(protocol.__file__)), '--metrics-source',str(UPSTREAM/'data_loaders/humanml/utils/metrics.py'),'--metrics-source-sha256',protocol.METRICS_SHA,'--output',str(args.output/'report.json')]
        for name in ('manifest','embeddings','geometry'):
            file=args.output/({'manifest':'manifest.json','embeddings':'embeddings.npz','geometry':'geometry.json'}[name]);command.extend(['--'+name,str(file),'--'+name+'-sha256',sha(file)])
        subprocess.run(command,check=True)
        write_json(args.output/'completion_receipt.json',{'status':'new_generation_three_stage_metrics_completed','generation_manifest_sha256':generation_sha,'cache_build_receipt_sha256':sha(args.output/'cache_build_receipt.json'),'report_sha256':sha(args.output/'report.json'),'old_results_untouched':True,'GPU_used':False,'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        write_json(args.output/'failure.json',{'status':'failed_preserved','context':context,'type':type(exc).__name__,'error':str(exc),'generation_manifest_sha256':generation_sha,'old_results_untouched':True});raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generation-manifest',type=Path,required=True);p.add_argument('--generation-sha256',required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--tokens',type=Path,required=True);p.add_argument('--encoder-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();build(args)

if __name__=='__main__':main()
