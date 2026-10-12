import pathlib,json,hashlib,numpy as np
root=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
out=root/'results/task_matched_feedback_v2_20261012'
def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
def obj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require(c,message):
    if not c: raise RuntimeError(message)
def motion_hash(p):
    h=hashlib.sha256()
    with np.load(p,allow_pickle=False) as a:
        for key in sorted(a.files):
            x=np.ascontiguousarray(a[key]);h.update(key.encode());h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()
v=read(out/'versions.json'); vh=sha(out/'versions.json')
require(vh=='812e1c56258da2d736205d60f1a2b4f5263b8734b8dc23dde89999ee14042c53','Versions changed')
runtime=read(out/'runtime_witness.json')
require(runtime['status']=='PASS' and runtime['versions_sha256']==vh,'Runtime witness mismatch')
labels=['val_shared_s11_r0','witness_zero_s11','witness_raw_replay_s11']
ledger=read(out/'budget_ledger.json')
require([r['label'] for r in ledger]==labels and all(r['exit_code']==0 and r['stage']=='measurement' for r in ledger),'Actual charged batches')
require(sum(r['seconds'] for r in ledger)<=1800 and all(r['seconds']<=180 for r in ledger),'Frozen measurement caps')
manifest=read(out/'source_manifest.json'); config=read(root/'configs/physics_guided_20261011.json')
expected_cases=[]
for r in manifest['rows']:
    if r['split']=='val':
        for offset in config['offsets_m']:
            expected_cases.append(dict(r,offset_m=offset,goal_joint_body='Pelvis' if r['goal_joint']=='pelvis' else 'R_Wrist',case_id=r['clip_id']+':'+str(offset)))
require(len(expected_cases)==72,'Manifest denominator')
proof=[]; metas=[]
for label in labels:
    p=out/'raw'/label; wanted=out/'requests'/(label+'.json'); req=read(wanted);meta=read(p/'metadata.json')
    metas.append(meta); lineage=read(p/'lineage.json');receipt=read(p/'receipt.json');metrics=read(p/'metrics.json')
    require(len(meta['cases'])==len(metrics)==receipt['cases']==72,'Actual raw denominator '+label)
    require(obj(meta['cases'])==obj(expected_cases),'Cases input order '+label)
    require(lineage['certification']=='bound_request_at_execution' and lineage['request']==req,'Execution binding '+label)
    for name,digest in lineage['outputs'].items():require(sha(p/name)==digest,'Output bytes '+label+'/'+name)
    require(receipt['status']=='complete' and receipt['trajectories_sha256']==sha(p/'trajectories.npz'),'Receipt '+label)
    require(req['versions']==vh and req['controller_state']=='t2m' and req['seed']==11 and req['noise_round']==0,'Request '+label)
    bind=read(p/'generation_input_binding.json');weights=read(p/'loaded_checkpoint_binding.json')
    require(bind['exact_loaded_dataset_bytes']==v['external_dataset_files'],'Actual loaded dataset '+label)
    require(bind['loaded_correction_object_sha256']==req.get('correction'),'Actual correction '+label)
    require(bind['request_sha256']==weights['request_sha256']==sha(wanted),'Child request '+label)
    require(bind['versions_sha256']==weights['versions_sha256']==vh and weights['controller_state']=='t2m','Child versions/controller '+label)
    require(sorted(weights['actual_loaded_weight_bytes'].values())==sorted([v['generator_weights_sha256'],v['controller_states']['t2m']['sha256']]),'Actual weights '+label)
    for name,key in [('initial_states.npz','initial_state_archive_sha256'),('observation_trace.npz','observation_trace_sha256')]:
        require(sha(p/name)==meta[key],'Export archive binding '+label)
    require(meta['initial_reset_gate_pass'] and not any(meta['initial_rigid_cache_consistency'].values()),'Reset gate '+label)
    with np.load(p/'trajectories.npz',allow_pickle=False) as a:
        require(a['actual'].shape[1]==72 and a['valid'].shape[1]==72 and a['terminated'].shape==(72,),'Raw all-case arrays '+label)
        require(np.array_equal(a['reference_frame'],np.arange(120)),'Reference clock '+label)
        shapes={k:list(a[k].shape) for k in a.files}
    log=(out/(label+'.log')).read_text()
    require('PHYSICS_GUIDED_COLLECTION_PASS 72' in log,'Batch terminal sentinel '+label)
    proof.append(dict(label=label,raw_cases=72,metadata_sha256=sha(p/'metadata.json'),lineage_sha256=sha(p/'lineage.json'),request_sha256=sha(wanted),dataset_file_count=len(bind['exact_loaded_dataset_bytes']),dataset_binding_sha256=sha(p/'generation_input_binding.json'),weights_binding_sha256=sha(p/'loaded_checkpoint_binding.json'),actual_loaded_weight_bytes=weights['actual_loaded_weight_bytes'],correction_object_sha256=bind['loaded_correction_object_sha256'],reference_clock=list(range(120)),trajectory_shapes=shapes,charged_seconds=next(r['seconds'] for r in ledger if r['label']==label),actual_exit_code=0,log_sha256=sha(out/(label+'.log'))))
same={}
for name in ['trajectories.npz','observation_trace.npz','initial_states.npz']:
    with np.load(out/'raw'/labels[0]/name,allow_pickle=False) as a,np.load(out/'raw'/labels[1]/name,allow_pickle=False) as b:
        require(set(a.files)==set(b.files),'Repeated field names '+name)
        for key in a.files:
            same[name+':'+key]=bool(np.array_equal(a[key],b[key]))
require(all(same.values()) and same==runtime['exact_same_input_arrays'],'Every same-input array')
require(metas[0]['generation_noise_sha256']==metas[1]['generation_noise_sha256'],'Actual paired noise')
corr=read(out/'feedback/feedback_witness_zero.json'); req=read(out/'requests/witness_zero_s11.json')
require(corr['no_feedback'] and obj(corr)==req['correction'],'Zero update object')
require(np.array_equal(np.asarray(corr['proposed_goal'],np.float32),np.asarray(metas[0]['original_goal'],np.float32)),'Zero update original goal')
require(np.array_equal(np.asarray(corr['proposed_heading'],np.float32),np.asarray(metas[0]['original_heading'],np.float32)),'Zero update original heading')
for name,digest in req['feedback_parent'].items():require(sha(pathlib.Path(corr['feedback_source'])/name)==digest,'Feedback exact bytes')
reference=out/'selected/witness_raw.npz'; replay_req=read(out/'requests/witness_raw_replay_s11.json')
require(motion_hash(reference)==replay_req['replay_motion'],'Fixed selected reference content binding')
with np.load(reference,allow_pickle=False) as selected,np.load(out/'raw'/labels[2]/'trajectories.npz',allow_pickle=False) as a:
    target=selected['reference'];canonical=(target-target[:,:,0:1,:])+target[:,:,0:1,:];valid=a['valid']
    err=float(np.max(np.abs(a['reference'][valid]-canonical[valid])))
    require(np.allclose(a['reference'][valid],canonical[valid],atol=1e-12,rtol=0),'Fixed replay same time/reference')
print(json.dumps(dict(status='PASS',versions_sha256=vh,runtime_witness_sha256=sha(out/'runtime_witness.json'),batches=proof,exact_same_input_arrays=same,same_input_array_count=len(same),selected_reference_file_sha256=sha(reference),selected_reference_motion_hash=motion_hash(reference),canonical_replay_max_error=err,replay_reference_clock_pass=True,zero_correction_bytes_sha256=sha(out/'feedback/feedback_witness_zero.json'),measurement_total_seconds=sum(r['seconds'] for r in ledger),simulation_calls_this_verifier=0,generation_calls_this_verifier=0),indent=2))
