"""Input/output binding and explicit retrospective certification for amended runs."""
import hashlib
import json
from pathlib import Path
import numpy as np


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def motion_hash(path):
    h=hashlib.sha256()
    with np.load(path,allow_pickle=False) as arrays:
        for key in sorted(arrays.files):
            x=np.ascontiguousarray(arrays[key]); h.update(key.encode())
            h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def request(root,out,split,mode,seed,round_id,kwargs):
    result=dict(split=split,mode=mode,seed=seed,round=round_id,
                noise_round=kwargs.get('noise_round',round_id),
                config=file_hash(root/'configs/physics_guided_20261011.json'),
                manifest=file_hash(out/'source_manifest.json'),
                thresholds=file_hash(out/'frozen_thresholds.json'),
                versions=file_hash(out/'versions.json'))
    if kwargs.get('correction_file'):
        path=Path(kwargs['correction_file']); data=json.loads(path.read_text())
        result['correction']=object_hash(data)
        previous=Path(data['feedback_source'])
        result['feedback_parent']={key:file_hash(previous/key) for key in ('metadata.json','metrics.json','trajectories.npz')}
    if kwargs.get('replay_file'): result['replay_motion']=motion_hash(kwargs['replay_file'])
    return result


def certify_cached(root,out,path,expected,kwargs,config):
    """Does not assert that a missing historical binding existed at execution."""
    from physics_guided_core import metrics
    receipt=json.loads((path/'receipt.json').read_text())
    meta=json.loads((path/'metadata.json').read_text())
    saved=json.loads((path/'metrics.json').read_text())
    assert receipt['status']=='complete' and receipt['cases']==len(meta['cases'])==len(saved)
    assert (meta['mode'],meta['seed'],meta['round'])==(expected['mode'],expected['seed'],expected['round'])
    assert receipt['trajectories_sha256']==file_hash(path/'trajectories.npz')
    manifest=json.loads((out/'source_manifest.json').read_text())
    source_rows=[r for r in manifest['rows'] if r['split']==expected['split']]
    wanted=[]
    for r in source_rows:
        for offset in config['offsets_m']:
            c=dict(r,offset_m=offset,goal_joint_body='Pelvis' if r['goal_joint']=='pelvis' else 'R_Wrist',
                   case_id=r['clip_id']+':'+str(offset))
            wanted.append(c)
    assert object_hash(meta['cases'])==object_hash(wanted), 'Cached complete case/source/text/goal order differs'
    if kwargs.get('correction_file'):
        feedback=json.loads(Path(kwargs['correction_file']).read_text())
        assert feedback['case_ids']==[r['case_id'] for r in meta['cases']]
        assert feedback['original_goal_sha256']==meta['original_goal_sha256']
        assert np.array_equal(np.asarray(feedback['proposed_goal'],np.float32),np.asarray(meta['proposed_goal'],np.float32))
        # Collector stores float32 model inputs; compare the same serialized
        # precision rather than an unrelated absolute tolerance near pi.
        assert np.array_equal(np.asarray(feedback['proposed_heading'],np.float32),np.asarray(meta['proposed_heading'],np.float32))
    else:
        assert np.array_equal(np.asarray(meta['proposed_goal']),np.asarray(meta['original_goal']))
        assert np.array_equal(np.asarray(meta['proposed_heading']),np.asarray(meta['original_heading']))
    if expected['mode']!='replay':
        # Every round uses the same sized explicit torch noise. The base round
        # is the independently retained noise witness for an equal-noise arm.
        label=expected['split']+'_'+('native' if expected['mode']=='native' else 'planned')+'_s%d_r%d'%(expected['seed'],expected['noise_round'])
        base=out/'raw'/label/'metadata.json'
        if base.exists(): assert meta['generation_noise_sha256']==json.loads(base.read_text())['generation_noise_sha256']
    thresholds=json.loads((out/'frozen_thresholds.json').read_text())['thresholds']
    replay_max_error=None
    with np.load(path/'trajectories.npz',allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
        assert np.array_equal(arrays['reference_frame'],np.arange(config['episode_steps_30hz']))
        if kwargs.get('replay_file'):
            with np.load(kwargs['replay_file'],allow_pickle=False) as replay:
                actual_ref=arrays['reference'][arrays['valid']]; wanted_ref=replay['reference'][arrays['valid']]
                replay_max_error=float(np.max(np.abs(actual_ref-wanted_ref)))
                assert np.allclose(actual_ref,wanted_ref,atol=1e-7,rtol=0), 'Stale replay input beyond coordinate roundtrip tolerance'
        recomputed=[metrics(arrays,i,meta,thresholds) for i in range(len(saved))]
        assert object_hash(recomputed)==object_hash(saved), 'Cached metrics differ from current measurement implementation'
    binding=path/'lineage.json'
    if binding.exists():
        old=json.loads(binding.read_text())
        if old['request']!=expected:
            assert {k:v for k,v in old['request'].items() if k!='versions'}=={k:v for k,v in expected.items() if k!='versions'}, 'Bound input changed'
            if old['certification']=='bound_request_at_execution':
                # Keep the original execution request/binding untouched. The
                # narrowly scoped amendment changes only evaluator resolution,
                # cache verification and final gate analysis, not construction.
                amendment=json.loads((out/'nonexecution_version_amendment.json').read_text())
                snapshots=amendment['previous_versions']
                assert old['request']['versions'] in snapshots
                previous=out/snapshots[old['request']['versions']]
                assert file_hash(previous)==old['request']['versions']
                assert expected['versions']==amendment['to_versions_sha256']
                before=json.loads(previous.read_text()); after=json.loads((out/'versions.json').read_text())
                assert {k:v for k,v in before.items() if k!='files'}=={k:v for k,v in after.items() if k!='files'}
                changed={k for k in set(before['files'])|set(after['files']) if before['files'].get(k)!=after['files'].get(k)}
                allowed={'scripts/physics_guided_semantics.py','scripts/physics_guided_lineage.py','scripts/physics_guided_analyze.py'}
                assert changed<=set(amendment['changed_files']) and changed<=allowed
                assert json.loads((out/'requests'/(path.name+'.json')).read_text())==old['request']
                compatibility=dict(original_lineage_sha256=file_hash(binding),original_execution_request=old['request'],
                    current_compatibility_request=expected,certification='exact_array_and_metric_compatibility_after_nonexecution_amendment',
                    amendment_sha256=file_hash(out/'nonexecution_version_amendment.json'),
                    outputs=old['outputs'],replay_reference_max_error_m=replay_max_error,replay_reference_tolerance_m=1e-7,
                    confirmatory=False)
                (path/'compatibility_after_amendment.json').write_text(json.dumps(compatibility,indent=2))
                assert old['outputs']=={name:file_hash(path/name) for name in ('metadata.json','metrics.json','trajectories.npz')}
                return meta
            # Only explicit retrospective compatibility can advance to a new
            # code snapshot. Preserve every earlier binding; never claim it
            # was the original execution request.
            assert old['certification'].startswith('retrospective')
            history=path/'lineage_history.json'
            records=json.loads(history.read_text()) if history.exists() else []
            records.append(old);history.write_text(json.dumps(records,indent=2))
            old['request']=expected;old['request_role']='retrospective_compatibility_target_not_execution_record'
            old['replay_reference_max_error_m']=replay_max_error
            binding.write_text(json.dumps(old,indent=2))
        assert old['outputs']=={name:file_hash(path/name) for name in ('metadata.json','metrics.json','trajectories.npz')}
    else:
        record=dict(request=expected,outputs={name:file_hash(path/name) for name in ('metadata.json','metrics.json','trajectories.npz')},
                    certification='retrospective_exact_array_and_metric_verification',
                    original_request_binding='not_recorded_at_execution',confirmatory=False,
                    request_role='retrospective_compatibility_target_not_execution_record',
                    replay_reference_max_error_m=replay_max_error,replay_reference_tolerance_m=1e-7)
        binding.write_text(json.dumps(record,indent=2))
    return meta
