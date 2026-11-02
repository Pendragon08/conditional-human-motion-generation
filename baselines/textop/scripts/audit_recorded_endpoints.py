"""Verify existing physical recordings then add separate private terminal metrics."""
import argparse,hashlib,json,os,pathlib,time
import numpy as np
from recorded_endpoint_metrics import endpoint_metrics
from strict_tracking_coverage import verify_trial
from quick_pilot_trace import FIELDS,WIDTH
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--run',required=True);p.add_argument('--verify-output',action='store_true');a=p.parse_args()
    assert a.run.isidentifier() and a.run.isascii() and os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});os.environ['CUDA_VISIBLE_DEVICES']=''
    sp=pathlib.Path(a.spec);s=json.loads(sp.read_text());assert s['schema']=='textop.recorded_endpoint_audit.v1'
    for n,h in s['bound_sources'].items():assert sha(ROOT/n)==h,n
    original=ROOT/'tracker_validation'/s['baseline_run'];rp=original/'receipt.json';receipt=json.loads(rp.read_text())
    assert sha(rp)==s['baseline_receipt_sha256'] and receipt['status']=='TEXTOP_QUICK_PILOT_EXECUTION_COMPLETED' and receipt['output_writers_closed']
    assert receipt['policy_checkpoint_completed_iterations']==5000 and receipt['policy_checkpoint_optimizer_updates']==100000
    ex=ROOT/'receipts'/('quick_pilot_'+s['baseline_run']+'_exit.json');assert sha(ex)==s['baseline_exit_sha256']
    exit=json.loads(ex.read_text());assert exit['child_exit_code']==0 and exit['supervision_passed'] and not exit['timed_out'] and exit['evaluation_receipt_sha256']==sha(rp)
    process=ROOT/'receipts'/('quick_pilot_'+s['baseline_run']+'_process_exit.json');resources=ROOT/'logs'/('quick_pilot_'+s['baseline_run']+'_resources.jsonl')
    assert sha(process)==exit['process_exit_sha256'] and sha(resources)==exit['resources_sha256']
    selection=ROOT/'tracker_validation/quick_pilot_selection_metadata_v1/selection.private.json';assert sha(selection)==s['private_selection_sha256']
    chosen=json.loads(selection.read_text())['stages']['sample'];assert len(chosen)==64
    plan=json.loads((ROOT/'configs/filter-training-plan.json').read_text());rows=[];joint_names=None;body_names=None;manifest_hashes=[]
    for name in plan['input_reference_exports']:
        folder=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/name;meta=json.loads((folder/'receipt.json').read_text());manifest=folder/'manifest.private.jsonl'
        assert meta['output_writers_closed'] and sha(manifest)==meta['private_manifest_sha256'];manifest_hashes.append(sha(manifest))
        if joint_names is None:joint_names=meta['joint_names'];body_names=meta['body_names']
        assert joint_names==meta['joint_names'] and body_names==meta['body_names']
        rows.extend(x for x in map(json.loads,manifest.read_text().splitlines()) if not x['mirrored'])
    fp=hashlib.sha256(json.dumps([dict(sha256=x['sha256'],frames=x['frames']) for x in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert len(rows)==8278 and fp==s['reference_fingerprint']==receipt['reference_fingerprint']
    selected={}
    for x in chosen:
        row=rows[x['global_index']]
        assert row['canonical_source']==x['canonical_source'] and row['frames']==x['frames'] and row['split']==x['split'] and row['sha256']==x['reference_sha256']
        selected[hashlib.sha256(row['canonical_source'].encode()).hexdigest()]=row
    index=original/'executed_traces.private.json';assert sha(index)==receipt['private_executed_trace_index_sha256']
    strict=original/'strict.private.json';journal=original/'strict_trials.private.jsonl'
    assert sha(strict)==receipt['private_strict_log_sha256'] and sha(journal)==receipt['private_trial_journal_sha256']
    trials={x['motion_id']:x['trials'] for x in json.loads(strict.read_text())['motions']}
    assert set(trials)==set(selected) and all(len(x)==1 for x in trials.values())
    canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
    assert sorted(canonical({'motion_id':k,'trial':v[0]}) for k,v in trials.items())==sorted(canonical(json.loads(line)) for line in journal.read_text().splitlines())
    records=json.loads(index.read_text());assert len(records)==64;results=[];seen=set();before={rp:sha(rp),ex:sha(ex),index:sha(index)}
    for record in records:
        folder=pathlib.Path(record['folder']);assert folder.resolve().parent==original/'execution_private'
        mp=folder/'trace.private.json';meta=json.loads(mp.read_text());motion=meta['motion_id'];assert motion in selected and motion not in seen;seen.add(motion);row=selected[motion]
        assert meta['reference_frames']==row['frames'] and meta['status']=='FINISHED_PHYSICAL_TRIAL' and meta['runtime_kind']=='isaaclab_physics'
        assert meta['phase']=='INITIAL_THEN_PRE_AUTORESET_TERMINATION_COMPUTE'
        assert meta['trial']==trials[motion][0];verify_trial(meta['trial'],row['frames'],0)
        assert meta['row_width']==WIDTH and meta['fields']==[{'name':k,'shape':list(v)} for k,v in FIELDS]
        npz=folder/'executed_poses.private.npz';binary=folder/'poses.private.bin';assert sha(npz)==meta['npz_sha256']==record['npz_sha256'] and sha(binary)==meta['binary_sha256']
        reference=pathlib.Path(row['file']);assert sha(reference)==row['sha256']
        with np.load(npz,allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
        with np.load(reference,allow_pickle=False) as z:ref={k:z[k].copy() for k in ['joint_pos','body_pos_w','body_quat_w']}
        raw=np.fromfile(binary,dtype='<f4').reshape(meta['samples'],meta['row_width']);offset=0
        for field in meta['fields']:
            count=int(np.prod(field['shape']));assert np.array_equal(raw[:,offset:offset+count].reshape((meta['samples'],)+tuple(field['shape'])),trace[field['name']]);offset+=count
        assert offset==meta['row_width'] and meta['samples']==record['samples']==meta['trial']['control_steps']+1
        metrics=endpoint_metrics(trace,ref,meta['trial'],joint_names,body_names)
        results.append({'motion_id':motion,'source':row['canonical_source'],'split':row['split'],'metrics':metrics,
          'trace_meta_sha256':sha(mp),'npz_sha256':sha(npz),'binary_sha256':sha(binary),'reference_sha256':row['sha256']})
    assert seen==set(selected) and sum(x['metrics']['reference_endpoint_independently_compared'] for x in results)==42
    assert sum(x['metrics']['physical_failed'] for x in results)==22 and all(sha(p)==h for p,h in before.items())
    valid=[x['metrics'] for x in results if x['metrics']['reference_endpoint_independently_compared']]
    names=['terminal_world_root_position_error_m','terminal_world_root_horizontal_error_m','terminal_world_root_height_error_abs_m','terminal_root_orientation_geodesic_rad','terminal_joint_rmse_rad','terminal_joint_abs_max_rad']
    aggregate={}
    for k in names:
        v=np.asarray([x[k] for x in valid]);aggregate[k]={'mean':float(v.mean()),'median':float(np.median(v)),'p95':float(np.percentile(v,95)),'max':float(v.max())}
    out=ROOT/'endpoint_diagnostics_private'/a.run
    private_bytes=json.dumps(results,indent=2,sort_keys=True,allow_nan=False).encode()
    receipt={'status':'RECORDED_5K_TERMINAL_REFERENCE_METRICS_VERIFIED','spec_sha256':sha(sp),'source_sha256':sha(__file__),
      'baseline_run':s['baseline_run'],'baseline_receipt_sha256':sha(rp),'baseline_exit_sha256':sha(ex),'baseline_trace_index_sha256':sha(index),
      'reference_fingerprint':fp,'reference_manifest_sha256':manifest_hashes,'closed_recordings':64,'terminal_compared_natural_completions':42,'early_failures_endpoint_not_extrapolated':22,
      'terminal_metrics':aggregate,'initial_world_root_offset_max_m':max(x['metrics']['initial_world_root_offset_m'] for x in results),
      'private_metrics_sha256':hashlib.sha256(private_bytes).hexdigest(),'original_artifacts_modified':False,'sdk_runs':0,'optimizer_updates':0,'model_loads':0,
      'new_physical_trials':0,'physical_acceptance_changed':False,'semantic_quality_accepted':False,
      'definition':'Terminal physical sample at elapsed(T-1)/50 compared to independent referenceT-1 in absolute source world coordinates, no yaw/root alignment; controller consumedT-2. Not a new physical trial or paper benchmark score.'}
    if a.verify_output:
        assert (out/'endpoint_metrics.private.json').read_bytes()==private_bytes
        assert json.loads((out/'summary.json').read_text())==receipt
        print(json.dumps({'status':'EXISTING_ENDPOINT_AUDIT_INDEPENDENTLY_RECOMPUTED','matched_trials':64,'new_sdk_runs':0,'original_artifacts_modified':False,'summary_sha256':sha(out/'summary.json')}));return
    out.mkdir(parents=True,exist_ok=False,mode=0o700)
    (out/'endpoint_metrics.private.json').write_bytes(private_bytes);(out/'summary.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    print(json.dumps(receipt))
if __name__=='__main__':main()
