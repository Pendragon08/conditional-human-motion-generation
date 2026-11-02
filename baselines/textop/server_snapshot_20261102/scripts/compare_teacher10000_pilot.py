"""Actual closed10k vs fixed5k recordings; private rows, public paired aggregates."""
import argparse,hashlib,json,os,pathlib
import numpy as np
from recorded_endpoint_metrics import endpoint_metrics
from strict_tracking_coverage import verify_trial
from quick_pilot_trace import FIELDS,WIDTH
from teacher10000_pilot_protocol import load_teacher,authorize_teacher,select_sources,settings_signature,STATUS,QUALITY
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()
def stats(values):
    a=np.asarray(values,dtype=np.float64)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(a.max())} if len(a) else {'n':0}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--verify-output',action='store_true');args=ap.parse_args()
    assert args.run.isidentifier() and args.run.isascii() and os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
    spec=ROOT/'configs/teacher10000-comparison-spec.json';s=json.loads(spec.read_text());assert s['schema']=='textop.teacher10000_paired_comparison.v1'
    assert sha(ROOT/'scripts/recorded_endpoint_metrics.py')==s['endpoint_kernel_sha256']
    pilot=ROOT/'configs/teacher10000-pilot-spec.json';p=json.loads(pilot.read_text());teacher=load_teacher(ROOT,p);cp=authorize_teacher(p,teacher)
    assert cp['sha256']==s['new_checkpoint_sha256'] and p['sample_launch_enabled']
    out=ROOT/'tracker_validation'/s['sample_run'];rp=out/'receipt.json';r=json.loads(rp.read_text());prefix='teacher10000_pilot_'+s['sample_run']
    ex_path=ROOT/'receipts'/(prefix+'_exit.json');ex=json.loads(ex_path.read_text());pr_path=ROOT/'receipts'/(prefix+'_process_exit.json');pr=json.loads(pr_path.read_text());resources=ROOT/'logs'/(prefix+'_resources.jsonl');outer=ROOT/'receipts'/(prefix+'_outer_exit.json')
    assert r['status']==STATUS and r['mode']=='teacher10000_pilot' and r['stage']=='sample' and r['quality_label']==QUALITY
    assert r['output_writers_closed'] and r['optimizer_updates']==0 and r['finished_trials']==r['replayed_original_sources']==64
    assert r['policy_checkpoint_completed_iterations']==10000 and r['policy_checkpoint_optimizer_updates']==200000 and r['policy_checkpoint_sha256']==s['new_checkpoint_sha256']
    assert r['source_sha256']==p['executor_sha256'] and r['evaluation_protocol_sha256']==p['protocol_sha256'] and r['trace_source_sha256']==p['trace_sha256']
    assert r['pilot_spec_sha256']==sha(pilot) and r['execution_settings_sha256']==settings_signature(p)
    assert ex['evaluation_receipt_sha256']==sha(rp) and ex['process_exit_sha256']==sha(pr_path) and ex['resources_sha256']==sha(resources)
    assert ex['numerical_status']==STATUS and ex['child_exit_code']==0 and ex['supervision_passed'] and not ex['timed_out']
    for k in ['child_pid','child_exit_code','child_reaped','timed_out','supervision_passed','supervisor_error','samples_written']:assert ex[k]==pr[k]
    assert pr['child_reaped'] and pr['supervisor_error'] is None and pr['sample_errors']==[] and not pr.get('cleanup_error')
    samples=[json.loads(x) for x in resources.read_text().splitlines()];assert len(samples)==pr['samples_written'] and len(samples)>=3
    assert samples[0]['phase']=='before_launch' and samples[-1]['phase']=='after_exit' and json.loads(outer.read_text())['actual_returncode']==0
    baseline=ROOT/'tracker_validation'/s['baseline_run'];brp=baseline/'receipt.json';assert sha(brp)==s['baseline_receipt_sha256']
    bex=ROOT/'receipts'/('quick_pilot_'+s['baseline_run']+'_exit.json');assert sha(bex)==s['baseline_exit_sha256'];br=json.loads(brp.read_text())
    assert br['policy_checkpoint_completed_iterations']==5000 and br['complete_trials']==42 and br['failed_trials']==22
    assert br['contact_profile_sha256']==r['contact_profile_sha256'] and br['reference_fingerprint']==r['reference_fingerprint']
    bp=ROOT/'endpoint_diagnostics_private/baseline5k_author_v1/endpoint_metrics.private.json';assert sha(bp)==s['baseline_endpoint_private_sha256'];old={x['motion_id']:x for x in json.loads(bp.read_text())};assert len(old)==64
    plan=json.loads((ROOT/'configs/filter-training-plan.json').read_text());rows=[];jn=None;bn=None
    for n in plan['input_reference_exports']:
        f=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/n;meta=json.loads((f/'receipt.json').read_text());mp=f/'manifest.private.jsonl';assert sha(mp)==meta['private_manifest_sha256']
        if jn is None:jn=meta['joint_names'];bn=meta['body_names']
        assert jn==meta['joint_names'] and bn==meta['body_names'];rows.extend(x for x in map(json.loads,mp.read_text().splitlines()) if not x['mirrored'])
    assert len(rows)==8278 and s['selection_sha256']==p['selection_sha256']
    selected={hashlib.sha256(rows[i]['canonical_source'].encode()).hexdigest():rows[i] for i in select_sources(rows,'sample',p)};assert set(selected)==set(old)
    log=out/'strict.private.json';journal=out/'strict_trials.private.jsonl';index=out/'executed_traces.private.json'
    assert sha(log)==r['private_strict_log_sha256'] and sha(journal)==r['private_trial_journal_sha256'] and sha(index)==r['private_executed_trace_index_sha256']
    trials={x['motion_id']:x['trials'] for x in json.loads(log.read_text())['motions']};assert set(trials)==set(old) and all(len(x)==1 for x in trials.values())
    canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
    assert sorted(canonical({'motion_id':k,'trial':v[0]}) for k,v in trials.items())==sorted(canonical(json.loads(x)) for x in journal.read_text().splitlines())
    records=json.loads(index.read_text());assert len(records)==64;seen=set();paired=[]
    for record in records:
        f=pathlib.Path(record['folder']);assert f.resolve().parent==out/'execution_private';mp=f/'trace.private.json';meta=json.loads(mp.read_text());key=meta['motion_id'];assert key in selected and key not in seen;seen.add(key);row=selected[key]
        assert meta['reference_frames']==row['frames'] and meta['trial']==trials[key][0];verify_trial(meta['trial'],row['frames'],0)
        assert meta['status']=='FINISHED_PHYSICAL_TRIAL' and meta['runtime_kind']=='isaaclab_physics' and meta['phase']=='INITIAL_THEN_PRE_AUTORESET_TERMINATION_COMPUTE'
        assert meta['fields']==[{'name':k,'shape':list(v)} for k,v in FIELDS] and meta['row_width']==WIDTH
        npz=f/'executed_poses.private.npz';binary=f/'poses.private.bin';assert sha(npz)==meta['npz_sha256']==record['npz_sha256'] and sha(binary)==meta['binary_sha256']
        assert meta['samples']==record['samples']==meta['trial']['control_steps']+1
        with np.load(npz,allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
        raw=np.fromfile(binary,dtype='<f4').reshape(meta['samples'],WIDTH);offset=0
        for key2,shape in FIELDS:
            n=int(np.prod(shape));assert np.array_equal(raw[:,offset:offset+n].reshape((meta['samples'],)+shape),trace[key2]);offset+=n
        refpath=pathlib.Path(row['file']);assert sha(refpath)==row['sha256']==old[key]['reference_sha256']
        with np.load(refpath,allow_pickle=False) as z:ref={k:z[k].copy() for k in ['joint_pos','body_pos_w','body_quat_w']}
        metrics=endpoint_metrics(trace,ref,meta['trial'],jn,bn)
        assert old[key]['source']==row['canonical_source'] and old[key]['split']==row['split']
        # Baseline was independently audited before this checkpoint; preserve it.
        paired.append({'motion_id':key,'source':row['canonical_source'],'split':row['split'],'frames':row['frames'],'old':old[key]['metrics'],'new':metrics,'new_trace_meta_sha256':sha(mp),'new_npz_sha256':sha(npz),'new_binary_sha256':sha(binary),'reference_sha256':row['sha256'],'current_target_metrics':meta['metrics']})
    assert seen==set(old)
    counts={k:0 for k in ['both_complete','new_only_complete','old_only_complete','both_failed']}
    for x in paired:
        a=x['old']['natural_duration_complete'];b=x['new']['natural_duration_complete'];counts['both_complete' if a and b else 'new_only_complete' if b else 'old_only_complete' if a else 'both_failed']+=1
    assert counts['both_complete']+counts['old_only_complete']==42 and counts['both_complete']+counts['new_only_complete']==r['complete_trials']
    groups={}
    for x in paired:
        dataset=pathlib.PurePosixPath(x['source'].replace('\\','/')).parts[0];duration='short_50_250' if x['frames']<=250 else 'medium_251_750' if x['frames']<=750 else 'long_751_2000'
        for group in ['split:'+x['split'],'dataset:'+dataset,'duration:'+duration]:
            z=groups.setdefault(group,{'count':0,'old_complete':0,'new_complete':0});z['count']+=1;z['old_complete']+=int(x['old']['natural_duration_complete']);z['new_complete']+=int(x['new']['natural_duration_complete'])
    common=[x for x in paired if x['old']['natural_duration_complete'] and x['new']['natural_duration_complete']]
    end={k:{'old_paired':stats([x['old'][k] for x in common]),'new_paired':stats([x['new'][k] for x in common]),'paired_change_new_minus_old':stats([x['new'][k]-x['old'][k] for x in common]),'new_all_completions':stats([x['new'][k] for x in paired if x['new']['natural_duration_complete']])} for k in ['terminal_world_root_position_error_m','terminal_root_orientation_geodesic_rad','terminal_joint_rmse_rad']}
    raw=json.dumps(paired,indent=2,sort_keys=True,allow_nan=False).encode();dest=ROOT/'teacher10000_comparison_private'/args.run
    summary={'status':'TEACHER10000_FIXED64_PAIRED_COMPARISON_RECORDED','sample_run':s['sample_run'],'baseline_run':s['baseline_run'],'spec_sha256':sha(spec),'source_sha256':sha(__file__),'new_receipt_sha256':sha(rp),'new_exit_sha256':sha(ex_path),'new_outer_exit_sha256':sha(outer),'new_checkpoint_sha256':cp['sha256'],'baseline_receipt_sha256':sha(brp),'baseline_endpoint_private_sha256':sha(bp),'source_count':64,'baseline_complete':42,'baseline_failed':22,'new_complete':r['complete_trials'],'new_failed':r['failed_trials'],'outcome_pairs':counts,'paired_terminal_metrics':end,'strata':groups,'private_paired_sha256':hashlib.sha256(raw).hexdigest(),'unattempted_original_sources':8214,'whole_library_accepted':False,'semantic_quality_accepted':False,'new_sdk_runs':0,'new_optimizer_updates':0,'original_runs_modified':False,'definition':s['definition'],'limitations':['Single fixed sample/checkpoint comparison, not full corpus or paper score.','Terminal comparison excludes earlier failures without extrapolation; paired denominator is explicit.','No foot contact force stream was recorded; no contact-conditioned slip acceptance claimed.','Weak-teacher failure does not prove motion infeasibility.']}
    if args.verify_output:
        assert (dest/'paired.private.json').read_bytes()==raw and json.loads((dest/'summary.json').read_text())==summary
        print(json.dumps({'status':'PAIRED64_INDEPENDENTLY_RECOMPUTED','matched_sources':64,'summary_sha256':sha(dest/'summary.json'),'new_sdk_runs':0}));return
    dest.mkdir(parents=True,exist_ok=False,mode=0o700);(dest/'paired.private.json').write_bytes(raw);(dest/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary))
if __name__=='__main__':main()
