"""Freeze existing lexical candidates for future 10k physics; no arrays or SDK."""
import collections,hashlib,json,os,pathlib
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
    poolroot=ROOT/'semantic_diagnostics_private/full_available_pool_v1'
    ps=json.loads((poolroot/'summary.json').read_text());pp=poolroot/'candidate_pool.private.json'
    assert sha(pp)==ps['private_pool_sha256'];pool=json.loads(pp.read_text())
    comp=ROOT/'teacher10000_comparison_private/paired64_v1';cj=json.loads((comp/'summary.json').read_text())
    assert cj['new_complete']==46 and cj['new_failed']==18 and not cj['semantic_quality_accepted']
    cp=comp/'paired.private.json';assert sha(cp)==cj['private_paired_sha256'];previous={x['source']:x for x in json.loads(cp.read_text())}
    config=json.loads((ROOT/'configs/full-library-capacity-spec.json').read_text());rows=[];hashes=[]
    for c in config['cohorts']:
        folder=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/c['run']
        r=json.loads((folder/'receipt.json').read_text());m=folder/'manifest.private.jsonl'
        assert sha(m)==r['private_manifest_sha256'];hashes.append(sha(m))
        rows.extend(x for x in map(json.loads,m.read_text().splitlines()) if not x['mirrored'])
    assert len(rows)==8278 and hashes==ps['reference_manifest_sha256']
    indexed={x['canonical_source']:(i,x) for i,x in enumerate(rows)};selected=[];wave=[]
    for p in pool:
        i,row=indexed[p['source']]
        assert p['reference_sha256']==row['sha256'] and p['frames']==row['frames'] and p['split']==row['split']
        if 'wave_stem' in p['lexical_groups']:wave.append(p)
        if 'wave_stem_and_right' not in p['lexical_groups']:continue
        old=previous.get(p['source'])
        if old:assert old['reference_sha256']==row['sha256'] and old['split']==row['split']
        selected.append({'source':p['source'],'original_index':i,'split':row['split'],'frames':row['frames'],'reference_sha256':row['sha256'],
          'existing_closed_10k_trial':bool(old),'existing_trial_natural_completion':old['new']['natural_duration_complete'] if old else None})
    selected.sort(key=lambda x:hashlib.sha256(x['source'].encode()).hexdigest())
    assert len(selected)==44 and dict(collections.Counter(x['split'] for x in selected))=={'train':27,'val':17}
    assert len(wave)==208 and len({x['source'] for x in selected})==44
    for x in selected:assert x['frames']>=2
    dest=ROOT/'semantic_diagnostics_private/semantic_physics_plan_v1';dest.mkdir(mode=0o700,exist_ok=False)
    private=dest/'selection.private.json';private.write_text(json.dumps(selected,indent=2));os.chmod(private,0o600)
    pending=[x for x in selected if not x['existing_closed_10k_trial']]
    summary={'status':'SEMANTIC_LEXICAL_PHYSICS_SCOPE_FROZEN_EXECUTOR_PENDING','selection_count':44,'split':dict(collections.Counter(x['split'] for x in selected)),
      'broader_wave_candidate_count':208,'existing_10k_trials_reused':44-len(pending),'new_physics_trials_needed':len(pending),
      'pending_split':dict(collections.Counter(x['split'] for x in pending)),
      'pending_frames':{'min':min(x['frames'] for x in pending),'max':max(x['frames'] for x in pending),'sum':sum(x['frames'] for x in pending)},
      'selection_sha256':sha(private),'pool_sha256':sha(pp),'paired_summary_sha256':sha(comp/'summary.json'),'paired_private_sha256':sha(cp),
      'checkpoint_sha256':cj['new_checkpoint_sha256'],'source_sha256':sha(__file__),'reference_fingerprint':ps['reference_fingerprint'],
      'launch_enabled':False,'new_sdk_runs':0,'new_optimizer_updates':0,'semantics_accepted':False,'corpus_accepted':False,
      'limits':['Lexical wave+right hints only, not verified action semantics.','Use whole references, preserve original train/val; no length truncation or thresholds relaxed.','Reuse exact closed 10k outcomes if present, no best-of retries.','No corpus expansion, reweighting or generator retraining has occurred.','Need a separate bound executor/spec and real document validation before new SDK replay.']}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
