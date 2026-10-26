"""Metadata-only candidate pool for later real filtering; never accepts a corpus."""
import collections,hashlib,json,os,pathlib,re
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
def main():
    assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});os.environ['CUDA_VISIBLE_DEVICES']=''
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cap=json.loads((ROOT/'configs/full-library-capacity-spec.json').read_text());rows=[];digests=[]
    for cohort in cap['cohorts']:
        folder=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/cohort['run']
        meta=json.loads((folder/'receipt.json').read_text());manifest=folder/'manifest.private.jsonl'
        assert meta['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED' and meta['output_writers_closed']
        digest=sha(manifest);assert digest==meta['private_manifest_sha256'];digests.append(digest)
        original=[x for x in map(json.loads,manifest.read_text().splitlines()) if not x['mirrored']];assert len(original)==cohort['sources'];rows+=original
    assert len(rows)==len({x['canonical_source'] for x in rows})==8278
    fp=hashlib.sha256(json.dumps([{'sha256':x['sha256'],'frames':x['frames']} for x in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert fp=='0b786ba08bf8f33aa92cb4e91ed72cd5b56159c4e251cedf88f429269141c1f7'
    out=ROOT/'semantic_diagnostics_private/full_available_pool_v1';out.mkdir(parents=True,mode=0o700,exist_ok=False)
    summary={};private=[]
    for split in ['train','val']:
        counters={k:collections.Counter() for k in ['walk_stem','wave_stem','wave_stem_and_right']}
        for row in rows:
            if row['split']!=split:continue
            predicates={'walk_stem':lambda t:bool(re.search(r'\bwalk\w*\b',t)),
              'wave_stem':lambda t:bool(re.search(r'\bwav(?:e|es|ed|ing)\b',t)),
              'wave_stem_and_right':lambda t:bool(re.search(r'\bwav(?:e|es|ed|ing)\b',t)) and bool(re.search(r'\bright\b',t))}
            found=[]
            for key,fn in predicates.items():
                aa=[x for x in row['annotations'] if fn(x['text'].lower())]
                counters[key]['sources']+=bool(aa);counters[key]['annotations']+=len(aa)
                counters[key]['annotated_seconds_sum_not_union']+=sum(x['end_s']-x['start_s'] for x in aa)
                if aa:found.append(key)
            if found:private.append(dict(source=row['canonical_source'],split=split,lexical_groups=found,frames=row['frames'],reference_sha256=row['sha256'],physical_acceptance=False))
        summary[split]={k:dict(v) for k,v in counters.items()}
    pp=out/'candidate_pool.private.json';pp.write_text(json.dumps(private,indent=2))
    result={'status':'LEXICAL_CANDIDATE_POOL_LOCATED_PHYSICS_AND_SEMANTICS_UNACCEPTED','source_count':8278,'reference_fingerprint':fp,
       'reference_manifest_sha256':digests,'split':summary,'private_pool_sha256':sha(pp),'source_sha256':sha(pathlib.Path(__file__)),
       'motion_arrays_reloaded':False,'dataset_accepted':False,'gpu_calls':0,'optimizer_updates':0,
       'limitations':'Regex lexical hints only; synonyms and valid concurrent labels may be missed. No actual robot or semantics acceptance; use original split and real physics before corpus inclusion.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
