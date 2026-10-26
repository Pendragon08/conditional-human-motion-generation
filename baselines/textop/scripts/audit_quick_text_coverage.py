"""Read-only lexical/temporal coverage audit; not a semantic quality scorer."""
import argparse,collections,hashlib,json,os,pathlib,re,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert re.fullmatch('[a-z0-9_]+',a.run)
    assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});os.environ['CUDA_VISIBLE_DEVICES']=''
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    spec=json.loads((ROOT/'configs/quick69-corpus-spec.json').read_text())
    src=ROOT/'quick_source_views/quick_sample64_nominal_v1/source_decisions.private.json'
    assert sha(src)==spec['private_source_view_sha256']
    raw=json.loads(src.read_text());rows=[x['reference'] for x in raw if x['include_in_candidate_view']]
    receipt=ROOT/'quick_generator_20261026/corpus_v1/corpus_receipt.json';cr=json.loads(receipt.read_text())
    assert len(rows)==cr['retained_sources']==42 and cr['quarantined_train_sources']==0
    result={};detail=[]
    for split in ['train','val']:
        rr=[x for x in rows if x['split']==split];total=collections.Counter();texts=set()
        categories={k:collections.Counter() for k in ['walk_stem','wave_stem','wave_stem_and_right']}
        for row in rr:
            anns=row['annotations'];frames=row['frames'];assert frames>=35
            total['sources']+=1;total['annotations']+=len(anns)
            texts.update(x['text'] for x in anns if x['text'])
            predicates={'walk_stem':lambda t:bool(re.search(r'\bwalk\w*\b',t)),
                'wave_stem':lambda t:bool(re.search(r'\bwav(?:e|es|ed|ing)\b',t)),
                'wave_stem_and_right':lambda t:bool(re.search(r'\bwav(?:e|es|ed|ing)\b',t)) and bool(re.search(r'\bright\b',t))}
            for key,fn in predicates.items():
                aa=[x for x in anns if fn(x['text'].lower())];categories[key]['sources']+=bool(aa);categories[key]['annotations']+=len(aa)
                categories[key]['annotated_seconds_sum_not_union']+=sum(x['end_s']-x['start_s'] for x in aa)
            counts=collections.Counter()
            # Exhaustive structural windows, NOT sampled training frequencies.
            # Official inclusive overlap and duplicate labels are deliberately preserved.
            for start in range(frames-35+1):
                for primitive in range(4):
                    first=start+8*primitive+2;last=start+8*primitive+10
                    matched=[x for x in anns if not(x['start_s']*50>last or first>x['end_s']*50)]
                    distinct={x['text'] for x in matched if x['text']}
                    counts['primitive_window_occurrences']+=1
                    counts['no_nonempty_label']+=not distinct
                    counts['multiple_distinct_labels']+=len(distinct)>1
                    counts['only_boundary_touch_nonempty']+=bool(distinct) and not any(x['text'] and min(x['end_s']*50,last)>max(x['start_s']*50,first) for x in matched)
            total.update(counts);detail.append({'source':row['canonical_source'],'split':split,'counts':dict(counts)})
        result[split]={'counts':dict(total),'unique_nonempty_texts':len(texts),'lexical_groups':{k:dict(v) for k,v in categories.items()}}
    out=ROOT/'semantic_diagnostics_private'/a.run;out.mkdir(mode=0o700,parents=True,exist_ok=False)
    private=out/'source_coverage.private.json';private.write_text(json.dumps(detail,indent=2))
    summary={'status':'QUICK_TEXT_COVERAGE_AUDITED_NOT_CAUSAL_PROOF','source_view_sha256':sha(src),'corpus_receipt_sha256':sha(receipt),
        'source_sha256':sha(pathlib.Path(__file__)),'split':result,'private_detail_sha256':sha(private),'sample_unix':time.time(),
        'limitations':['Lexical regex counts are not semantic labels or causal proof.','All legal window occurrences, not action-weighted observed minibatch frequency.',
         'Concurrent old labels can both be valid; ambiguity is not automatic bad annotation.','No corpus/text/cache/model modified. No root cause claimed.'],
        'sdk_runs':0,'model_loads':0,'gpu_calls':0,'optimizer_updates':0,'data_deleted':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
