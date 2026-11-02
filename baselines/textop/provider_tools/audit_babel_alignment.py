"""Compare original licensed BABEL labels with all released TextOp sequence IDs.

Exact strings/categories and explicit source aliases only. Time differences are
reported with a1e-6-second numerical tolerance; this does not prove physical
tracking, SMPL-X motion-phase alignment, or final training-set acceptance.
"""
import collections,datetime,hashlib,html,json,pathlib,sys
from session import REPO
sys.path.insert(0,str(REPO/'baselines/textop/scripts'))
from reconcile_raw_sources import source_key
from restore_indexed_cache import sha
BASE=REPO/'baselines/textop';DEST=BASE/'results/data_preparation_20261025'
TASK=REPO.parent.parent/'textop_reproduction_20261024'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024');BABEL=LOCAL/'datasets/BABEL'
sys.stdout.reconfigure(encoding='utf-8')

def annotation(v):
    block=v.get('frame_ann')
    if isinstance(block,dict):
        return [[float(x['start_t']),float(x['end_t']),x['proc_label'],x['act_cat']] for x in block['labels']]
    return [[0.,float(v['dur']),x['proc_label'],x['act_cat']] for x in v['seq_ann']['labels']]

def main():
    path=TASK/'private_assets/release_input_index.private.json';rows=json.loads(path.read_text(encoding='utf-8'))
    assert len(rows)==8291 and len({(x['split'],str(x['babel_sid'])) for x in rows})==8291
    original={split:json.loads((BABEL/(split+'.json')).read_text(encoding='utf-8')) for split in ['train','val']}
    original_hashes={split:sha(BABEL/(split+'.json')) for split in original}
    totals=collections.Counter();groups=collections.defaultdict(collections.Counter);issues=[];paired=[];max_error=0.;wrapper_count=0
    for row in rows:
        split=row['split'];sid=str(row['babel_sid']);totals['release_sequences']+=1;groups[split]['sequences']+=1
        problems=[];v=original[split].get(sid)
        if v is None:problems.append('id_not_in_original_split')
        else:
            if int(v['babel_sid'])!=int(sid):problems.append('original_id_field_differs')
            # Reproduce official pack_dataset.process_babel_json literally:
            # BABEL's outer dataset wrapper is removed before source lookup.
            original_parts=pathlib.PurePosixPath(v['feat_p']).parts
            assert len(original_parts)>2
            packed_source='/'.join(original_parts[1:])
            wrapper_count+=source_key(v['feat_p'])!=source_key(row['source'])
            if source_key(packed_source)!=source_key(row['source']):problems.append('source_path_differs_after_official_packing')
            expected=annotation(v);actual=row['frame_ann'];totals['original_segment_labels_for_release_subset']+=len(expected)
            groups[split]['original_segment_labels']+=len(expected)
            if len(expected)!=len(actual):problems.append('annotation_count_differs')
            else:
                for a,b in zip(expected,actual):
                    delta=max(abs(a[0]-float(b[0])),abs(a[1]-float(b[1])));max_error=max(max_error,delta)
                    if delta>1e-6:problems.append('annotation_times_differ')
                    if a[2]!=b[2]:problems.append('processed_text_differs')
                    if a[3]!=b[3]:problems.append('action_categories_differ')
            if not problems:
                groups[split]['validated_sequences']+=1;totals['validated_sequences']+=1
                paired.append({**row,'frame_ann':expected,'original_babel_sha256':original_hashes[split],
                               'original_BABEL_labels_verified':True,'physical_filter_passed':False})
        if problems:
            totals.update(set(problems));issues.append({'split':split,'babel_sid':sid,'reasons':sorted(set(problems))})
    ids_train=set(original['train']);ids_val=set(original['val']);assert not ids_train&ids_val
    # Private derivative indexes stay beside their licensed original labels.
    (BABEL/'textop_alignment_issues.private.json').write_text(json.dumps(issues),encoding='utf-8')
    (BABEL/'textop_original_verified_pairs.private.json').write_text(json.dumps(paired),encoding='utf-8')
    result={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'status':'ORIGINAL_BABEL_LABELS_VALIDATED_FOR_ALL_RELEASE_PAIRS' if not issues else 'ORIGINAL_BABEL_DIFFERENCES_REQUIRE_REVIEW',
            'counts':dict(totals),'split_counts':{k:dict(v) for k,v in groups.items()},'mismatch_sequences':len(issues),
            'time_tolerance_seconds':1e-6,'max_observed_time_error_seconds':max_error,
            'id_train_val_overlap':len(ids_train&ids_val),'release_index_sha256':sha(path),
            'original_file_sha256':original_hashes,
            'pinned_official_commit':'ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22',
            'official_pack_dataset_sha256':sha(BASE/'upstream/dataset/pack_dataset.py'),
            'verified_pair_index_sha256':sha(BABEL/'textop_original_verified_pairs.private.json'),
            'source_matching':'official pack_dataset.process_babel_json drops first wrapper directory; then explicit aliases and literal space/underscore normalization; no fuzzy matching',
            'initial_path_wrapper_mismatches_resolved':wrapper_count,
            'raw_annotation_files_published':False,'physical_filter_executed':False,
            'smplx_motion_phase_alignment_independently_verified':False,'generator_dataset_accepted':False}
    (DEST/'babel_alignment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (LOCAL/'BABEL与TextOp配对核查_20261025.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
