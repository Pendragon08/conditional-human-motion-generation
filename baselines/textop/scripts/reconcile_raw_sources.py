"""Resolve AMASS export filename changes without fuzzy or semantic matching.

Private indices retain labels and exact source paths. Public output is aggregate.
Only documented group aliases, a repeated group directory, and the SMPL-X
export's literal space-to-underscore renaming are normalized. Ambiguous matches
fail closed. Existing 500 preprocessing outputs are neither modified nor counted
as newly processed.
"""
import argparse, collections, json, pathlib

ALIASES={'BioMotionLab_NTroje':'BMLrub','MPI_HDM05':'HDM05','MPI_mosh':'MoSh','MPI_Limits':'PosePrior','SSM_synced':'SSM','DFaust_67':'DFaust','Eyes_Japan_Dataset':'EyesJapanDataset','Transitions_mocap':'Transitions','MPI_EGOMOTION':'EGOMOTION'}

def source_key(value):
    parts=list(pathlib.PurePosixPath(str(value).replace('\\','/')).parts)
    parts[0]=ALIASES.get(parts[0],parts[0])
    if len(parts)>2 and ALIASES.get(parts[1],parts[1])==parts[0]:parts.pop(1)
    parts[-1]=pathlib.PurePosixPath(parts[-1]).stem
    for suffix in ['_poses','_stageii','_stagei']:
        if parts[-1].endswith(suffix):parts[-1]=parts[-1][:-len(suffix)]
    return '/'.join(p.replace(' ','_') for p in parts)

def main(args):
    import joblib
    args.output.mkdir(parents=True,exist_ok=True)
    raw=collections.defaultdict(list)
    for path in args.raw_amass.rglob('*_stageii.npz'):
        parts=path.relative_to(args.raw_amass).parts
        if 'extracted' not in parts:continue
        key=source_key('/'.join(parts[parts.index('extracted')+1:]));raw[key].append(str(path))
    # Multiple physical copies are acceptable only when their bytes agree.
    import hashlib
    def sha(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        return h.hexdigest()
    ambiguous={key:paths for key,paths in raw.items() if len(paths)>1 and len({sha(p) for p in paths})>1}
    old=json.loads(args.previous_pairs.read_text());old_ids={(x['split'],str(x['babel_sid'])) for x in old}
    all_rows=[];new=[];groups=collections.Counter();unmatched=collections.Counter()
    for split in ['train','val']:
        data=joblib.load(args.dataset/(split+'.pkl'))
        for x in data:
            key=source_key(x['feat_p']);paths=[] if key in ambiguous else raw.get(key,[])
            row={'split':split,'babel_sid':x['babel_sid'],'source':x['feat_p'],'canonical_source':key,
                 'frames':len(x['motion']['dof']),'fps':float(x['motion']['fps']),
                 'frame_ann':x['frame_ann'],'raw_stageii_candidates':paths,
                 'matching_rule':'explicit aliases, repeated group removal, literal space to underscore; no fuzzy matching'}
            all_rows.append(row)
            if paths:
                groups[key.split('/')[0]]+=1
                if (split,str(x['babel_sid'])) not in old_ids:new.append(row)
            else:unmatched[key.split('/')[0]]+=1
        del data
    (args.output/'release_input_index.private.json').write_text(json.dumps(all_rows))
    (args.output/'additional_existing_raw_pairs.private.json').write_text(json.dumps(new))
    summary={'release_sequences':len(all_rows),'previous_exact_raw_matches':len(old_ids),
             'additional_existing_raw_matches':len(new),'matched_by_group':dict(groups),
             'still_unmatched_by_group':dict(unmatched),'ambiguous_normalized_keys':len(ambiguous),
             'newly_retargeted_sequences':0,'physical_filter_executed':False,
             'note':'Additional input availability is not completed retargeting or dynamic filtering'}
    (args.output/'raw_reconciliation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=pathlib.Path,required=True);p.add_argument('--raw-amass',type=pathlib.Path,required=True);p.add_argument('--previous-pairs',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);main(p.parse_args())
