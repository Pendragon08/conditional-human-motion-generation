"""Audit released RobotMDAR pairs without modifying motions or annotations.

Full manifests contain licensed labels/source identifiers and stay private.
Only aggregate diagnostics and checksums are suitable for public reporting.
"""
import argparse,collections,hashlib,json,pathlib,time
import joblib,numpy as np,torch

ALIASES={'BioMotionLab_NTroje':'BMLrub','MPI_HDM05':'HDM05','MPI_mosh':'MoSh','MPI_Limits':'PosePrior','SSM_synced':'SSM','DFaust_67':'DFaust','Eyes_Japan_Dataset':'EyesJapanDataset','Transitions_mocap':'Transitions','MPI_EGOMOTION':'EGOMOTION'}
def canonical(p):
    parts=pathlib.PurePosixPath(str(p).replace('\\','/')).parts
    group=ALIASES.get(parts[0],parts[0]);stem=pathlib.Path(parts[-1]).stem
    for suffix in ['_poses','_stageii','_stagei']:
        if stem.endswith(suffix):stem=stem[:-len(suffix)]
    return '/'.join([group,*parts[1:-1],stem])
def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    raw={};rawbase=args.raw_amass
    for p in rawbase.rglob('*_stageii.npz'):
        parts=p.relative_to(rawbase).parts
        if 'extracted' not in parts:continue
        start=parts.index('extracted')+1
        key=canonical('/'.join(parts[start:]))
        raw.setdefault(key,[]).append(str(p))
    summary={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'released 23DoF / 57D dataset audit only; not physical tracking filtering','raw_stageii_unique':len(raw),'splits':{},'unchanged_source_files':{},'physical_filter_reexecuted':False,'private_dataset_available':False,'paper_target':{'dof':29,'feature_dim':69,'contact_height_m':0.2,'contact_squared_displacement':0.002,'filter_training_iterations':40000,'relaxed_failure_threshold':0.05,'strict_filter':'remove any failing motion'}}
    names={};ids={};fingerprints={};allmatches=[];allissues=collections.Counter();groups=collections.Counter()
    for split in ['train','val']:
        source=args.dataset/(split+'.pkl');data=joblib.load(source)
        st=collections.Counter();dofs=collections.Counter();fps=collections.Counter();dur=0;names[split]=set();ids[split]=set();fingerprints[split]=set()
        embeddings=torch.load(args.dataset/(split+'_text_embed.pkl'),map_location='cpu')
        with (args.output/(split+'_sequence_manifest.private.jsonl')).open('w') as out:
            for i,x in enumerate(data):
                m=x['motion'];q=np.asarray(m['dof']);n=len(q);fr=float(m['fps']);max_time=n/fr;dofs[q.shape[1]]+=1;fps[fr]+=1;dur+=max_time
                errors=[]
                for k in ['dof','root_trans_offset','root_rot','contact_mask']:
                    a=np.asarray(m[k])
                    if len(a)!=n:errors.append('length_mismatch_'+k)
                    if not np.isfinite(a).all():errors.append('nonfinite_'+k)
                norm=np.linalg.norm(m['root_rot'],axis=-1)
                qerr=float(np.max(np.abs(norm-1)))
                if qerr>1e-3:errors.append('quaternion_norm')
                if not np.isin(m['contact_mask'],[0,1]).all():errors.append('nonbinary_contact')
                if int(x['length'])!=n:st['declared_length_differs_from_motion']+=1
                name=x['feat_p'];key=canonical(name);names[split].add(key);ids[split].add(str(x['babel_sid']));groups[key.split('/')[0]]+=1
                h=hashlib.sha256()
                for k in ['dof','root_trans_offset','root_rot']:h.update(np.ascontiguousarray(m[k]).tobytes())
                mh=h.hexdigest();fingerprints[split].add(mh)
                valid_ann=0;overhang=0;max_overhang=0;missing_text=0
                for a,b,text,cats in x['frame_ann']:
                    st['segment_text_pairs']+=1
                    if not (np.isfinite(a) and np.isfinite(b) and b>a and a>=0):st['invalid_annotation_time']+=1;continue
                    overlap=max(0,min(float(b),max_time)-max(float(a),0))
                    if overlap>0:valid_ann+=1
                    else:st['annotation_without_motion_overlap']+=1
                    if b>max_time:overhang+=1;max_overhang=max(max_overhang,float(b)-max_time)
                    if not isinstance(text,str) or not text.strip():st['empty_text']+=1
                    if text not in embeddings:missing_text+=1
                st['annotations_end_after_motion']+=overhang;st['missing_cached_embeddings']+=missing_text
                st['valid_sequences']+=not errors;st['invalid_sequences']+=bool(errors)
                for e in errors:allissues[e]+=1
                record={'split':split,'index':i,'source':name,'canonical_source':key,'babel_sid':x['babel_sid'],'frames':n,'fps':fr,'dof':q.shape[1],'motion_duration_s':max_time,'annotation_duration_s':float(x['duration']),'annotations':len(x['frame_ann']),'valid_overlap_annotations':valid_ann,'max_annotation_overhang_s':max_overhang,'root_quaternion_norm_max_error':qerr,'errors':errors,'motion_array_sha256':mh,'raw_stageii_candidates':raw.get(key,[])}
                out.write(json.dumps(record)+'\n')
                if key in raw:allmatches.append({**record,'frame_ann':x['frame_ann']})
        summary['splits'][split]={'sequences':len(data),'motion_duration_s':dur,'dof_counts':dict(dofs),'fps_counts':dict(fps),'unique_motion_sha256':len(fingerprints[split]),'unique_sources':len(names[split]),'checks':dict(st)}
        h=hashlib.sha256()
        with source.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        summary['unchanged_source_files'][split]={'sha256':h.hexdigest(),'bytes':source.stat().st_size}
        del data
    summary['train_val_overlap']={'source_ids':len(names['train']&names['val']),'babel_ids':len(ids['train']&ids['val']),'exact_motion_fingerprints':len(fingerprints['train']&fingerprints['val'])}
    summary['source_dataset_counts']=dict(groups);summary['structure_errors']=dict(allissues)
    summary['raw_pairable_sequences']=len(allmatches);summary['raw_pairable_splits']=dict(collections.Counter(x['split'] for x in allmatches))
    summary['raw_missing_sequences']=sum(s['sequences'] for s in summary['splits'].values())-len(allmatches)
    summary['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (args.output/'available_raw_pairs.private.json').write_text(json.dumps(allmatches,indent=2))
    (args.output/'audit_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=pathlib.Path,required=True);p.add_argument('--raw-amass',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);main(p.parse_args())
