"""Read-only Step2 prerequisite audit. Never launches or resumes training.

Current progress receipts are useful diagnostics, not acceptance evidence.
Passing this checker alone is not an automatic launcher or a numerical witness.
"""
import argparse,hashlib,json,pathlib

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def audit(plan,acceptance):
    blockers=[]
    if acceptance is None:
        return {'ready':False,'training_launched':False,'blockers':['No accepted dataset and69D runtime evidence exists; preprocessing progress is insufficient']}
    data=acceptance.get('dataset',{});runtime=acceptance.get('runtime',{})
    for key,value in {'dof':29,'feature_dim':69,'fps':50,'physical_filter_passed':True,'pairing_validated':True,'train_val_source_disjoint':True,'normalization_fit_split':'train','missing_public_sources':0}.items():
        actual=data.get(key)
        valid=(actual is value) if isinstance(value,bool) else (type(actual) in [int,float] and actual==value) if isinstance(value,(int,float)) else actual==value
        if not valid:blockers.append('dataset.'+key)
    if data.get('train_sequences',0)<1 or data.get('val_sequences',0)<1:blockers.append('Nonempty accepted train/val splits required')
    for key in ['vae69_forward_backward_passed','ldm69_conditioning_and_gradients_passed','fk29_geometry_loss_passed','fresh_document_witness_passed','paper_code_differences_audited']:
        if runtime.get(key) is not True:blockers.append('runtime.'+key)
    # Require real file references; a hand-written true flag cannot satisfy these.
    references=acceptance.get('artifacts',{})
    for key in ['filter_evidence','accepted_train_manifest','accepted_val_manifest','train_only_statistics','clip_cache_receipt','runtime69_witness','paper_code_audit']:
        entry=references.get(key,{})
        if not entry.get('path') or not entry.get('sha256'):
            blockers.append('Missing artifact '+key);continue
        path=pathlib.Path(entry['path'])
        if not path.is_file() or digest(path)!=entry['sha256']:blockers.append('Missing or changed artifact '+key)
    # The filter evidence must itself pass the existing strict protocol verifier.
    if not blockers:
        from apply_tracking_filter import apply
        evidence=json.loads(pathlib.Path(references['filter_evidence']['path']).read_text(encoding='utf-8'))
        try:
            decisions=apply(evidence)
            if not any(x['decision']=='retain' for x in decisions):blockers.append('No retained motions in actual filter evidence')
        except Exception as error:blockers.append('Dynamic-filter evidence rejected: '+repr(error))
    if not plan.get('launch_enabled'):blockers.append('69D launch recipe remains disabled pending completed preparation and validation')
    return {'ready':not blockers,'training_launched':False,'blockers':blockers,'note':'This is a prerequisite audit; accepted motion contents and runtime must still be checked by the actual launcher'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=pathlib.Path,required=True);parser.add_argument('--acceptance',type=pathlib.Path);parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
    acceptance=json.loads(args.acceptance.read_text(encoding='utf-8')) if args.acceptance and args.acceptance.exists() else None
    result=audit(json.loads(args.plan.read_text(encoding='utf-8')),acceptance)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=True))
    raise SystemExit(0 if result['ready'] else 2)
