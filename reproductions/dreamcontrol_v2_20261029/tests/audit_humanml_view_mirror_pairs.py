"""Verify counterpart handling in the supplementary exact-source view."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    view=OUT/'view.jsonl';report=json.loads((OUT/'report.json').read_text())
    if sha(view)!=report['view_sha256']:raise ValueError('view hash mismatch')
    rows={r['motion_id']:r for r in map(json.loads,view.read_text().splitlines())}
    missing=[];mismatched=[];pairs=0
    for mid,row in rows.items():
        counterpart=mid[1:] if mid.startswith('M') else 'M'+mid
        if counterpart not in rows:missing.append(mid);continue
        other=rows[counterpart]
        if mid.startswith('M'):continue
        pairs+=1
        if row['mirrored'] or not other['mirrored'] or row['exact_source_path']!=other['exact_source_path'] or row['official_split']!=other['official_split'] or row['view_disposition']!=other['view_disposition']:
            mismatched.append({'id':mid,'mirror_id':counterpart})
    result={'script_sha256':sha(__file__),'view_sha256':sha(view),'paired_ids':pairs,'unpaired_ids':sorted(missing),'mismatched_pairs':mismatched,
            'limitations':'Checks counterpart labels and view decisions only; does not generate or validate mirrored geometry, motion, coordinates, semantics or training quality.'}
    with (OUT/'mirror_pair_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps({'pairs':pairs,'unpaired':len(missing),'mismatched':len(mismatched)}),flush=True)
if __name__=='__main__':main()
