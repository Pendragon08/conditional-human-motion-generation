"""Independent direct ZIP/NPZ verification of the integrated preflight."""
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    cfg=json.loads((HERE/'config.json').read_text());c=cfg['caption_archive_binding']
    manifest=json.loads(Path(cfg['manifest']).read_text())['records'];states=json.loads(Path(c['state_manifest']).read_text())['records']
    receipts=json.loads(Path(c['evidence_manifest']).read_text())['receipts']
    state_by_id={x['sample_id']:x for x in states};evidence_by_id={x['sample_id']:x for x in receipts}
    result=json.loads(next(x for x in (HERE/'preflight.log').read_text().splitlines() if x.startswith('{')))
    binding=result['caption_archive_binding'];assert binding['training_ready'] is False
    archive=Path(c['archive']);assert sha(archive)==binding['archive_sha256']
    by_id={x['sample_id']:x for x in binding['cases']};cases=[]
    with zipfile.ZipFile(archive) as z:
        for row in manifest:
            sid=row['sample_id'];e=evidence_by_id[sid];state=state_by_id[sid];record=by_id[sid]
            receipt=Path(c['evidence_root'])/e['receipt_path'];assert sha(receipt)==e['receipt_sha256']
            r=json.loads(receipt.read_text());member=re.search(r'texts/[0-9]{6}\.txt',row['caption_source']).group()
            raw=z.read(member);assert hashlib.sha256(raw).hexdigest()==r['caption_member_sha256']
            fields=raw.decode('utf8').splitlines()[r['selected_line']].split('#')
            assert fields[0]==row['caption']==r['caption'] and fields[2:]==r['tags']==record['raw_tags']
            path=Path(c['state_root'])/state['state_path'];assert sha(path)==state['state_sha256']
            with np.load(path,allow_pickle=False) as f:t=f['sample_times'].astype(float)
            assert len(t)-1==record['feature_intervals'] and abs(t[0]-record['nominal_sample_range_s'][0])<1e-10
            assert abs(t[-1]-record['nominal_sample_range_s'][1])<1e-10
            assert fields[2:] == ['0.0','0.0'] and record['tag_interpretation']=='whole_clip_convention'
            assert r['source_range_s'][0]<=t[0]<=t[-1]<=r['source_range_s'][1]
            cases.append({'sample_id':sid,'caption_archive_member':member,'line_zero_based':r['selected_line'],
                          'sample_points':len(t),'feature_intervals':len(t)-1,'nominal_range_s':[float(t[0]),float(t[-1])]})
    assert len(cases)==4 and 'Ran 3 tests' in (HERE/'tests.log').read_text() and 'OK' in (HERE/'tests.log').read_text()
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    out={'config_sha256':sha(HERE/'config.json'),'preflight_log_sha256':sha(HERE/'preflight.log'),
         'archive_sha256':sha(archive),'module_sha256':sha(HERE/'audit_caption_archive_binding.py'),
         'trainer_sha256':sha(HERE/'train_robot_prior.py'),'cases':cases,'run_rejected':True,'training_ready':False,
         'limitations':'Rechecks same zip/state evidence and nominal timeline; does not establish semantics or actual source clock.'}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
