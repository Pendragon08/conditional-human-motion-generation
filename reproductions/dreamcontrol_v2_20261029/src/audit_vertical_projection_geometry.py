"""Independently aggregate per-frame geometry and verify root-shift identity."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PROJECT=ROOT/'vertical_support_projection_20261101'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    report=json.loads((HERE/'report.json').read_text())
    assert sha(HERE/'frames.jsonl')==report['frames_sha256']
    assert sha(PROJECT/'report.json')==report['projection_sha256']
    rows={}
    for line in (HERE/'frames.jsonl').read_text().splitlines():
        row=json.loads(line);rows.setdefault(row['sample_id'],[]).append(row)
    results=[];worst=0.
    for c in report['cases']:
        sid=c['sample_id'];these=rows[sid]
        assert len(these)==c['frames'] and [x['frame'] for x in these]==list(range(c['frames']))
        assert sha(PROJECT/(sid+'.npz'))==c['new_state_sha256']
        with np.load(PROJECT/(sid+'.npz'),allow_pickle=False) as f:
            delta=f['height_shift_m'].copy()
        assert np.array_equal(delta,np.asarray([r['shift_m'] for r in these]))
        gaps=[]
        for tag in ('before','after'):
            arr=np.asarray([r[tag]['gaps_m'] for r in these]);gaps.append(arr)
            np.testing.assert_allclose(arr.min(axis=0),c['foot_min_gap_'+tag+'_m'],rtol=0,atol=1e-14)
            assert (arr<0).sum(axis=0).tolist()==c[tag+'_below']
            for kind in ('floor','self'):
                assert sum(bool(r[tag][kind+'_contacts']) for r in these)==c[tag+'_'+kind]
        err=float(np.max(np.abs(gaps[1]-gaps[0]-delta[:,None])));worst=max(worst,err)
        assert err < 1e-8
        changed=sum(r['before']['self_contacts']!=r['after']['self_contacts'] for r in these)
        assert changed==c['self_changed']
        results.append({'sample_id':sid,'frames':len(these),'max_gap_identity_error_m':err,
                        'floor_contact_frames_before':c['before_floor'],'floor_contact_frames_after':c['after_floor'],
                        'self_contact_frames_before':c['before_self'],'self_contact_frames_after':c['after_self']})
    out={'report_sha256':sha(HERE/'report.json'),'frames_sha256':sha(HERE/'frames.jsonl'),
         'script_sha256':sha(Path(__file__)),'frames_verified':sum(x['frames'] for x in results),
         'max_gap_identity_error_m':worst,'cases':results,'training_ready':False,
         'limitations':'Reaggregates the same saved engine outputs; does not independently rerun collision detection or establish a real floor.'}
    with (HERE/'audit.json').open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out))

if __name__=='__main__':main()
