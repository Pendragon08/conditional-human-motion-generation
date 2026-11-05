"""Check receipt, caption archive, sample timing and recorded review evidence."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
CAN=ROOT/'ssm_anchored_20hz_candidates_20261031'
HUMAN=ROOT/'ssm_rawfps_human_batch_20261031/011120'
REVIEW=ROOT/'ssm_anchored_caption_review_20261031'
ZIP=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    r=json.loads((HERE/'semantic_review.json').read_text()); assert r['sample_id']=='011120'
    assert sha(ZIP)==r['caption_archive_sha256']
    with zipfile.ZipFile(ZIP) as z: member=z.read('texts/011120.txt')
    assert hashlib.sha256(member).hexdigest()==r['caption_member_sha256']
    parts=member.decode('utf8').splitlines()[r['selected_line']].split('#')
    assert parts[0]==r['caption'] and parts[2:4]==r['tags']==['0.0','0.0']
    assert sha(CAN/'011120.npz')==r['candidate_sha256'] and sha(HUMAN/'human_rawfps.npz')==r['human_sha256']
    assert sha(REVIEW/'report.json')==r['review_sha256'] and sha(REVIEW/'event_audit.json')==r['event_audit_sha256']
    review=json.loads((REVIEW/'report.json').read_text()); c=next(x for x in review['cases'] if x['id']=='011120')
    assert all(sha(REVIEW/p['file'])==p['sha256'] for p in c['pages'])
    with np.load(HUMAN/'human_rawfps.npz',allow_pickle=False) as f: st=f['source_times'].copy(); p=f['human'].astype(float)
    with np.load(CAN/'011120.npz',allow_pickle=False) as f: t=f['sample_times'].copy(); x=f['features'].copy()
    np.testing.assert_allclose(r['sample_range_s'],[t[0],t[-1]],rtol=0,atol=1e-12)
    np.testing.assert_allclose(r['source_range_s'],[st[0],st[-1]],rtol=0,atol=1e-12)
    np.testing.assert_allclose(r['excluded_head_tail_s'],[t[0]-st[0],st[-1]-t[-1]],rtol=0,atol=1e-12)
    ends=[max(sum(float(p[k,j,c]-p[k-1,j,c])**2 for c in range(3))**.5 for j in range(22)) for k in (1,len(st)-1)]
    np.testing.assert_allclose(ends,r['endpoint_max_adjacent_human_joint_displacement_m'],rtol=0,atol=1e-12)
    assert len(t)==101 and x.shape==(100,263) and r['training_ready'] is False
    out={'receipt_sha256':sha(HERE/'semantic_review.json'),'checked_frame_count':len(t),
         'caption_member_sha256':hashlib.sha256(member).hexdigest(),'image_hashes_verified':len(c['pages']),
         'training_ready':False,'script_sha256':sha(Path(__file__)),
         'limitations':'Checks recorded evidence, not an independent semantic reviewer, physical truth, source identity or acceptance.'}
    with (HERE/'independent_audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
