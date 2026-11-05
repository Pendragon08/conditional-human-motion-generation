"""Record a narrow manual semantic review for one caption; not training approval."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SRC=ROOT/'ssm_rawfps_human_batch_20261031/011120'
CAN=ROOT/'ssm_anchored_20hz_candidates_20261031'
REVIEW=ROOT/'ssm_anchored_caption_review_20261031'
ZIP=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    human=json.loads((SRC/'human_report.json').read_text())
    candidate=json.loads((CAN/'011120_report.json').read_text())
    ca=json.loads((CAN/'independent_audit.json').read_text())
    review=json.loads((REVIEW/'report.json').read_text())
    event=json.loads((REVIEW/'event_measurements.json').read_text())
    ea=json.loads((REVIEW/'event_audit.json').read_text())
    assert sha(REVIEW/'event_measurements.json')==ea['event_report_sha256']
    assert sha(REVIEW/'report.json')==event['review_report_sha256']
    reviewed=next(c for c in review['cases'] if c['id']=='011120')
    measured=next(c for c in event['cases'] if c['id']=='011120')
    assert len(reviewed['pages'])==6 and all(sha(REVIEW/p['file'])==p['sha256'] for p in reviewed['pages'])
    assert next(c for c in ca['cases'] if c['id']=='011120')['output_sha256']==sha(CAN/'011120.npz')
    assert reviewed['input_sha256']==candidate['output_sha256']==sha(CAN/'011120.npz')
    assert human['artifact_sha256']==reviewed['source_sha256']==sha(SRC/'human_rawfps.npz')
    with zipfile.ZipFile(ZIP) as z: raw=z.read('texts/011120.txt')
    assert hashlib.sha256(raw).hexdigest()==human['caption_member_sha256']
    lines=[line.split('#') for line in raw.decode('utf8').splitlines()]
    assert lines[0][0]=='the person is doing arm and squat exercises.' and lines[0][2:4]==['0.0','0.0']
    assoc=candidate['caption_records'][0]
    assert assoc['caption']==lines[0][0] and assoc['raw_tags']==lines[0][2:4]
    assert assoc['association']['matching_feature_intervals']==list(range(100))
    with np.load(SRC/'human_rawfps.npz',allow_pickle=False) as f:
        source_times=f['source_times'].copy(); poses=f['human'].astype(float)
    with np.load(CAN/'011120.npz',allow_pickle=False) as f:
        times=f['sample_times'].copy()
    assert len(times)==101 and times[0]>=source_times[0] and times[-1]<=source_times[-1]
    # Descriptive endpoint changes, not a claim of measured stillness.
    first_delta=float(np.linalg.norm(poses[1]-poses[0],axis=1).max())
    last_delta=float(np.linalg.norm(poses[-1]-poses[-2],axis=1).max())
    assert measured['human']['left']['min_knee_frame'] in range(50,66)
    assert measured['robot']['left']['min_knee_frame'] in range(50,66)
    out={'sample_id':'011120','caption':lines[0][0],'caption_member_sha256':human['caption_member_sha256'],
         'caption_archive_sha256':sha(ZIP),'selected_line':0,'tags':lines[0][2:4],
         'candidate_sha256':sha(CAN/'011120.npz'),'human_sha256':sha(SRC/'human_rawfps.npz'),
         'review_sha256':sha(REVIEW/'report.json'),'event_audit_sha256':sha(REVIEW/'event_audit.json'),
         'sample_range_s':[float(times[0]),float(times[-1])],
         'source_range_s':[float(source_times[0]),float(source_times[-1])],
         'excluded_head_tail_s':[float(times[0]-source_times[0]),float(source_times[-1]-times[-1])],
         'endpoint_max_adjacent_human_joint_displacement_m':[first_delta,last_delta],
         'manual_semantic_assessment':{'status':'supported_for_this_exported_motion',
           'reviewer':'Codex assistant; visual inspection of every 20 Hz frame from two projections',
           'reason':'Both fitted human and G1 show arm raising, knee bending, and arm lowering; the selected broad original text does not claim a wall, speed, force, or floor contact.',
           'semantic_accepted_for_exported_samples':True},
         'training_ready':False,'script_sha256':sha(Path(__file__)),
         'limitations':['Semantic assessment is a bounded visual judgment, not ground-truth action recognition.',
           'The full-clip annotation is clipped at both ends; report gives omitted durations and endpoint changes.',
           'Source identity, actual FPS, body-model equivalence, licensing for the full pipeline, collision and dynamics remain separately unaccepted.',
           'No new split assignment, training manifest, training run or physical claim.']}
    with (HERE/'semantic_review.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'sample_id':'011120','semantic_supported':True,'training_ready':False,'endpoint_delta': [first_delta,last_delta]}))


if __name__=='__main__':main()
