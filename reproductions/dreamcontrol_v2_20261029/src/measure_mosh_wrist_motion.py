"""Descriptive whole-crop wrist trajectories for manual caption review."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMAN=ROOT/'mosh_000798_fullrate_20261031/output'
CROP=ROOT/'mosh_000798_crop_candidate_20261031'
REVIEW=ROOT/'mosh_all_frames_review_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def measure(p):
    out={}
    for side,s,w in [('left',16,20),('right',17,21)]:
        q=p[:,w]-p[:,s];delta=np.diff(q,axis=0)
        out[side]={'coordinate_ranges_m':np.ptp(q,axis=0).tolist(),
                   'cumulative_displacement_m':float(np.linalg.norm(delta,axis=1).sum()),
                   'max_interval_displacement_m':float(np.linalg.norm(delta,axis=1).max())}
    return out


def main():
    review=json.loads((REVIEW/'report.json').read_text());hr=json.loads((HUMAN/'human_report.json').read_text())
    candidate=CROP/'candidate.npz';source=HUMAN/'human_120hz.npz'
    assert sha(candidate)==review['candidate_sha256'] and sha(source)==hr['artifact_sha256']
    with np.load(candidate,allow_pickle=False) as d:r=d['points_z_up_world'].copy();frames=d['frames_20hz'].copy()
    with np.load(source,allow_pickle=False) as d:h=d['human'].copy();indices=d['source_frames'].copy()
    positions=np.searchsorted(indices,frames);assert np.array_equal(indices[positions],frames)
    human=h[positions];assert r.shape==human.shape==(149,22,3)
    report={'sample_id':'MOSH_000798_crop','candidate_sha256':sha(candidate),'human_sha256':sha(source),
            'review_sha256':sha(REVIEW/'report.json'),'human':measure(human),'robot':measure(r),
            'training_ready':False,'script_sha256':sha(Path(__file__)),
            'limitations':'Wrist relative to shoulder, not hand/finger motion, speed truth, physical contact or automatic dance classification.'}
    with (HERE/'wrist_motion.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()
