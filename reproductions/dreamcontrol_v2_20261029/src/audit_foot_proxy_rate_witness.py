"""Recount foot displacement decisions from previously hashed joint windows."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=Path(sys.argv[1]);r=json.loads(report.read_text());speed=r['nominal_speed_threshold_mps'];assert speed==.05
tested=0
for rec in r['samples']:
    folder=ROOT/rec['source'];source=folder/'human_source.npz';assert sha(source)==rec['source_sha256'] and sha(folder/'report.json')==rec['source_report_sha256']
    with np.load(source,allow_pickle=False) as d:p=d['human_joints'];frames=d['input_frame_indices'];assert float(d['input_fps'])==120
    def mask(steps,rate):
        bits=[]
        for j,k in zip(steps[:-1],steps[1:]):
            row=[]
            for joint in (7,10,8,11):
                delta=[float(p[k,joint,a])-float(p[j,joint,a]) for a in range(3)]
                row.append(sum(v*v for v in delta)<(speed/rate)**2)
            bits.append(row)
        return np.array(bits,dtype=bool)
    high=mask(list(range(120)),120);unscaled=mask(list(range(120)),20);low=mask(list(range(0,120,6)),20)
    blocks=high[:114].reshape(19,6,4)
    assert high.sum(axis=0).tolist()==rec['high_scaled_low_speed_counts']
    assert unscaled.sum(axis=0).tolist()==rec['high_unscaled_low_speed_counts']
    assert low.sum(axis=0).tolist()==rec['target_low_speed_counts']
    assert int(np.count_nonzero(high!=unscaled))==rec['high_scaled_vs_unscaled_channel_decisions']
    assert int(np.count_nonzero(low!=np.all(blocks,axis=1)))==rec['target_vs_high_all_six_channel_decisions']
    assert int(np.count_nonzero(low!=np.any(blocks,axis=1)))==rec['target_vs_high_any_six_channel_decisions']
    assert rec['target_last_sample_source_frame']==int(frames[114]) and rec['unrepresented_tail_source_frames']==frames[115:].tolist();tested+=1
out={'status':'passed','report_sha256':sha(report),'samples_checked':tested,'checks':'scalar world-coordinate displacement, rates, block counts and endpoint; audit does not independently execute feature extractor or verify actual contact/FPS','script_sha256':sha(Path(__file__))}
with Path(sys.argv[2]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
