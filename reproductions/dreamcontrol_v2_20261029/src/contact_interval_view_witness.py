"""Check the isolated bin view against previously audited GRAB label runs."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from contact_interval_view import contact_interval_view

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
source=ROOT/'grab_hand_event_sampling_20261031/report.json'
audit=ROOT/'grab_hand_event_sampling_20261031/audit.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='21d0acce69896f0bab4ab06c8badd5322051c637b323e17ff2d308e82359831f'
assert sha(audit)=='735b8996107bb587e1455c4060c3c927fd2e9c5009e9f926cb444e30c7fd26e4'
data=json.loads(source.read_text()); checked=0; example=None
for motion in data['motions']:
    for hand in ('left','right'):
        state=np.zeros((motion['frames'],1),dtype=bool)
        for a,b in motion[hand]['intervals_half_open']:state[a:b]=True
        valid=np.ones_like(state)
        for phase in range(6):
            bins=list(range(phase,motion['frames'],6))
            if bins[0]!=0: bins.insert(0,0)
            if bins[-1]!=motion['frames']:bins.append(motion['frames'])
            out=contact_interval_view(state,valid,bins,declared_source_fps=motion['declared_fps'])
            start=out['source_frame_begin_inclusive']; end=out['source_frame_end_exclusive']
            full=((start-phase)%6==0)&(end-start==6)
            missed_bin=np.sum(full & out['any_known_positive'][:,0] & ~out['first_sample_positive'][:,0])
            assert missed_bin==motion[hand]['phase_comparison'][phase]['full_bins_with_any_not_sampled']
            assert start[0]==0 and end[-1]==motion['frames'] and np.array_equal(start[1:],end[:-1])
            if motion['member']=='s4/waterbottle_drink_1.npz' and hand=='right' and phase==2:
                i=np.flatnonzero((start<=185)&(end>185))[0]
                example={'bin_frames_half_open':[int(start[i]),int(end[i])], 'first_label':bool(out['first_sample_positive'][i,0]),'any_label':bool(out['any_known_positive'][i,0]),'positive_fraction':float(out['positive_frame_fraction_lower'][i,0]),'source_labels':state[start[i]:end[i],0].astype(int).tolist()}
            checked+=1
assert checked==48 and example is not None
result={'cases_checked':checked,'s4_phase_2':example,'source_report_sha256':sha(source),'audit_sha256':sha(audit),'view_script_sha256':sha(Path(sys.modules['contact_interval_view'].__file__)),'witness_script_sha256':sha(Path(__file__)),'limitations':'Reconstructed discrete label runs; all frames treated as observed solely for this check. Discrete bin OR differs from a sample and is not continuous-time contact, force, established physics, author pipeline, or training data.'}
with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
