"""Illustrate frame-rate dependence of the existing HumanML foot proxy."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from feature263 import encode

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
TRAJ=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
FILES=('grab_s4_drink_human_source_20261030','grab_s5_cup_human_source_20261030')
SPEED=.05  # exploratory, not a paper-validated threshold
LOW=(SPEED/20)**2; HIGH=(SPEED/120)**2
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def direct(p,threshold):
    return ((np.diff(p[:,[7,10,8,11]],axis=0)**2).sum(axis=2)<threshold).astype(np.int8)
def main():
    output=[]
    for name in FILES:
        directory=ROOT/name;r=json.loads((directory/'report.json').read_text()); source=directory/'human_source.npz'; assert sha(source)==r['artifact_sha256']
        with np.load(source,allow_pickle=False) as f:
            p=f['human_joints'].astype(float);frame=f['input_frame_indices'];fps=float(f['input_fps'])
        assert p.shape==(120,22,3) and fps==120 and np.array_equal(np.diff(frame),np.ones(119,dtype=int))
        low=p[::6]; a=encode(p,foot_velocity_squared_threshold=HIGH)[:,-4:]; b=encode(p,foot_velocity_squared_threshold=LOW)[:,-4:]; c=encode(low,foot_velocity_squared_threshold=LOW)[:,-4:]
        assert np.array_equal(a,direct(p,HIGH)) and np.array_equal(b,direct(p,LOW)) and np.array_equal(c,direct(low,LOW))
        align=a[:114].reshape(19,6,4)
        output.append({'source':name,'source_sha256':sha(source),'source_report_sha256':sha(directory/'report.json'),'frame_first_last':[int(frame[0]),int(frame[-1])],'source_frames':len(p),'source_fps_declared':fps,'target_samples':len(low),'target_feature_rows':len(c),'target_last_sample_source_frame':int(frame[114]),'unrepresented_tail_source_frames':list(map(int,frame[115:])),'high_scaled_low_speed_counts':a.sum(axis=0).astype(int).tolist(),'high_unscaled_low_speed_counts':b.sum(axis=0).astype(int).tolist(),'target_low_speed_counts':c.sum(axis=0).astype(int).tolist(),'high_scaled_vs_unscaled_channel_decisions':int(np.count_nonzero(a!=b)),'target_vs_high_all_six_channel_decisions':int(np.count_nonzero(c!=align.all(axis=1))),'target_vs_high_any_six_channel_decisions':int(np.count_nonzero(c!=align.any(axis=1)))})
    # Exact line motion: same physical speed, threshold below speed.
    v=.06;p=np.arange(120)[:,None]*np.array([[v/120,0,0]]);t20=np.diff(p[::6],axis=0);t120=np.diff(p,axis=0)
    assert np.all((t20*t20).sum(axis=1)>=LOW) and np.all((t120*t120).sum(axis=1)<LOW) and np.all((t120*t120).sum(axis=1)>=HIGH)
    result={'source_code_sha256':sha(TRAJ/'data_loaders/humanml/scripts/motion_process.py'),'encoder_sha256':sha(Path(sys.modules['feature263'].__file__)),'script_sha256':sha(Path(__file__)),'nominal_speed_threshold_mps':SPEED,'threshold_20hz_displacement_squared_m2':LOW,'threshold_120hz_displacement_squared_m2':HIGH,'synthetic_constant_speed_mps':v,'synthetic_20hz_proxy':False,'synthetic_120hz_same_threshold_proxy':True,'synthetic_120hz_scaled_threshold_proxy':False,'samples':output,'limitations':'Foot bits are squared per-frame displacement only; no height, force, true FPS or real contact. Two nonrandom GRAB human-model windows, not robot data or author v2. Rate scaling is exact only for constant-velocity translation, not a prescription for real motion or anti-aliasing. Selected fixed thresholds are exploratory.'}
    with Path(sys.argv[1]).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
if __name__=='__main__':main()
