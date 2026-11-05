"""Independently check all human/robot right-ankle review curves and binding."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
DATA=ROOT/'ssm_new6_20hz_resume_20261101';CURVES=ROOT/'ssm_new5_caption_curves_20261101';HUMAN=ROOT/'ssm_new6_human_20261101';HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def val(p):
    a=(p[2]-p[1])+(p[17]-p[16]);a[2]=0.;a=a/np.linalg.norm(a)
    d=p[8]-p[2]
    return float(np.dot(d,a)),float(d[2])
def main():
    sid='004028';path=HERE/(sid+'_annotation.json');review=json.loads(path.read_text());artifact=DATA/(sid+'.npz');curve=CURVES/(sid+'_curves.npz')
    if sha(artifact)!=review['feature_sha256'] or sha(curve)!=review['curve_sha256'] or sha(CURVES/(sid+'.png'))!=review['figure_sha256']:
        raise ValueError('review input changed')
    with np.load(artifact,allow_pickle=False) as f:t=f['sample_times'];robot=f['points_z_up_world'];feat=f['features'];rawframes=f['raw_source_frames']
    with np.load(curve,allow_pickle=False) as f:hc=f['human'];rc=f['robot'];assert np.array_equal(f['sample_times'],t)
    with np.load(HUMAN/sid/'human_rawfps.npz',allow_pickle=False) as f:st=f['source_times'];human=f['human']
    if len(feat)+1!=len(t) or [int(rawframes[0]),int(rawframes[-1])]!=review['source_frame_bounds']:
        raise ValueError('frame support differs')
    np.testing.assert_allclose(t*review['declared_source_fps'],review['sample_source_frame_positions'],rtol=0,atol=1e-8)
    hv=[];rv=[]
    for i,at in enumerate(t):
        j=int(np.searchsorted(st,at,side='right')-1)
        if j<0 or j>=len(st)-1:raise ValueError('outside support')
        weight=float((at-st[j])/(st[j+1]-st[j]))
        p=np.asarray(human[j],dtype=np.float64)*(1-weight)+np.asarray(human[j+1],dtype=np.float64)*weight
        hv.append(val(p));rv.append(val(robot[i].copy()))
    hv=np.array(hv);rv=np.array(rv)
    diff=max(float(np.max(abs(hv-hc[:,[5,9]]))),float(np.max(abs(rv-rc[:,[5,9]]))))
    if diff>1e-6:raise ValueError('right ankle curves differ')
    events={}
    for name,a in [('human',hv),('robot',rv)]:
        ix=int(np.argmax(a[:,0]));events[name]={'lateral_peak_index':ix,'lateral_from_start_m':float(a[ix,0]-a[0,0]),
            'lateral_end_minus_start_m':float(a[-1,0]-a[0,0]),'height_rise_m':float(np.max(a[:,1])-a[0,1]),
            'height_end_minus_start_m':float(a[-1,1]-a[0,1])}
        for k in events[name]:
            if abs(events[name][k]-review['motion_events'][name][k])>1e-6:raise ValueError('event summary differs')
    result={'sample_id':sid,'annotation_sha256':sha(path),'sample_count':len(t),'max_curve_difference_m':diff,
        'events':events,'training_ready':False,'limitations':['Independent coordinate recomputation, same body model and input robot points.',
        'Manual label is subjective, no independent semantic review, real clock, contact or physics acceptance.']}
    with (HERE/(sid+'_audit.json')).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
