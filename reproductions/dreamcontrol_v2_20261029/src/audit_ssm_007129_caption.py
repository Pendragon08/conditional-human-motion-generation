"""Independently recompute all reviewed wrist-height samples and text binding."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
DATA=ROOT/'ssm_new6_20hz_resume_20261101';CURVES=ROOT/'ssm_new5_caption_curves_20261101';HUMAN=ROOT/'ssm_new6_human_20261101';HERE=Path(__file__).resolve().parent
ZIP=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sid='007129';path=HERE/(sid+'_review.json');review=json.loads(path.read_text());artifact=DATA/(sid+'.npz');curve=CURVES/(sid+'_curves.npz')
    assert sha(artifact)==review['candidate_sha256'] and sha(curve)==review['curve_sha256']
    assert sha(CURVES/(sid+'.png'))==review['figure_sha256'] and sha(CURVES/'report.json')==review['curve_report_sha256']
    assert sha(ZIP)==review['caption_archive_sha256']
    with zipfile.ZipFile(ZIP) as z:raw=z.read('texts/'+sid+'.txt')
    fields=raw.decode('utf8').splitlines()[review['selected_line']].split('#')
    assert hashlib.sha256(raw).hexdigest()==review['caption_member_sha256'] and fields[0]==review['caption'] and fields[2:]==review['tags']
    with np.load(artifact,allow_pickle=False) as f:t=f['sample_times'];robot=f['points_z_up_world'];feature=f['features']
    with np.load(curve,allow_pickle=False) as f:hc=f['human'];rc=f['robot'];assert np.array_equal(f['sample_times'],t)
    with np.load(HUMAN/sid/'human_rawfps.npz',allow_pickle=False) as f:st=f['source_times'];human=f['human']
    assert len(t)==len(feature)+1 and len(hc)==len(rc)==len(t)
    hcalc=[];rcalc=[]
    for i,at in enumerate(t):
        j=int(np.searchsorted(st,at,side='right')-1)
        if j<0 or j>=len(st)-1:raise ValueError('sample outside support')
        alpha=float((at-st[j])/(st[j+1]-st[j]));hh=[];rr=[]
        for wrist,shoulder in ((20,16),(21,17)):
            z0=float(human[j,wrist,2]-human[j,shoulder,2]);z1=float(human[j+1,wrist,2]-human[j+1,shoulder,2])
            hh.append((1-alpha)*z0+alpha*z1);rr.append(float(robot[i,wrist,2]-robot[i,shoulder,2]))
        hcalc.append(hh);rcalc.append(rr)
    hcalc=np.array(hcalc);rcalc=np.array(rcalc)
    maxdiff=max(float(np.max(abs(hcalc-hc[:,:2]))),float(np.max(abs(rcalc-rc[:,:2]))))
    if maxdiff>1e-6:raise ValueError('wrist curves do not match states')
    ranges={k:float(np.ptp(a[:,col])) for k,a,col in [('human_left',hcalc,0),('human_right',hcalc,1),('robot_left',rcalc,0),('robot_right',rcalc,1)]}
    if any(abs(ranges[k]-review['wrist_height_ranges_m'][k])>1e-6 for k in ranges):raise ValueError('range differs')
    result={'sample_id':sid,'receipt_sha256':sha(path),'sample_count':len(t),'max_wrist_curve_difference_m':maxdiff,
        'recomputed_wrist_height_ranges_m':ranges,'training_ready':False,
        'limitations':['Manual semantics are not independently reviewed; peak finder and model reconstruction are shared.',
            'Wrist proxy is not hand geometry, source clock, ground contact, dynamics or training acceptance.']}
    with (HERE/(sid+'_audit.json')).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
