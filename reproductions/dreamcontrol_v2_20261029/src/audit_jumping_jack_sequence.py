"""Recompute all 46 descriptive frames with scalar coordinates and same robot FK."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from g1_fk import URDFFK

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_family_disjoint_preview_v2_20261101'
SRC=ROOT/'ssm_three_human_candidates_20261031'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def manual(p):
    ans=[]
    initial=[float(p[0,7,2]),float(p[0,8,2])]
    for row in p:
        dist=math.sqrt(sum((float(row[7,k])-float(row[8,k]))**2 for k in (0,1)))
        ans.append([float(row[20,2])-float(row[16,2]),float(row[21,2])-float(row[17,2]),dist,
                    float(row[7,2])-initial[0],float(row[8,2])-initial[1]])
    return np.asarray(ans)


def main():
    rep=json.loads((HERE/'report.json').read_text());artifact=HERE/'observables.npz'
    assert sha(artifact)==rep['observables_sha256'] and sha(URDF)==rep['urdf_sha256']
    rec=next(x for x in json.loads((BASE/'state_manifest.json').read_text())['records'] if x['sample_id']=='003017')
    assert sha(BASE/rec['state_path'])==rep['state_sha256']
    with np.load(BASE/rec['state_path'],allow_pickle=False) as f:
        root=f['root_wxyz_xyz'].copy();q=f['joint_pos'].copy();t=f['sample_times'].copy();names=f['joint_names'].tolist()
    bot=URDFFK(URDF).forward(root[:,4:],root[:,[1,2,3,0]],q,names)
    witness=SRC/'003017_joint_witness.npz';assert sha(witness)==rep['human_source_evidence']['sha256']
    with np.load(witness,allow_pickle=False) as f:st=f['source_times'].copy();hm=f['original_joints_humanml'].astype(float)
    human=np.empty((len(t),22,3))
    for i,time in enumerate(t):
        hi=min(int(np.searchsorted(st,time,side='right')),len(st)-1);lo=max(hi-1,0)
        weight=float((time-st[lo])/(st[hi]-st[lo]))
        for j in range(22):
            # HumanML (-x,z,y) to source x/y/z; scalar interpolation is independent of np.interp.
            for c,src in enumerate((0,2,1)):
                human[i,j,c]=(-1 if c==0 else 1)*((1-weight)*hm[lo,j,src]+weight*hm[hi,j,src])
    with np.load(artifact,allow_pickle=False) as f:
        np.testing.assert_allclose(f['times'],t,atol=1e-12,rtol=0)
        herror=float(np.abs(f['human']-manual(human)).max());rerror=float(np.abs(f['robot']-manual(bot)).max())
        assert herror<1e-10 and rerror<1e-10
        values={}
        for side in ('human','robot'):
            z=f[side]
            values[side]={str(i):{'wrist_above_shoulder_m':[float(z[i,0]),float(z[i,1])],
                                  'ankle_horizontal_separation_m':float(z[i,2])} for i in (0,13,17,23,28)}
    report={'report_sha256':sha(HERE/'report.json'),'observables_sha256':sha(artifact),
            'frames_recomputed':len(t),'human_max_abs_error':herror,'robot_max_abs_error':rerror,
            'selected_frames':values,'training_ready':False,'script_sha256':sha(Path(__file__)),
            'limitations':'Uses same fitted human source and robot FK; no independent ground, biomechanics, jumping or caption acceptance.'}
    with (HERE/'audit.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()
