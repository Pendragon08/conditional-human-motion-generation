"""Check saved BML feature artifact without re-running FK or encoding."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    label=sys.argv[1]; here=Path.cwd()/label
    solve=ROOT/'bml_global_candidates_20261031'/label
    report=json.loads((here/'report.json').read_text())
    assert report['training_ready'] is False and report['label']==label
    assert report['script_sha256']==sha(Path.cwd()/'export.py')
    assert report['feature_sha256']==sha(here/'candidate.npz')
    assert report['source_trajectory_sha256']==sha(solve/'trajectory.npz')
    assert report['source_report_sha256']==sha(solve/'report.json')
    assert report['source_audit_sha256']==sha(solve/'audit_report.json')
    with np.load(solve/'trajectory.npz',allow_pickle=False) as s, np.load(here/'candidate.npz',allow_pickle=False) as d:
        for key in ('source_frames','source_times','root_wxyz_xyz','joint_pos','joint_names'):
            np.testing.assert_array_equal(s[key],d[key])
        f=d['features'].copy(); p=d['points_z_up_world'].copy(); decoded=d['decoded_points'].copy()
        frames=d['source_frames'].copy(); times=d['source_times'].copy()
    assert f.shape==(59,263) and p.shape==(60,22,3) and decoded.shape==(59,22,3)
    assert np.isfinite(f).all() and np.isin(f[:,-4:],[0.,1.]).all()
    assert np.all(np.diff(frames)==6)
    np.testing.assert_allclose(times,frames/120.,atol=1e-12,rtol=0)
    # Independent normalization arithmetic: Z-up to Y-up, ground, initial yaw.
    v=p[:,:,[0,2,1]].astype(float);v[:,:,2]*=-1
    v[:,:,1]-=v[:,:,1].min();v[:,:,0]-=v[0,0,0];v[:,:,2]-=v[0,0,2]
    across=(v[0,2]-v[0,1])+(v[0,17]-v[0,16]);across[1]=0
    forward=np.cross(np.array([0.,1.,0.]),across);forward/=np.linalg.norm(forward)
    angle=-np.arctan2(forward[0],forward[2]);c=np.cos(angle);s=np.sin(angle)
    x=v[:,:,0].copy();z=v[:,:,2].copy();v[:,:,0]=c*x+s*z;v[:,:,2]=-s*x+c*z
    error=float(np.linalg.norm(decoded-v[:-1],axis=-1).max())
    assert error<1e-4 and abs(error-report['decode_max_error_m'])<1e-8
    out={'candidate_report_sha256':sha(here/'report.json'),'feature_sha256':sha(here/'candidate.npz'),
         'frames_verified':len(frames),'feature_rows':len(f),'normalization_error_m':error,
         'training_ready':False,'limitations':'No independent FK or complete encoding, semantic, true FPS, model or physical validation.'}
    with (here/'audit_report.json').open('x',encoding='utf8') as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
