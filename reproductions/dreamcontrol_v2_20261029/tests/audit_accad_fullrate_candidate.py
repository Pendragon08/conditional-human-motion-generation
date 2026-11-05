"""Recheck provenance, selected weighted overlaps, timeline and robot features."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN=ROOT/'accad_013267_pyroki_fullrate_20261031'
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report=json.loads((HERE/'report.json').read_text());trajectory=HERE/'candidate.npz'
    assert sha(trajectory)==report['output_sha256'] and sha(RUN/'audit_report.json')==report['windows_audit_sha256']
    assert sha(RUN/'report.json')==report['windows_report_sha256']
    rows=[json.loads(line) for line in (RUN/'windows.jsonl').read_text().splitlines()]
    with np.load(trajectory,allow_pickle=False) as f:
        raw=f['raw_frames'].copy();blend=f['blended_120hz'].copy();first=f['first_solution'].copy()
        frames=f['frames_20hz'].copy();filtered=f['filtered_20hz'].copy();features=f['features'].copy()
        points=f['points_z_up_world'].copy();decoded=f['decoded_points'].copy()
    np.testing.assert_array_equal(raw,np.arange(1,355));np.testing.assert_array_equal(frames,np.arange(6,355,6))
    assert features.shape==(58,263) and filtered.shape==(59,34) and np.isfinite(features).all()
    assert np.isin(features[:,-4:],[0.,1.]).all()
    weights=[1,2,3,4,3,2,1];max_difference=0.
    for frame in (6,180,354):
        values=[]
        for row in rows:
            if frame not in row['sample_indices']:continue
            path=RUN/row['artifact'];assert sha(path)==row['artifact_sha256']
            with np.load(path,allow_pickle=False) as f:
                j=int(np.flatnonzero(f['source_frames']==frame)[0])
                value=np.r_[f['root_wxyz_xyz'][j],f['joint_pos'][j]].astype(float)
                values.append((weights[j],value))
        assert values
        reference=values[0][1][:4];total=np.zeros(34);denom=0.
        for weight,value in values:
            if np.dot(value[:4],reference)<0:value[:4]*=-1
            total+=weight*value;denom+=weight
        expected=total/denom;expected[:4]/=np.linalg.norm(expected[:4])
        actual=blend[frame-1];direction=1. if np.dot(expected[:4],actual[:4])>=0 else -1.
        expected[:4]*=direction
        max_difference=max(max_difference,float(np.max(np.abs(expected-actual))))
        assert np.max(np.abs(values[0][1][4:]-first[frame-1,4:]))<1e-6
    assert max_difference<1e-6
    p=points[:,:,[0,2,1]].astype(float);p[:,:,2]*=-1;p[:,:,1]-=p[:,:,1].min()
    p[:,:,0]-=p[0,0,0];p[:,:,2]-=p[0,0,2]
    across=(p[0,2]-p[0,1])+(p[0,17]-p[0,16]);across[1]=0
    direction=np.cross([0.,1.,0.],across);direction/=np.linalg.norm(direction)
    theta=-np.arctan2(direction[0],direction[2]);x=p[:,:,0].copy();z=p[:,:,2].copy()
    p[:,:,0]=np.cos(theta)*x+np.sin(theta)*z;p[:,:,2]=-np.sin(theta)*x+np.cos(theta)*z
    error=float(np.linalg.norm(decoded-p[:-1],axis=-1).max())
    assert error<1e-4 and abs(error-report['decode_max_error_m'])<1e-8
    result={'candidate_report_sha256':sha(HERE/'report.json'),'artifact_sha256':sha(trajectory),
            'selected_blend_frames':[6,180,354],'max_blend_component_error':max_difference,
            'normalization_error_m':error,'training_ready':False,
            'limitations':['Checks selected blends, not full overlap replay or independently filtered rotations.',
                           'No independent FK/full encoding, actual FPS, caption semantics or physical quality verification.']}
    with (HERE/'audit_report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
