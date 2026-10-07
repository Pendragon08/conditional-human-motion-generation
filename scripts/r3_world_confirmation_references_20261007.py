"""Select fresh reference IDs using static pre-locked rules, never dynamics."""
import argparse
import ast
import hashlib
import json
import pickle
import re
import shutil
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def canonical_root(data):
    pos=np.asarray(data['global_position'],dtype=np.float64)
    w,x,y,z=np.asarray(data['global_pose'].rotation().wxyz,dtype=np.float64)[20]
    yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    c,s=np.cos(yaw),np.sin(yaw)
    rot=np.asarray([[c,s,0],[-s,c,0],[0,0,1]])
    return (pos-pos[20])@rot.T

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    assert not a.output.exists(), 'Preserve selection, never replace confirmation set'
    a.output.mkdir(parents=True)
    src=root/'third_party/DreamControl/TrajGen/sample'
    for sub in ['Kick_sim/retarget.py','Kick_sim1/refine_motions.py']:
        assert sha(a.stage/sub)==sha(src/sub),sub
    names=ast.literal_eval(re.search(r'joint_names = (\[.*?\])',(a.stage/'Kick_sim/retarget.py').read_text(),re.S).group(1))
    urdf=root/'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    limits={j.attrib['name']:(float(j.find('limit').attrib['lower']),float(j.find('limit').attrib['upper']))
            for j in ET.parse(urdf).getroot().findall('joint') if j.attrib['name'] in names}
    lo=np.asarray([limits[n][0] for n in names]);hi=np.asarray([limits[n][1] for n in names])
    olddir=src/'Kick_sim2';oldfiles=sorted(olddir.glob('*.pkl'));assert len(oldfiles)==66
    old=[pickle.load(f.open('rb')) for f in oldfiles]
    oldq=np.stack([np.asarray(d['joints']) for d in old]);oldroot=np.stack([canonical_root(d) for d in old])
    generated=np.load(a.stage/'Kick_sim/results.npy',allow_pickle=True).item()
    assert generated['motion'].shape==(32,22,3,196),generated['motion'].shape
    assert np.isfinite(generated['motion']).all()
    rows=[];accepted=[];accepted_data=[]
    for ident in range(32):
        f=a.stage/'Kick_sim2'/('%d_n.pkl'%ident)
        row={'original_id':ident,'file':f.name,'accepted_static':False,'reasons':[]}
        if not f.exists(): row['reasons'].append('official_retarget_or_refine_rejected')
        else:
            d=pickle.load(f.open('rb'));q=np.asarray(d['joints']);pos=np.asarray(d['global_position']);quat=np.asarray(d['global_pose'].rotation().wxyz)
            row['sha256']=sha(f);row['frames']=len(q);row['grab_idx']=int(d['grab_idx'])
            if q.shape!=(215,27) or pos.shape!=(215,3) or quat.shape!=(215,4):row['reasons'].append('shape')
            elif not all(np.isfinite(v).all() for v in [q,pos,quat]):row['reasons'].append('nonfinite')
            else:
                row['joint_limit_max_violation_rad']=float(np.maximum(lo-q,q-hi).max().clip(min=0))
                row['quat_norm_max_error']=float(np.abs(np.linalg.norm(quat,axis=1)-1).max())
                if row['joint_limit_max_violation_rad']>1e-4:row['reasons'].append('joint_limit')
                if row['quat_norm_max_error']>1e-3:row['reasons'].append('quat_norm')
                if not 0<=row['grab_idx']<215:row['reasons'].append('event_out_of_bounds')
                canon=canonical_root(d)
                qr=np.sqrt(np.mean((oldq-q)**2,axis=(1,2)));rr=np.sqrt(np.mean((oldroot-canon)**2,axis=(1,2)))
                row['nearest_original_q_rmse_rad']=float(qr.min())
                row['original_near_duplicate_count']=int(((qr<.01)&(rr<.001)).sum())
                if row['original_near_duplicate_count']:row['reasons'].append('original_near_duplicate')
                row['prior_candidate_near_duplicate_count']=sum(int(np.sqrt(np.mean((pq-q)**2))<.01 and
                    np.sqrt(np.mean((pr-canon)**2))<.001) for pq,pr in accepted_data)
                if row['prior_candidate_near_duplicate_count']:row['reasons'].append('prior_candidate_near_duplicate')
            row['accepted_static']=not row['reasons']
            if row['accepted_static']:
                accepted.append((ident,f));accepted_data.append((q,canon))
        rows.append(row)
    selected=accepted[:11];directory=a.output/'selected_refs';directory.mkdir()
    files=[]
    for ident,f in selected:
        dst=directory/('%04d_n.pkl'%ident);shutil.copyfile(f,dst)
        files.append({'original_id':ident,'file':dst.name,'bytes':dst.stat().st_size,'sha256':sha(dst)})
    selected_ids={i for i,f in selected}
    for row in rows:
        row['selection_outcome']='selected' if row['original_id'] in selected_ids else 'eligible_not_selected' if row['accepted_static'] else 'rejected_static'
    result={'status':'COMPLETE' if len(selected)==11 else 'BLOCKED','stage':'static_selection',
            'seed':20261007,'generated_count':32,'accepted_static_count':len(accepted),
            'selected_count':len(selected),'rejected_count':32-len(accepted),'rows':rows,
            'selected':files,'reference_dir':str(directory),'selection_rule':'first11_numeric_ID_static_only',
            'novelty_alignment':'same215frames, no time warping; root subtractframe20position and inverseframe20yaw; joints unchanged',
            'physical_results_used_for_selection':False,'extra_IK':False,'formal_PPO_started':False,
            'generator_weight_sha256':'886b722e95496f4ceafd704c4e1d0578888733d12ce67429677b80a91a9c36d0',
            'human_results_sha256':sha(a.stage/'Kick_sim/results.npy'),
            'source_sha256':{sub:sha(a.stage/sub) for sub in ['Kick_sim/retarget.py','Kick_sim1/refine_motions.py']},
            'URDF_sha256':sha(urdf),'created_unix':time.time()}
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    assert len(selected)==11,'Insufficient fresh static references: stop, no resampling'
    print('WORLD_FRESH_CONFIRMATION_REFERENCES_COMPLETE',len(selected),len(accepted),32,flush=True)

if __name__=='__main__':main()
