"""Blend audited KIT robot windows and export unaccepted 263D previews."""
import argparse,hashlib,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import blend_robot_windows as blend_module
import feature263 as feature_module
import g1_fk as fk_module
from blend_robot_windows import blend_windows
from feature263 import canonicalize_z_up,encode
from g1_fk import URDFFK

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1)
    base=ROOT/'kit_dart_hypothesis_work_20261102'; human_base=ROOT/'kit_dart_hypothesis_human_20261102'; audit_path=base/'robot_windows_audit_20261102.json';audit_hash=sha(audit_path);audit=json.loads(audit_path.read_text());index={x['id']:x for x in audit['cases']}
    if set(index)!={'001199','006160','007754','006923','001171','007259','001893','009236'} or sha(URDF)!=audit['urdf_sha256']:raise ValueError('audit/URDF mismatch')
    limits={}
    for j in ET.parse(URDF).getroot().findall('joint'):
        lim=j.find('limit')
        if j.get('type')=='revolute' and lim is not None:limits[j.get('name')]=(float(lim.get('lower')),float(lim.get('upper')))
    module_paths=[Path(x.__file__).resolve() for x in (blend_module,feature_module,fk_module)]
    module_hashes={str(p):sha(p) for p in module_paths}
    if shutil.disk_usage(a.out.parent).free < 8*1024**3+64*1024**2:raise RuntimeError('insufficient disk budget')
    a.out.mkdir(exist_ok=False);cases=[]
    for sid in ('001199','007754','006923','001171','007259','001893','009236'):
        if index[sid]['joint_values_outside_declared_limits'] != 0:raise ValueError('quarantined limit violation')
        d=base/'robot_windows'/sid; human=human_base/('human_'+sid); report=json.loads((d/'report.json').read_text());source=human/'human_20hz_dart_hypothesis.npz'
        if sha(d/'report.json')!=index[sid]['solver_report_sha256'] or sha(source)!=index[sid]['human_sha256'] or sha(d/'windows.jsonl')!=report['rows_sha256']:raise ValueError('audited input changed')
        with np.load(source,allow_pickle=False) as f:source_frames=f['source_frames'].copy();source_times=f['source_times'].copy()
        rows=[json.loads(line) for line in (d/'windows.jsonl').read_text().splitlines()];windows=[]
        for row in rows:
            p=d/row['artifact']
            if sha(p)!=row['sha256']:raise ValueError('window changed')
            with np.load(p,allow_pickle=False) as f:windows.append({'frames':f['sample_indices'].copy(),'root_wxyz_xyz':f['root_wxyz_xyz'].copy(),'joint_pos':f['joint_pos'].copy(),'joint_names':f['joint_names'].copy()})
        merged=blend_windows(windows);idx=merged['frames'];np.testing.assert_array_equal(idx,np.arange(1,len(source_frames)));state=merged['blended'];names=list(merged['joint_names'])
        if set(names)!=set(limits):raise ValueError('actuator limits do not match')
        lo=np.array([limits[x][0] for x in names]);hi=np.array([limits[x][1] for x in names]);max_excess=float(np.maximum(np.maximum(lo-state[:,7:],state[:,7:]-hi),0).max())
        if max_excess>0:raise ValueError('blended state exceeds joint limits')
        points=URDFFK(URDF).forward(state[:,4:7],state[:,[1,2,3,0]],state[:,7:],names);features=encode(points,foot_velocity_squared_threshold=.002)
        with torch.no_grad():decoded=recover_from_ric(torch.from_numpy(features),22).numpy()
        error=float(np.linalg.norm(decoded-canonicalize_z_up(points)[:-1],axis=-1).max())
        if features.shape!=(len(state)-1,263) or not np.isfinite(features).all() or error>1e-4:raise ValueError('encoding self-check failed')
        target=a.out/(sid+'.npz')
        with target.open('xb') as f:np.savez_compressed(f,features=features,root_wxyz_xyz=state[:,:7],joint_pos=state[:,7:],sample_times=source_times[idx],source_frames=source_frames[idx],sample_indices=idx,joint_names=np.asarray(names),points_z_up_world=points,decoded_points=decoded)
        result={'id':sid,'output_sha256':sha(target),'window_audit_sha256':audit_hash,'human_sha256':sha(source),'solver_report_sha256':sha(d/'report.json'),'script_sha256':sha(Path(__file__)),'module_sha256':module_hashes,'states':len(state),'feature_shape':list(features.shape),'sample_times_first_last':[float(source_times[idx[0]]),float(source_times[idx[-1]])],'max_joint_limit_excess_rad':max_excess,'decode_selfcheck_max_error_m':error,'caption_accepted':False,'training_ready':False,'limitations':['Weighted overlap blending and z=0 solver are approximate; no physical guarantee.','100 Hz assumed from DART, source header says 120 Hz; stride without anti-alias filter.','Time and source identity unresolved; caption unaccepted.','Decoded points reuse the same feature encoder and kinematics.']}
        with (a.out/(sid+'_report.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
        cases.append(result)
    if sha(audit_path)!=audit_hash or any(sha(p)!=module_hashes[str(p)] for p in module_paths):raise ValueError('audit or module changed during export')
    with (a.out/'report.json').open('x',encoding='utf-8') as f:json.dump({'cases':cases,'training_ready':False},f,indent=2,allow_nan=False)
    print(json.dumps({'cases':cases,'training_ready':False}),flush=True)
if __name__=='__main__':main()
