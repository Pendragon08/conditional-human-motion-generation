"""Re-read isolated KIT robot windows and recompute overlap and limit metrics.

Checks numerical consistency only; not an acceptance of physical feasibility,
true clock, source identity, text semantics, or model equivalence.
"""
import argparse,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    parent=ROOT/'kit_dart_hypothesis_work_20261102'; human_parent=ROOT/'kit_dart_hypothesis_human_20261102';tree=ET.parse(URDF).getroot();limits={}
    for j in tree.findall('joint'):
        v=j.find('limit')
        if j.get('type') in ('revolute','prismatic') and v is not None:limits[j.get('name')]=(float(v.get('lower')),float(v.get('upper')))
    cases=[]
    for sample in ('001199','006160','007754','006923','001171','007259','001893','009236'):
        human=human_parent/('human_'+sample);rdir=parent/'robot_windows'/sample;report=json.loads((rdir/'report.json').read_text());human_audit=json.loads((human/'audit.json').read_text())
        source=human/'human_20hz_dart_hypothesis.npz'
        if report['hypothesis_fps']!=100. or report['source_declared_fps']!=120. or sha(human/'report.json')!=report['human_report_sha256'] or sha(human/'audit.json')!=report['human_audit_sha256'] or sha(source)!=report['human_sha256'] or sha(source)!=human_audit['artifact_sha256'] or sha(URDF)!=report['urdf_sha256'] or sha(rdir/'windows.jsonl')!=report['rows_sha256']:raise ValueError('source changed')
        with np.load(source,allow_pickle=False) as z:source_frames=z['source_frames'].copy();times=z['source_times'].copy();points=z['human'].copy()
        rows=[json.loads(x) for x in (rdir/'windows.jsonl').read_text().splitlines()]
        if not np.all(np.diff(source_frames)==5) or not np.allclose(times,source_frames/100.,atol=1e-12,rtol=0):raise ValueError('hypothetical clock mismatch')
        if len(rows)!=report['windows']:raise ValueError('window count mismatch')
        starts=list(range(0,len(points)-7,4))
        if starts[-1]!=len(points)-8:starts.append(len(points)-8)
        if len(starts)!=len(rows):raise ValueError('expected window count differs')
        prev=None;covered=set();max_joint=0.;max_root=0.;max_rot=0.;max_limit=0.;limit_values=0;names_expected=None;quat_error=0.
        for wi,(row,start) in enumerate(zip(rows,starts)):
            target=rdir/row['artifact'];idx_expected=np.arange(start+1,start+8)
            if row['window']!=wi or sha(target)!=row['sha256']:raise ValueError('window hash/index mismatch')
            with np.load(target,allow_pickle=False) as z:
                idx=z['sample_indices'].copy();sf=z['source_frames'].copy();st=z['source_times'].copy();q=z['joint_pos'].copy();rt=z['root_wxyz_xyz'].copy();names=z['joint_names'].tolist()
            if q.shape!=(7,27) or rt.shape!=(7,7) or len(names)!=27 or len(set(names))!=27 or not np.isfinite(q).all() or not np.isfinite(rt).all():raise ValueError('invalid shape or values')
            if not np.array_equal(idx,idx_expected) or not np.array_equal(sf,source_frames[idx]) or not np.array_equal(st,times[idx]):raise ValueError('source time mismatch')
            if names_expected is None:names_expected=names
            if names!=names_expected or any(n not in limits for n in names):raise ValueError('joint names mismatch')
            low=np.array([limits[n][0] for n in names]);high=np.array([limits[n][1] for n in names]);excess=np.maximum(np.maximum(low-q,q-high),0.)
            max_limit=max(max_limit,float(excess.max()));limit_values+=int(np.count_nonzero(excess>0));quat_error=max(quat_error,float(np.max(np.abs(np.linalg.norm(rt[:,:4],axis=1)-1))))
            if prev is not None:
                common=np.intersect1d(prev['idx'],idx);old=np.searchsorted(prev['idx'],common);new=np.searchsorted(idx,common)
                max_joint=max(max_joint,float(np.max(np.abs(prev['q'][old]-q[new]))));max_root=max(max_root,float(np.max(np.linalg.norm(prev['rt'][old,4:]-rt[new,4:],axis=1))))
                x=Rotation.from_quat(prev['rt'][old][:,[1,2,3,0]]);y=Rotation.from_quat(rt[new][:,[1,2,3,0]]);max_rot=max(max_rot,float(np.max((x.inv()*y).magnitude())))
            covered.update(idx.tolist());prev={'idx':idx,'q':q,'rt':rt}
        if covered!=set(range(1,len(points))):raise ValueError('incomplete frame coverage')
        cases.append({'id':sample,'windows':len(rows),'covered_samples':len(covered),'human_sha256':sha(source),'solver_report_sha256':sha(rdir/'report.json'),'max_joint_limit_excess_rad':max_limit,'joint_values_outside_declared_limits':limit_values,'max_root_quaternion_norm_error':quat_error,'overlap_max_joint_difference_rad':max_joint,'overlap_max_root_difference_m':max_root,'overlap_max_rotation_difference_rad':max_rot})
    out={'cases':cases,'urdf_sha256':sha(URDF),'audit_script_sha256':sha(Path(__file__)),'training_ready':False,'limitations':['Same output windows, URDF and solver assumptions; no solver rerun.','Unstitched windows, 100 Hz is a hypothesis, no text semantic, physical-contact, clock, identity or licensing acceptance.']}
    with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
