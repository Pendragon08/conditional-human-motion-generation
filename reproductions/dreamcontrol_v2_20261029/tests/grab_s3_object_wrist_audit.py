"""Object-frame wrist distance diagnostic for S3 anchored/reference trajectories.

Wrist center is not a gripper surface. This checks alignment proxies only, not
interaction preservation, collisions, contact forces, physics or training.
"""
import hashlib,json,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import sys
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_human_source_20261030'))
sys.path.insert(1,str(ROOT/'foot_contact_fit_20261030'))
from grab_restricted_reader import read_grab_field
from g1_fk import URDFFK

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p,expected):
    if sha(p)!=expected:raise ValueError('hash mismatch: '+str(p))
    with np.load(p,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def chosen(windows):
    result={};priority=np.array([1,2,3,4,3,2,1])
    for w in windows:
        for i,f in enumerate(w['frames']):
            f=int(f)
            if f not in result or priority[i]>result[f][0]:
                result[f]=(int(priority[i]),w['root_wxyz_xyz'][i],w['joint_pos'][i])
    frames=np.array(sorted(result));return frames,np.stack([result[f][1] for f in frames]),np.stack([result[f][2] for f in frames])

def main():
    base=ROOT/'pyroki_grab_s3_windowed_20261030';early=ROOT/'pyroki_s3_overlap_anchor_20261030';tail=ROOT/'pyroki_s3_anchor_tail_20261030'
    brep=json.loads((base/'report.json').read_text());arep=json.loads((early/'report.json').read_text());trep=json.loads((tail/'report.json').read_text())
    if sha(base/'windows.jsonl')!=brep['rows_sha256'] or sha(early/'windows.jsonl')!=arep['rows_sha256'] or sha(tail/'windows.jsonl')!=trep['rows_sha256']:
        raise ValueError('window list changed')
    br=[json.loads(x) for x in (base/'windows.jsonl').read_text().splitlines()]
    ar={x['window']:x for x in map(json.loads,(early/'windows.jsonl').read_text().splitlines())}
    tr={x['window']:x for x in map(json.loads,(tail/'windows.jsonl').read_text().splitlines())}
    if len(br)!=29 or len(ar)!=5 or len(tr)!=23:raise ValueError('missing windows')
    source=ROOT/'grab_s3_human_source_20261030/human_source.npz'
    if sha(source)!=brep['source_sha256'] or sha(source)!=arep['source_sha256'] or sha(source)!=trep['source_sha256']:
        raise ValueError('source mismatch')
    original=[]; anchored=[]
    for i,row in enumerate(br):
        b=load(base/row['artifact'],row['artifact_sha256']); original.append(b)
        a=b if i==0 else load((early if i<=5 else tail)/f'window_{i:02d}.npz',(ar if i<=5 else tr)[i]['artifact_sha256'])
        if not np.array_equal(a['frames'],b['frames']) or not np.array_equal(a['joint_names'],b['joint_names']):raise ValueError('different trajectory frames')
        anchored.append(a)
    frame0,r0,q0=chosen(original);frame1,r1,q1=chosen(anchored)
    if not np.array_equal(frame0,frame1) or not np.array_equal(frame0,np.arange(264,382)):raise ValueError('frame mismatch')
    with np.load(source,allow_pickle=False) as d:source_frames=d['input_frame_indices'].copy();human=d['human_joints'].copy();fps=float(d['input_fps'])
    index=np.searchsorted(source_frames,frame0)
    if fps!=120. or not np.array_equal(source_frames[index],frame0):raise ValueError('human timeline mismatch')
    archive=ROOT/'data/GRAB/grab__s3.zip';source_receipt=json.loads((source.parent/'report.json').read_text())
    if sha(archive)!=source_receipt['archive_sha256']:raise ValueError('archive changed')
    with zipfile.ZipFile(archive) as z:
        obj=read_grab_field(z,source_receipt['source_member'],'object');contact=read_grab_field(z,source_receipt['source_member'],'contact')
    ref=PurePosixPath(obj['object_mesh'])
    if ref.is_absolute() or '..' in ref.parts or ref.parts[:2]!=('tools','object_meshes'):raise ValueError('unsafe asset path')
    mesh=ROOT/'grab_mesh_stage_20261030'/str(ref);vertices=np.asarray(trimesh.load(str(mesh),process=False).vertices)
    labels=np.asarray(contact['object']);trans=np.asarray(obj['params']['transl']);orient=np.asarray(obj['params']['global_orient'])
    if labels.shape[0]!=len(trans) or labels.shape[1]!=len(vertices) or orient.shape!=trans.shape or trans.shape[1]!=3:
        raise ValueError('object/contact shape mismatch')
    tree=cKDTree(vertices);rot=Rotation.from_rotvec(orient[frame0]);translation=trans[frame0]
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    if sha(urdf)!=brep['urdf_sha256']:raise ValueError('URDF mismatch')
    names=original[0]['joint_names'].tolist();fk=URDFFK(urdf)
    wrists=[human[index][:,[20,21],:],fk.forward(r0[:,4:],r0[:,[1,2,3,0]],q0,names,links=('left_wrist_yaw_link','right_wrist_yaw_link')),
            fk.forward(r1[:,4:],r1[:,[1,2,3,0]],q1,names,links=('left_wrist_yaw_link','right_wrist_yaw_link'))]
    names_out=['human','baseline_robot','anchored_robot'];per_frame=[]
    for i,f in enumerate(frame0):
        value={'frame':int(f),'object_labeled_vertices':int((labels[f]!=0).sum())}
        for name,points in zip(names_out,wrists):
            local=rot[i].inv().apply(points[i]-translation[i]);value[name+'_wrist_to_object_vertex_m']=tree.query(local,workers=1)[0].tolist()
        per_frame.append(value)
    subset=[x for x in per_frame if x['object_labeled_vertices']>0]
    summary={}
    for name in names_out:
        d=np.asarray([x[name+'_wrist_to_object_vertex_m'] for x in per_frame]);sel=np.asarray([x[name+'_wrist_to_object_vertex_m'] for x in subset])
        summary[name]={'all_frames_median_left_right_m':np.median(d,axis=0).tolist(),
                       'contact_labeled_frames_median_left_right_m':np.median(sel,axis=0).tolist() if len(sel) else None}
    out_path=Path(__file__).with_name('frames.jsonl')
    with out_path.open('x',encoding='utf8') as f:
        for x in per_frame:f.write(json.dumps(x,allow_nan=False)+'\n')
    report={'source_sha256':sha(source),'archive_sha256':sha(archive),'object_mesh':str(ref),'object_mesh_sha256':sha(mesh),
            'base_rows_sha256':sha(base/'windows.jsonl'),'early_rows_sha256':sha(early/'windows.jsonl'),'tail_rows_sha256':sha(tail/'windows.jsonl'),
            'script_sha256':sha(__file__),'frames_sha256':sha(out_path),'fps':fps,'frames':len(frame0),
            'frames_with_object_contact_label':len(subset),'summary':summary,
            'warning':'Wrist center to nearest object VERTEX, not hand mesh, contact surface or force; contact labels indicate some body part, not necessarily each wrist. Both robot variants use short independent windows selected by center, not validated trajectory stitching. Human/robot proportions differ, old SMPL-X model/version and object transforms need independent verification. No floor, collision response, dynamics, author-v2 objective or training acceptance.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
