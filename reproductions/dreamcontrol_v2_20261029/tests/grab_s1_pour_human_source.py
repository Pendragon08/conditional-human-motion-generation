"""Reconstruct one predetermined S1 pour span from verified GRAB inputs.

Locked-head SMPL-X is an existing model variant; version equivalence to GRAB
is not established. No robot training data is generated.
"""
import contextlib,hashlib,io,json,sys,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
import torch,smplx,trimesh
from grab_restricted_reader import read_grab_field

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
SOURCE='s1/teapot_pour_1.npz'
ARCHIVE_HASH='656c25eb05c6f03e80555c481581147defca6cdba4b5e7cd3ca4645cdac3f07d'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(1);archive=ROOT/'data/GRAB/grab__s1.zip'
    if sha(archive)!=ARCHIVE_HASH:raise ValueError('archive hash mismatch')
    queue=ROOT/'grab_remaining_contact_queue_20261030';done=json.loads((queue/'COMPLETE.json').read_text());entry=next(x for x in done['subjects'] if x['subject']=='s1')
    receipt=queue/'s1/report.json';rows=queue/'s1/rows.jsonl'
    if sha(receipt)!=entry['report_sha256']:raise ValueError('source audit report changed')
    receipt_doc=json.loads(receipt.read_text())
    if sha(rows)!=receipt_doc['records_sha256'] or receipt_doc['archive_sha256']!=ARCHIVE_HASH:raise ValueError('source audit rows changed')
    matched=[x for x in map(json.loads,rows.read_text().splitlines()) if x['member']==SOURCE]
    if len(matched)!=1 or matched[0]['status']!='ok':raise ValueError('source sequence not structurally checked')
    matched=matched[0]
    with zipfile.ZipFile(archive) as z:
        body=read_grab_field(z,SOURCE,'body');obj=read_grab_field(z,SOURCE,'object');contact=read_grab_field(z,SOURCE,'contact')
        with np.load(io.BytesIO(z.read(SOURCE)),allow_pickle=False) as d:
            fps=float(d['framerate'].item());n=int(d['n_frames'].item());gender=str(d['gender'].item());pca=int(d['n_comps'].item());subject=str(d['sbj_id'].item())
    if subject!='s1' or gender!='male' or fps!=120 or pca!=24 or n<120 or n!=matched['frames']:raise ValueError('source metadata mismatch')
    root=ROOT/'grab_mesh_stage_20261030';ref=PurePosixPath(body['vtemp']);objref=PurePosixPath(obj['object_mesh'])
    for path,kind in ((ref,'subject_meshes'),(objref,'object_meshes')):
        if path.is_absolute() or '..' in path.parts or path.parts[:2]!=('tools',kind):raise ValueError('unsafe asset path')
    template=(root/str(ref)).resolve(strict=True);meshpath=(root/str(objref)).resolve(strict=True)
    if not template.is_relative_to(root.resolve()) or not meshpath.is_relative_to(root.resolve()) or sha(template)!=matched['template_sha256'] or sha(meshpath)!=matched['mesh_sha256']:raise ValueError('asset hash mismatch')
    mesh=trimesh.load(str(template),process=False);vertices=np.asarray(mesh.vertices)
    if vertices.shape!=(10475,3) or not np.isfinite(vertices).all():raise ValueError('invalid body template')
    start=max(0,n//2-60);frames=np.arange(start,start+120,dtype=np.int64)
    fields=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
    tensors={k:torch.as_tensor(np.asarray(body['params'][k])[frames],dtype=torch.float32) for k in fields}
    with contextlib.redirect_stdout(sys.stderr):
        model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=pca,batch_size=120,use_pca=True,ext='npz')
        if not np.array_equal(np.asarray(mesh.faces),np.asarray(model.faces)):raise ValueError('template and model topology mismatch')
        model.v_template.copy_(torch.as_tensor(vertices,dtype=model.v_template.dtype))
        with torch.no_grad():human=model(**tensors,return_verts=False).joints[:,:22].detach().cpu().numpy()
    if human.shape!=(120,22,3) or not np.isfinite(human).all():raise ValueError('invalid human joints')
    labels=np.asarray(contact['object'])[frames]
    if labels.shape!=(120,matched['object_vertices']):raise ValueError('object labels shape mismatch')
    right=np.any((labels==22)|((labels>=41)&(labels<=55)),axis=1);left=np.any((labels==21)|((labels>=26)&(labels<=40)),axis=1)
    trans=np.asarray(obj['params']['transl'])[frames];orient=np.asarray(obj['params']['global_orient'])[frames]
    if trans.shape!=(120,3) or orient.shape!=trans.shape or not (np.isfinite(trans).all() and np.isfinite(orient).all()):raise ValueError('invalid object timeline')
    out=Path(__file__).with_name('human_source.npz')
    with out.open('xb') as f:np.savez_compressed(f,human_joints=human,input_frame_indices=frames,input_fps=np.array(fps),object_transl=trans,object_orient_rotvec=orient,right_hand_labels=right,left_hand_labels=left)
    doc={'archive_sha256':ARCHIVE_HASH,'source_member':SOURCE,'source_frames':n,'gender':gender,'fps':fps,'selected_first_frame':int(frames[0]),'selected_last_frame':int(frames[-1]),
         'right_hand_labeled_frames':int(right.sum()),'left_hand_labeled_frames':int(left.sum()),'object_mesh':str(objref),
         'source_audit_report_sha256':sha(receipt),'source_audit_rows_sha256':sha(rows),'template_sha256':sha(template),'object_mesh_sha256':sha(meshpath),
         'model_sha256':sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')),'script_sha256':sha(Path(__file__)),'artifact_sha256':sha(out),
         'warning':'One predetermined middle-second GRAB S1 teapot-pour span, source FPS declared not independently measured. Existing locked-head SMPL-X version may differ from the source model. Object-label presence is derived annotation, not measured contact force. No retargeting, physical validation or training data.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(doc,f,indent=2,allow_nan=False)
    print(json.dumps(doc),flush=True)
if __name__=='__main__':main()
