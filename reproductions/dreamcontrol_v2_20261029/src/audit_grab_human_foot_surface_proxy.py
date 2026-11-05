"""Sample human skinning-weight foot regions; not measured ground or contact."""
import contextlib,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import smplx,torch,trimesh
from grab_restricted_reader import read_grab_field

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
STAGE=ROOT/'grab_mesh_stage_20261030'
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
REGISTRY=ROOT/'grab_source_registry_20261030/source_registry.jsonl'
INPUTS=(('s1/teapot_pour_1.npz','grab_s1_pour_human_source_20261030/human_source.npz'),
        ('s2/airplane_fly_1.npz','grab_human_robot_pilot_20261030/pilot_arrays.npz'),
        ('s3/apple_eat_1.npz','grab_s3_human_source_20261030/human_source.npz'),
        ('s4/waterbottle_drink_1.npz','grab_s4_drink_human_source_20261030/human_source.npz'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(1);registry_sha='27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4'
    if sha(REGISTRY)!=registry_sha:raise ValueError('registry changed')
    records={x['member']:x for x in map(json.loads,REGISTRY.read_text().splitlines()) if x['member'] in dict(INPUTS)}
    if len(records)!=len(INPUTS):raise ValueError('source registry coverage')
    results=[]
    for member,source in INPUTS:
        row=records[member];archive=ROOT/'data/GRAB'/row['archive'];source=ROOT/source
        if sha(archive)!=row['archive_sha256']:raise ValueError('archive changed')
        with np.load(source,allow_pickle=False) as data:
            allframes=data['input_frame_indices'].copy();expected_joints=data['human_joints'].copy();fps=float(data['input_fps'])
        if len(allframes)!=120 or expected_joints.shape!=(120,22,3) or fps!=row['fps_declared'] or not np.all(np.diff(allframes)==1):raise ValueError('unexpected input timeline')
        select=np.arange(0,120,6);frames=allframes[select]
        with zipfile.ZipFile(archive) as z:
            body=read_grab_field(z,member,'body')
            with np.load(io.BytesIO(z.read(member)),allow_pickle=False) as meta:
                gender=str(meta['gender'].item());pca=int(meta['n_comps'].item())
        if body['vtemp']!=row['template'] or gender!=row['gender']:raise ValueError('template source mismatch')
        template=(STAGE/row['template']).resolve(strict=True)
        if not template.is_relative_to(STAGE.resolve()) or sha(template)!=row['template_sha256']:raise ValueError('template changed')
        mesh=trimesh.load(str(template),process=False);keys=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
        params={k:torch.as_tensor(np.asarray(body['params'][k])[frames],dtype=torch.float32) for k in keys}
        with contextlib.redirect_stdout(sys.stderr):
            model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=pca,batch_size=len(frames),use_pca=True,ext='npz')
            if not np.array_equal(np.asarray(mesh.faces),np.asarray(model.faces)):raise ValueError('template topology mismatch')
            model.v_template.copy_(torch.as_tensor(np.asarray(mesh.vertices),dtype=model.v_template.dtype))
            weights=model.lbs_weights.detach().cpu().numpy()
            if weights.ndim!=2 or weights.shape[0]!=10475 or weights.shape[1]<12:raise ValueError('skinning weights shape')
            masks=[(weights[:,7]+weights[:,10])>=.5,(weights[:,8]+weights[:,11])>=.5]
            if any(int(m.sum())<50 for m in masks) or np.any(masks[0]&masks[1]):raise ValueError('foot proxy mask invalid')
            with torch.no_grad():
                out=model(**params,return_verts=True);vertices=out.vertices.cpu().numpy();joints=out.joints[:,:22].cpu().numpy()
        joint_diff=float(np.max(np.abs(joints-expected_joints[select])))
        if joint_diff>2e-5:raise ValueError('joint reconstruction differs')
        per=[]
        for i,frame in enumerate(frames):
            mins=[float(vertices[i,mask,2].min()) for mask in masks]
            per.append({'frame':int(frame),'time_s_if_declared_fps_correct':float(frame/fps),'left_skinning_region_min_z_m':mins[0],'right_skinning_region_min_z_m':mins[1]})
        left=np.array([r['left_skinning_region_min_z_m'] for r in per]);right=np.array([r['right_skinning_region_min_z_m'] for r in per])
        results.append({'member':member,'archive_sha256':row['archive_sha256'],'member_sha256':row['member_sha256'],
                        'source_npz_sha256':sha(source),'template_sha256':row['template_sha256'],
                        'model_sha256':sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')),
                        'skinning_region_vertex_counts':[int(m.sum()) for m in masks],
                        'max_joint_difference_m':joint_diff,'frames':per,
                        'left_min_z_range_m':[float(left.min()),float(left.max())],
                        'right_min_z_range_m':[float(right.min()),float(right.max())]})
        print(json.dumps({'member':member,'left_min_z_range_m':results[-1]['left_min_z_range_m'],'right_min_z_range_m':results[-1]['right_min_z_range_m']}),flush=True)
    output={'script_sha256':sha(__file__),'registry_sha256':registry_sha,'reader_sha256':sha(OUT/'grab_restricted_reader.py'),'subjects':results,
            'limitations':'20 predetermined samples per one-second sequence, approximate SMPL-X LBS weight masks (ankle+foot >=0.5), not independently verified sole segmentation. Min z is a vertex proxy, not measured ground/contact, source FPS truth, robot dynamics or training acceptance. Four selected actions and existing locked-head models, not full GRAB.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({'complete':True,'sequences':len(results)}),flush=True)
if __name__=='__main__':main()
