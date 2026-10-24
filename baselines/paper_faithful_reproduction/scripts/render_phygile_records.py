"""Render all four recorded diagnostic episodes, without advancing physics.

Historical files store measured body positions and joint angles, not root
quaternions. A rigid Kabsch fit recovers root orientation from those body points;
the maximum FK reconstruction residual is checked and reported per clip.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--records-root',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    contract_path=a.records_root/'private_assets_continue500_v1/contract.json'
    contract=json.loads(contract_path.read_text(encoding='utf-8'))
    spec=mujoco.MjSpec.from_file(str(a.model))
    spec.worldbody.add_geom(name='video_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[10,10,.1],rgba=[.24,.28,.32,1])
    spec.worldbody.add_light(name='video_light',pos=[0,-3,5],dir=[0,.3,-1],diffuse=[.9,.9,.9])
    model=spec.compile();model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    data=mujoco.MjData(model)
    body_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in contract['tracked_body_names']]
    joint_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['joint_names']]
    if min(body_ids+joint_ids)<0:raise ValueError('Recorded model names missing from visual asset')
    addresses=model.jnt_qposadr[joint_ids]
    camera=mujoco.MjvCamera();camera.azimuth=135;camera.elevation=-15;camera.distance=3.1
    renderer=mujoco.Renderer(model,height=480,width=640)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    receipt={'source_sha256':sha(__file__),'model_sha256':sha(a.model),'contract_sha256':sha(contract_path),
             'mode':'visual replay of measured states; no new physics steps',
             'root_rotation':'rigid Kabsch fit from recorded body positions and measured joint angles',
             'residual_tolerance_m':.002,'checkpoint_updates':2500,'paper_benchmark':False,'clips':[]}
    out=a.output/'PhyGile_actual_tracking_all4.mp4'
    writer=imageio.get_writer(str(out),format='FFMPEG',fps=50,codec='libx264',quality=8,macro_block_size=1)
    try:
        for clip in range(4):
            entries=[]
            for engine,relative in [('Isaac Lab / PhysX',f'isaac_continued_private_v1/isaac_clip{clip}'),
                                    ('Unitree Mjlab / MuJoCo',f'continue500_mjlab_comparison_v1/official_clip{clip}')]:
                path=a.records_root/relative/'raw_pre_reset.npz'
                raw=np.load(path,allow_pickle=False);poses=[];errors=[]
                for q,target in zip(raw['robot_joint_pos'],raw['robot_body_pos']):
                    data.qpos[:]=model.qpos0;data.qpos[:3]=0;data.qpos[3:7]=[1,0,0,0];data.qpos[addresses]=q
                    mujoco.mj_forward(model,data)
                    source=data.xpos[body_ids].copy()
                    ac=source-source.mean(0);bc=target-target.mean(0)
                    u,_,vt=np.linalg.svd(ac.T@bc)
                    parity=np.eye(3);parity[2,2]=np.linalg.det(u@vt)
                    row_rotation=u@parity@vt
                    translation=target.mean(0)-source.mean(0)@row_rotation
                    quat=np.empty(4);mujoco.mju_mat2Quat(quat,row_rotation.T.flatten())
                    pose=data.qpos.copy();pose[:3]=translation;pose[3:7]=quat
                    data.qpos[:]=pose;mujoco.mj_forward(model,data)
                    errors.append(float(np.max(np.linalg.norm(data.xpos[body_ids]-target,axis=-1))))
                    poses.append(pose)
                if max(errors)>.002:raise ValueError(f'Reconstruction error {engine} clip{clip}: {max(errors)}')
                row={'engine':engine,'clip':clip,'raw_sha256':sha(path),'recorded_frames':len(poses),
                     'full_clip_frames':149,'full_clip_completed':len(poses)==149 and not bool(raw['terminated'].any()),
                     'fk_reconstruction_max_m':max(errors),'terminated':bool(raw['terminated'].any())}
                receipt['clips'].append(row);entries.append((engine,np.asarray(poses),row))
            for frame in range(149):
                canvas=Image.new('RGB',(1280,600),(15,22,33));draw=ImageDraw.Draw(canvas)
                draw.text((20,10),f'PhyGile diagnostic checkpoint: 2,500 PPO iterations | clip {clip+1}/4',font=font,fill='white')
                for side,(engine,poses,row) in enumerate(entries):
                    actual_index=min(frame,len(poses)-1);data.qpos[:]=poses[actual_index];mujoco.mj_forward(model,data)
                    camera.lookat[:]=[data.qpos[0],data.qpos[1],.8]
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render()),(side*640,80))
                    draw.text((side*640+14,45),engine,font=font,fill=(100,210,255))
                    ended=frame>=len(poses)
                    label=(f'FAILED at {len(poses)/50:.2f}s | final recorded pose held' if ended else f'Actual frame {actual_index+1}/{len(poses)} | t={(actual_index+1)/50:.2f}s')
                    draw.text((side*640+14,565),label,font=small,fill=(255,125,125) if ended else 'white')
                if clip==0 and frame==0:canvas.save(a.output/'PhyGile_preview.png')
                writer.append_data(np.asarray(canvas))
    finally:
        writer.close();renderer.close()
    receipt.update(video=str(out),video_sha256=sha(out),output_frames=4*149,fps=50)
    (a.output/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
