"""New target/error visualization of the same three admitted Pick references.

The targets come from the immutable diffusion input file, transformed using its
recorded transform. Measured positions come from the native robot wrist link.
The object is explicitly static: neither grasp nor policy success is claimed.
"""
import argparse,csv,json,subprocess,time
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--original-report',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.output_dir.mkdir(exist_ok=False);app=AppLauncher(a).app
import torch
from PIL import Image,ImageDraw,ImageFont
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from stage3_pick29_env import Pick29EnvCfg,ASSET
from stage3_pick29_core import BODY_JOINTS,sha,SCENE_SHA

FONT='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
FONT_EN='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FLOOR=8*1024**3+456*1024**2
YELLOW=(255,209,66);CYAN=(33,230,248);GREEN=(101,248,135);BLUE=(134,181,255)

def main():
    torch.cuda.set_per_process_memory_fraction(.08,device=a.device)
    old=json.loads(a.original_report.read_text());selected=old['selected'];assert len(selected)==3
    sources=[];goals=[];source_proofs=[]
    for item in selected:
        assert sha(item['path'])==item['sha256'] and sha(item['postopt_report'])==item['postopt_report_sha256']
        proof=json.loads(Path(item['postopt_report']).read_text());assert proof['eligible_for_calibrated_scene_reference']
        kin=json.loads(Path(item['kinematic_report']).read_text());assert sha(kin['sample'])==kin['sample_sha256']
        with np.load(item['path'],allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        with np.load(kin['sample'],allow_pickle=False) as z:
            m=json.loads(z['metadata_utf8'].tobytes())
            goal=z['target_points']@np.asarray(m['canonical_to_world_rotation']).T+np.asarray(m['canonical_to_world_translation'])
        assert m['candidate_id']==item['candidate_id'] and len(s['q'])==120 and float(s['fps'])==20
        assert np.allclose(goal[40,21],s['task_object_position']+[0,0,.04],atol=2e-7)
        assert np.allclose(goal[80,21],s['task_object_position']+[0,0,.20],atol=2e-7)
        sources.append(s);goals.append(goal[:,21]);source_proofs.append({'sample':kin['sample'],'sha256':kin['sample_sha256']})
    cfg=Pick29EnvCfg();cfg.scene.num_envs=1;cfg.sim.device=a.device;cfg.sim.render_interval=1
    cfg.scene.robot.spawn.rigid_props.disable_gravity=True
    for name in ['overview','closeup']:
        setattr(cfg.scene,name,CameraCfg(prim_path='/World/'+name,width=640,height=480,data_types=['rgb'],update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=28.,clipping_range=(.1,50.))))
    env=ManagerBasedRLEnv(cfg=cfg);writer=None
    try:
        env.reset();robot=env.scene['robot'];ids=robot.find_joints(BODY_JOINTS,preserve_order=True)[0]
        wi=robot.find_bodies('right_wrist_yaw_link')[0][0];origin=env.scene.env_origins[0]
        font=ImageFont.truetype(FONT,22);small=ImageFont.truetype(FONT,19);big=ImageFont.truetype(FONT,29)
        en=ImageFont.truetype(FONT_EN,18)
        def project(camera,points):
            pts=torch.as_tensor(np.asarray(points),device=env.device,dtype=torch.float32)+origin
            data=camera.data
            local=math_utils.quat_apply_inverse(data.quat_w_ros[0].expand(len(pts),-1),pts-data.pos_w[0])
            xyz=(data.intrinsic_matrices[0]@local.T).T
            assert bool((local[:,2]>.01).all()),'annotation behind camera'
            return (xyz[:,:2]/xyz[:,2:]).cpu().numpy()
        video=a.output_dir/'Pick29_targets_and_errors.mp4'
        writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1280x760','-r','20','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19',
            '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        all_metrics=[];summaries=[];frames=0;began=time.monotonic()
        for k,(s,goal) in enumerate(zip(sources,goals)):
            obj=env.scene['object'];obj_p=np.asarray(s['task_object_position'])
            state=obj.data.default_root_state.clone();state[:,:3]=torch.as_tensor(obj_p,device=env.device)+origin
            state[:,3:7]=torch.as_tensor(s['task_object_quaternion_wxyz'],device=env.device);state[:,7:]=0
            viewpoints=[([2.,-2.,1.65],[.18,0,.75]),([obj_p[0]+.72,obj_p[1]-.85,1.27],(obj_p+[0,0,.045]).tolist())]
            for name,(eye,look) in zip(['overview','closeup'],viewpoints):
                env.scene[name].set_world_poses_from_view(torch.tensor([eye],device=env.device,dtype=torch.float32)+origin,torch.tensor([look],device=env.device,dtype=torch.float32)+origin)
            for _ in range(6):env.sim.render()
            history=[];err=[];stored_fk=[];object_drifts=[]
            for f in range(120):
                assert __import__('shutil').disk_usage('/data').free>FLOOR
                all_q=robot.data.default_joint_pos.clone();all_q[:,ids]=torch.as_tensor(s['q'][f],device=env.device)
                action=torch.zeros(1,31,device=env.device);action[:,-2:]=1-2*torch.as_tensor(s['hand_closed'][f],device=env.device)
                env.action_manager.process_action(action)
                for name in ['left_hand','right_hand']:
                    term=env.action_manager.get_term(name);all_q[:,term._joint_ids]=term.processed_actions
                root=torch.tensor(np.r_[s['root_pos'][f],s['root_quat'][f],np.zeros(6)][None],device=env.device,dtype=torch.float32);root[:,:3]+=origin
                robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(all_q,torch.zeros_like(all_q));robot.set_joint_position_target(all_q)
                obj.write_root_state_to_sim(state);env.scene.write_data_to_sim();env.sim.forward();env.scene.update(.05);env.sim.render()
                actual=(robot.data.body_pos_w[0,wi]-origin).cpu().numpy();history.append(actual.copy())
                error=float(np.linalg.norm(actual-goal[f]));err.append(error)
                stored_fk.append(float(np.linalg.norm(actual-s['keypoints'][f,21])))
                drift=float(torch.linalg.norm(obj.data.root_pos_w[0]-state[0,:3]));object_drifts.append(drift)
                images=[]
                for name in ['overview','closeup']:
                    camera=env.scene[name];camera.update(.05,force_recompute=True)
                    im=Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy().astype(np.uint8));d=ImageDraw.Draw(im)
                    path=project(camera,goal);trail=project(camera,np.asarray(history))
                    # Pixel overlays are projections of recorded3D coordinates, not hand-drawn goals.
                    d.line([tuple(map(float,v)) for v in path],fill=YELLOW,width=2)
                    if f>0:d.line([tuple(map(float,v)) for v in trail],fill=CYAN,width=3)
                    for point,color,label,anchor in [(path[40],YELLOW,'2s 接近点',(365,360)),(path[80],GREEN,'4s 抬手目标',(365,35))]:
                        x,y=map(float,point);d.ellipse((x-9,y-9,x+9,y+9),outline=color,width=3)
                        if name=='closeup':
                            d.line([(x,y),(anchor[0]-8,anchor[1]+10)],fill=color,width=2)
                            d.rectangle((anchor[0]-3,anchor[1]-2,anchor[0]+220,anchor[1]+29),fill=(12,19,30))
                            d.text(anchor,label,font=font,fill=color)
                    x,y=map(float,trail[-1]);gx,gy=map(float,path[f])
                    d.ellipse((gx-13,gy-13,gx+13,gy+13),outline=YELLOW,width=2)
                    d.line((x-6,y,x+6,y),fill=CYAN,width=3);d.line((x,y-6,x,y+6),fill=CYAN,width=3)
                    corners=obj_p+np.asarray([[xx,yy,zz] for xx in [-.025,.025] for yy in [-.025,.025] for zz in [-.1,.1]])
                    box=project(camera,corners)
                    for i in range(8):
                        for j in range(i+1,8):
                            if (i^j) in [1,2,4]:d.line([tuple(map(float,box[i])),tuple(map(float,box[j]))],fill=BLUE,width=2)
                    if name=='closeup':
                        center=project(camera,obj_p[None])[0];anchor=(20,425)
                        d.line([tuple(center),(anchor[0]+180,anchor[1]+10)],fill=BLUE,width=2)
                        d.rectangle((14,420,260,456),fill=(12,19,30));d.text(anchor,'目标物体（固定）',font=font,fill=BLUE)
                    d.rectangle((8,8,305,37),fill=(12,19,30));d.text((15,10),'全身参考' if name=='overview' else '手部与目标位置放大',font=small,fill='white')
                    images.append(im)
                canvas=Image.new('RGB',(1280,760),(18,24,35));canvas.paste(images[0],(0,95));canvas.paste(images[1],(640,95));d=ImageDraw.Draw(canvas)
                d.text((22,8),'Pick：29自由度 G1 参考动作与目标位置对照',font=big,fill='white')
                phase='接近物体' if f<40 else '闭手并抬手' if f<80 else '目标位置保持'
                d.text((22,53),f'样本 {selected[k]["candidate_id"]}  |  {f/20:.2f} 秒 / 5.95 秒  |  {phase}  |  20 Hz',font=font,fill=(180,209,242))
                d.text((22,588),f'右腕实际位置 vs 当前右腕目标：距离 {1000*error:.2f} mm',font=big,fill=GREEN)
                d.text((22,628),'黄色：要求的右腕路径 / 当前目标圈    青色：机器人右腕实际轨迹    绿色：抬手目标点',font=small,fill=(236,239,247))
                d.text((22,665),f'接近点 z={goal[40,2]:.2f}m；抬手点 z={goal[80,2]:.2f}m；目标抬手高度变化 16cm。',font=font,fill=(226,232,243))
                d.text((22,704),'参考姿态回放；物体固定。这里检查右腕到位，不代表已抓住或搬动物体。',font=font,fill=(255,202,111))
                writer.stdin.write(np.asarray(canvas).tobytes());frames+=1
                all_metrics.append({'candidate_id':selected[k]['candidate_id'],'frame':f,'time_s':f/20,
                    'actual_wrist_world':actual.tolist(),'requested_wrist_world':goal[f].tolist(),'wrist_distance_m':error,
                    'native_vs_saved_fk_error_m':stored_fk[-1],'object_position_drift_m':drift})
                if f in [40,80]:canvas.save(a.output_dir/f'candidate_{selected[k]["candidate_id"]}_t{int(f/20)}s.jpg',quality=92)
            assert max(stored_fk)<1e-4 and max(object_drifts)<1e-6
            summaries.append({'candidate_id':selected[k]['candidate_id'],'distance_at_2s_mm':err[40]*1000,
                'distance_at_4s_mm':err[80]*1000,'distance_at_5_95s_mm':err[119]*1000,'maximum_path_distance_mm':max(err)*1000,
                'contact_wrist_goal_world':goal[40].tolist(),'lift_wrist_goal_world':goal[80].tolist(),
                'native_vs_saved_fk_max_error_m':max(stored_fk),'object_max_drift_m':max(object_drifts)})
        writer.stdin.close();assert writer.wait(timeout=60)==0;writer=None
        (a.output_dir/'metrics.json').write_text(json.dumps(all_metrics,indent=2))
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries',
            'stream=width,height,avg_frame_rate,nb_read_frames:format=duration','-of','json',str(video)],text=True))
        assert frames==360 and probe['streams'][0]['nb_read_frames']=='360' and probe['streams'][0]['avg_frame_rate']=='20/1'
        report={'status':'rendered_verified','scope':'native wrist target position verification of postoptimized reference poses; no policy or object-grasp success',
            'original_report_sha256':sha(a.original_report),'asset_sha256':sha(ASSET),'scene_sha256':SCENE_SHA,'selected':selected,
            'source_proofs':source_proofs,'target_source':'immutable generated sample target_points plus recorded canonical-to-world transform',
            'measurement_source':'native right_wrist_yaw_link origin after injected reference state; not palm or fingertip contact',
            'summaries':summaries,'metrics_sha256':sha(a.output_dir/'metrics.json'),'video':video.name,'video_sha256':sha(video),
            'ffprobe':probe,'elapsed_s':time.monotonic()-began,'object_motion':'fixed at initial per-reference pose'}
        (a.output_dir/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:
        if writer is not None:writer.stdin.close();writer.terminate();writer.wait(timeout=10)
        env.close()
try:main()
except BaseException as e:
    import traceback
    failure={'status':'failed_preserved','error':repr(e),'traceback':traceback.format_exc()}
    (a.output_dir/'failure.json').write_text(json.dumps(failure,indent=2))
    print(failure['traceback'],flush=True)
    raise
finally:app.close(wait_for_replicator=False)
