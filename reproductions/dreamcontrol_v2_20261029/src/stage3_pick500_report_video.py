"""Render admitted diffusion references on the actual calibrated Isaac robot.

This is pose playback, not a learned policy rollout or a grasp-success demo.
The object's initial pose stays fixed; no carried-object path is fabricated.
"""
import argparse,json,hashlib,subprocess,time
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();args.output_dir.mkdir(exist_ok=False)
app=AppLauncher(args).app
import torch
from PIL import Image,ImageDraw,ImageFont
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from stage3_pick29_env import Pick29EnvCfg,ASSET
from stage3_pick29_core import BODY_JOINTS,sha,SCENE_SHA

def main():
    torch.cuda.set_per_process_memory_fraction(.08,device=args.device)
    manifest=json.loads(args.manifest.read_text());refs=manifest['references']
    # Pick three different target locations from admitted entries, not raw candidates.
    positions=np.asarray([x['task_object_position'] for x in refs]);chosen=[0]
    chosen.append(int(np.linalg.norm(positions-positions[0],axis=1).argmax()))
    chosen.append(int(np.minimum(*[np.linalg.norm(positions-positions[i],axis=1) for i in chosen]).argmax()))
    assert len(set(chosen))==3
    sources=[];selected=[]
    for i in chosen:
        item=refs[i];assert sha(item['path'])==item['sha256']
        proof=json.loads(Path(item['postopt_report']).read_text())
        assert sha(item['postopt_report'])==item['postopt_report_sha256'] and proof['eligible_for_calibrated_scene_reference']
        with np.load(item['path'],allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        assert s['joint_names'].tolist()==BODY_JOINTS and float(s['fps'])==20 and len(s['q'])==120
        meta=json.loads(s['metadata_utf8'].tobytes())
        selected.append({**item,'candidate_id':meta['candidate_id']});sources.append(s)
    cfg=Pick29EnvCfg();cfg.scene.num_envs=1;cfg.sim.device=args.device
    cfg.sim.render_interval=1;cfg.scene.robot.spawn.rigid_props.disable_gravity=True
    for name in ['overview','side']:
        setattr(cfg.scene,name,CameraCfg(prim_path='/World/'+name,width=640,height=480,data_types=['rgb'],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=28.,clipping_range=(.1,50.))))
    env=ManagerBasedRLEnv(cfg=cfg)
    writer=None
    try:
        env.reset();robot=env.scene['robot'];ids=robot.find_joints(BODY_JOINTS,preserve_order=True)[0]
        origin=env.scene.env_origins[0]
        eyes=[[2.0,-2.0,1.65],[.15,-2.5,1.25]];target=[.18,0,.75]
        for name,eye in zip(['overview','side'],eyes):
            env.scene[name].set_world_poses_from_view(torch.tensor([eye],device=env.device)+origin,
                                                      torch.tensor([target],device=env.device)+origin)
        for _ in range(8):env.sim.render()
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
        small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        video=args.output_dir/'Pick29_generated_references.mp4'
        writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1280x600','-r','20','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20',
            '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        snapshots=[];frame_count=0;began=time.monotonic()
        for k,s in enumerate(sources):
            obj=env.scene['object'];state=obj.data.default_root_state.clone()
            state[:,:3]=torch.tensor(s['task_object_position'],device=env.device)+origin
            state[:,3:7]=torch.tensor(s['task_object_quaternion_wxyz'],device=env.device);state[:,7:]=0
            for f in range(120):
                all_q=robot.data.default_joint_pos.clone();all_q[:,ids]=torch.tensor(s['q'][f],device=env.device)
                action=torch.zeros(1,31,device=env.device);action[:,-2:]=1-2*torch.tensor(s['hand_closed'][f],device=env.device)
                env.action_manager.process_action(action)
                for name in ['left_hand','right_hand']:
                    term=env.action_manager.get_term(name);all_q[:,term._joint_ids]=term.processed_actions
                root=torch.tensor(np.r_[s['root_pos'][f],s['root_quat'][f],np.zeros(6)][None],device=env.device,dtype=torch.float32)
                root[:,:3]+=origin
                robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(all_q,torch.zeros_like(all_q))
                robot.set_joint_position_target(all_q);obj.write_root_state_to_sim(state)
                env.scene.write_data_to_sim();env.sim.forward();env.scene.update(.05);env.sim.render()
                images=[]
                for name in ['overview','side']:
                    camera=env.scene[name];camera.update(.05,force_recompute=True)
                    images.append(camera.data.output['rgb'][0,:,:,:3].cpu().numpy().astype(np.uint8))
                canvas=Image.new('RGB',(1280,600),(18,24,35));canvas.paste(Image.fromarray(np.concatenate(images,axis=1)),(0,60))
                draw=ImageDraw.Draw(canvas)
                draw.text((22,10),'DreamControl-v2 adaptation | Pick | G1 29DoF | 20 Hz',font=font,fill=(240,246,255))
                phase='Approach' if f<40 else 'Close hand and lift' if f<80 else 'Hold'
                draw.text((22,36),f'Reference {k+1}/3 | Candidate {selected[k]["candidate_id"]} | t={f/20:.2f}s | {phase}',font=small,fill=(112,206,255))
                draw.text((22,549),'GENERATED REFERENCE POSE PLAYBACK - not a PPO policy rollout or grasp-success result',font=small,fill=(255,207,113))
                draw.text((22,575),'Calibrated robot asset; target object stays at its initial pose. Left: oblique view; right: side view.',font=small,fill=(220,227,240))
                writer.stdin.write(np.asarray(canvas).tobytes());frame_count+=1
                if f in [0,40,80,119]:snapshots.append(canvas.copy().resize((640,300)))
                if f%40==0:print(json.dumps({'candidate':selected[k]['candidate_id'],'frame':f,'rendered_frames':frame_count}),flush=True)
        writer.stdin.close();assert writer.wait(timeout=60)==0;writer=None
        sheet=Image.new('RGB',(2560,900))
        for i,im in enumerate(snapshots):sheet.paste(im,((i%4)*640,(i//4)*300))
        sheet.save(args.output_dir/'Pick29_contact_sheet.jpg',quality=88)
        clips=[]
        for i,item in enumerate(selected):
            clip=args.output_dir/('Pick29_reference_%d.mp4'%item['candidate_id'])
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),'-ss',str(i*6),'-t','6',
                '-c:v','libx264','-preset','fast','-crf','20','-threads','2','-an','-movflags','+faststart',str(clip)],check=True)
            clips.append({'file':clip.name,'sha256':sha(clip),'bytes':clip.stat().st_size,'candidate_id':item['candidate_id']})
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames:format=duration','-of','json',str(video)],text=True))
        assert probe['streams'][0]['nb_read_frames']=='360' and probe['streams'][0]['avg_frame_rate']=='20/1'
        result={'status':'rendered_verified','scope':'reference pose playback, not learned policy execution or grasp success',
            'manifest':str(args.manifest),'manifest_sha256':sha(args.manifest),'asset_sha256':sha(ASSET),'scene_sha256':SCENE_SHA,
            'body_dof':29,'reference_fps':20,'frames_per_reference':120,'video_frames':frame_count,'selected':selected,
            'video':video.name,'video_sha256':sha(video),'video_bytes':video.stat().st_size,'ffprobe':probe,'clips':clips,
            'elapsed_seconds':time.monotonic()-began}
        (args.output_dir/'report.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    finally:
        if writer is not None:
            writer.stdin.close();writer.terminate();writer.wait(timeout=10)
        env.close()

try:main()
finally:app.close(wait_for_replicator=False)
