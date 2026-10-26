"""CPU/Mesa rendering of recorded poses, never another control simulation.

The test mode makes conspicuously labeled synthetic renderer fixtures. Actual
mode is initially disabled and accepts only closed, provenance-bound5k pilots.
"""
import argparse, hashlib, importlib.metadata, json, os, pathlib, re, shutil, time, traceback
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def write(path,value):
 temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def render_signature(spec):
 keys=['required_versions','environment','width','height','fps','test_frames','visual_model_xml_sha256','mesh_files_sha256','mesa_vendor_sha256']
 return hashlib.sha256(json.dumps({k:spec[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def actual_trace(pilot_run,trial_number,render_spec):
 assert render_spec['actual_rendering_enabled'] is True, 'Actual-pose rendering is disabled; no media output'
 from quick_pilot_protocol import load_teacher, authorize_teacher, settings_signature, QUALITY, STATUS
 spec=json.loads((ROOT/'configs/quick-pilot-spec.json').read_text())
 teacher=load_teacher(ROOT,spec);cp=authorize_teacher(spec,teacher)
 assert re.fullmatch('[a-z0-9_]+',pilot_run)
 out=ROOT/'tracker_validation'/pilot_run;rpath=out/'receipt.json'
 r=json.loads(rpath.read_text());e=json.loads((ROOT/'receipts'/('quick_pilot_'+pilot_run+'_exit.json')).read_text())
 assert r['status']==STATUS and r['mode']=='quick_pilot' and r['quality_label']==QUALITY and r['stage'] in ['smoke','sample']
 assert r['output_writers_closed'] is True and r['optimizer_updates']==0 and r['physical_tracking_executed'] is True
 assert r['policy_checkpoint_sha256']==cp['sha256'] and r['policy_checkpoint_completed_iterations']==5000
 assert r['reference_fingerprint']==spec['expected_reference_fingerprint']
 assert r['source_sha256']==spec['executor_sha256'] and r['execution_settings_sha256']==settings_signature(spec)
 assert e['evaluation_receipt_sha256']==sha(rpath) and e['numerical_status']==STATUS
 assert e['child_reaped'] is True and e['child_exit_code']==0 and e['supervision_passed'] is True and e['timed_out'] is False
 assert e['supervisor_error'] is None and not e.get('cleanup_error') and e['sample_errors']==[]
 for file,key in [(ROOT/'receipts'/('quick_pilot_'+pilot_run+'_process_exit.json'),'process_exit_sha256'),
                  (ROOT/'logs'/('quick_pilot_'+pilot_run+'_resources.jsonl'),'resources_sha256')]:assert sha(file)==e[key]
 index=out/'executed_traces.private.json';assert sha(index)==r['private_executed_trace_index_sha256']
 entries=json.loads(index.read_text());assert 0<=trial_number<len(entries)
 entry=entries[trial_number];folder=pathlib.Path(entry['folder'])
 assert folder.resolve()==folder and folder.parent==out/'execution_private'
 meta=json.loads((folder/'trace.private.json').read_text());path=folder/'executed_poses.private.npz'
 assert meta['runtime_kind']=='isaaclab_physics' and meta['status']=='FINISHED_PHYSICAL_TRIAL'
 assert sha(folder/'poses.private.bin')==meta['binary_sha256'] and sha(path)==meta['npz_sha256']==entry['npz_sha256']
 assert meta['phase']=='INITIAL_THEN_PRE_AUTORESET_TERMINATION_COMPUTE'
 assert meta['samples']==entry['samples']==meta['trial']['control_steps']+1
 witnesses=render_spec['software_render_witnesses'];assert len(witnesses)==2 and len({w['run'] for w in witnesses})==2
 for witness in witnesses:
  rp=ROOT/'video_validation'/witness['run']/'receipt.json';ep=ROOT/'receipts'/('execution_video_'+witness['run']+'_exit.json')
  assert sha(rp)==witness['receipt_sha256'] and sha(ep)==witness['exit_sha256']
  wr=json.loads(rp.read_text());we=json.loads(ep.read_text())
  assert wr['status']=='SOFTWARE_RENDERER_SYNTHETIC_WITNESS_PASSED' and wr['mode']=='renderer_test'
  assert wr['source_sha256']==render_spec['renderer_sha256'] and wr['render_signature']==render_signature(render_spec)
  assert wr['nvidia_libraries_loaded']==[] and 'llvmpipe' in wr['gl_renderer'].lower()
  assert wr['frames_encoded']==wr['frames_decoded']==render_spec['test_frames'] and wr['new_physics_steps']==0
  assert we['child_exit_code']==0 and we['supervision_passed'] is True and we['timed_out'] is False and we['evaluation_receipt_sha256']==sha(rp)
 return path,meta,dict(pilot_run=pilot_run,trial_in_finish_order=trial_number,policy_checkpoint_sha256=cp['sha256'],
                       pilot_receipt_sha256=sha(rpath),execution_trace_sha256=sha(path))

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--mode',choices=['renderer_test','actual'],required=True)
 p.add_argument('--run',required=True);p.add_argument('--pilot-run');p.add_argument('--trial',type=int,default=0);a=p.parse_args()
 assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
 spec=json.loads(pathlib.Path(a.spec).read_text());path=meta=None;provenance={}
 if a.mode=='actual':path,meta,provenance=actual_trace(a.pilot_run,a.trial,spec)
 for key,value in spec['environment'].items():assert os.environ.get(key)==value, key
 assert set(os.sched_getaffinity(0))=={0,1,2,3}
 assert all(importlib.metadata.version(name)==version for name,version in spec['required_versions'].items())
 assert sha(__file__)==spec['renderer_sha256'] and sha(spec['mesa_vendor_file'])==spec['mesa_vendor_sha256']
 assert sha(spec['visual_model_xml'])==spec['visual_model_xml_sha256']
 assert all(sha(file)==value for file,value in spec['mesh_files_sha256'].items())
 assert shutil.disk_usage(ROOT).free>spec['minimum_disk_free_bytes']+spec['maximum_video_bytes']
 out=ROOT/('video_validation' if a.mode=='renderer_test' else 'recorded_videos_private')/a.run
 out.parent.mkdir(exist_ok=True);out.mkdir(exist_ok=False)
 r=dict(mode=a.mode,status='STARTING',source_sha256=sha(__file__),render_signature=render_signature(spec),
        quality_label='SYNTHETIC_RENDERER_VALIDATION_NOT_AN_EXPERIMENT_RESULT' if a.mode=='renderer_test' else spec['quality_label'],
        new_physics_steps=0,new_optimizer_updates=0,sdk_runs=0,actual_policy_result_video=a.mode=='actual',
        role=spec['renderer_role'],provenance=provenance,output_writers_closed=False)
 renderer=writer=None;start=time.monotonic()
 try:
  import numpy as np, mujoco, imageio.v2 as imageio, imageio_ffmpeg
  from PIL import Image,ImageDraw,ImageFont
  import xml.etree.ElementTree as E
  xml=pathlib.Path(spec['visual_model_xml']);tree=E.parse(xml);tree.find('compiler').set('meshdir',str(xml.parent/'meshes'))
  # Visual ground/lights only; mj_step is prohibited below. Shared model untouched.
  worlds=tree.findall('worldbody');assert worlds;world=worlds[0]
  # MuJoCo merges repeated worldbody sections; ElementTree.find sees only one.
  planes=[g for w in worlds for g in w.findall('geom') if g.get('type')=='plane']
  assert len(planes)<=1,'Ambiguous visual ground; do not overlay coplanar floors'
  if planes:
   floor=planes[0];floor.set('pos','0 0 0');ground_mode='reused_existing_world_plane'
  else:
   floor=E.SubElement(world,'geom',name='video_floor',type='plane',size='2000 2000 .1',rgba='.72 .77 .8 1');ground_mode='added_single_world_plane'
  assert len([g for w in worlds for g in w.findall('geom') if g.get('type')=='plane'])==1
  E.SubElement(world,'light',name='video_light',pos='2 -2 4',dir='-.5 .5 -1',diffuse='.9 .9 .9',castshadow='true')
  model=mujoco.MjModel.from_xml_string(E.tostring(tree.getroot(),encoding='unicode'));data=mujoco.MjData(model)
  compiled_planes=[i for i in range(model.ngeom) if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE and model.geom_bodyid[i]==0]
  assert len(compiled_planes)==1,'Compiled visual model must contain exactly one world plane'
  r.update(visual_ground_count=len(compiled_planes),visual_ground_mode=ground_mode,visual_ground_top_z=0,
           visual_ground_count_checked_in_compiled_model=True,visual_world_sections=len(worlds))
  assert model.nq==36 and model.nv==35
  joints=[i for i in range(model.njnt) if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE]
  names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in joints]
  free=[i for i in range(model.njnt) if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE];assert len(free)==1 and len(joints)==29
  if a.mode=='renderer_test':
   count=spec['test_frames'];source_names=list(reversed(names));poses=np.zeros((count,29),dtype=np.float64)
   poses[:,source_names.index('left_shoulder_pitch_joint')]=.15*np.sin(np.arange(count)*.2)
   roots=np.zeros((count,3));roots[:,0]=np.arange(count)*.004;roots[:,2]=.793
   quats=np.zeros((count,4));quats[:,0]=1;times=np.arange(count)/50.
   trial_label='SYNTHETIC POSES - NOT AN EXPERIMENT RESULT'
  else:
   with np.load(path,allow_pickle=False) as z:
    assert str(z['runtime_kind'])=='isaaclab_physics'
    assert str(z['capture_phase'])=='INITIAL_THEN_PRE_AUTORESET_TERMINATION_COMPUTE'
    source_names=z['joint_names'].tolist();poses=z['joint_pos'].astype(np.float64);roots=z['root_pos'].astype(np.float64)
    quats=z['root_quat_wxyz'].astype(np.float64);times=z['time_seconds'].copy()
   count=len(poses);assert count==meta['samples'];trial_label='FAILURE' if meta['trial']['failed'] else 'NOMINAL DURATION COMPLETED; ENDPOINT UNCHECKED'
  assert 2<=count<=spec['maximum_frames'] and len(set(source_names))==29 and set(source_names)==set(names)
  assert poses.shape==(count,29) and roots.shape==(count,3) and quats.shape==(count,4)
  assert all(np.isfinite(x).all() for x in [poses,roots,quats,times]) and np.allclose(times,np.arange(count)/50.,atol=1e-8)
  assert np.allclose(np.linalg.norm(quats,axis=1),1.,atol=1e-3)
  order=[source_names.index(name) for name in names];addresses=[int(model.jnt_qposadr[j]) for j in joints];base=int(model.jnt_qposadr[free[0]])
  model.vis.global_.offwidth=spec['width'];model.vis.global_.offheight=spec['height']
  renderer=mujoco.Renderer(model,height=spec['height'],width=spec['width'])
  def forbidden(*args,**kwargs):raise AssertionError('No control/dynamics step is allowed in pose video rendering')
  mujoco.mj_step=forbidden
  camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=3.2;camera.azimuth=135;camera.elevation=-15
  font=ImageFont.load_default(size=15);target=out/'video.mp4';pixel_hashes=[]
  for i in range(count):
   data.qpos[base:base+3]=roots[i];data.qpos[base+3:base+7]=quats[i];data.qpos[addresses]=poses[i,order]
   assert np.array_equal(data.qpos[addresses],poses[i,order]);mujoco.mj_forward(model,data)
   camera.lookat[:]=roots[i]+np.array([0,0,-.1]);renderer.update_scene(data,camera=camera);pixels=renderer.render()
   if i==0:
    from OpenGL import GL
    text=lambda key:GL.glGetString(key).decode()
    r.update(gl_vendor=text(GL.GL_VENDOR),gl_renderer=text(GL.GL_RENDERER),gl_version=text(GL.GL_VERSION))
    r['nvidia_libraries_loaded']=[line for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'libnvidia' in line.lower()]
    assert 'mesa' in r['gl_vendor'].lower() and 'llvmpipe' in r['gl_renderer'].lower() and not r['nvidia_libraries_loaded']
    writer=imageio.get_writer(str(target),fps=spec['fps'],codec='libx264',quality=7,macro_block_size=16,ffmpeg_log_level='error')
   pixel_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
   image=Image.fromarray(pixels);draw=ImageDraw.Draw(image);draw.rectangle((0,0,spec['width'],78),fill=(12,17,26))
   draw.text((10,5),'RENDERER TEST' if a.mode=='renderer_test' else 'RECORDED ISAACLAB | G1 TRACKER5000',font=font,fill=(255,225,90))
   draw.text((10,27),trial_label,font=font,fill=(255,255,255))
   draw.text((10,49),f'Pose replay only; no MuJoCo simulation | {i+1}/{count} | t={times[i]:.2f}s',font=font,fill=(210,225,240))
   frame=np.asarray(image);writer.append_data(frame)
   if i==0:image.save(out/'first_frame.png')
  writer.close();writer=None;renderer.close();renderer=None
  reader=imageio_ffmpeg.read_frames(str(target),pix_fmt='rgb24');video_meta=next(reader);decoded=sum(1 for _ in reader)
  assert decoded==count and video_meta['fps']==50 and tuple(video_meta['size'])==(spec['width'],spec['height'])
  assert len(set(pixel_hashes))>=2 and target.stat().st_size<=spec['maximum_video_bytes']
  r.update(status='SOFTWARE_RENDERER_SYNTHETIC_WITNESS_PASSED' if a.mode=='renderer_test' else 'RECORDED_ISAACLAB_POSE_VIDEO_CREATED',
           frames_encoded=count,frames_decoded=decoded,fps=50,video_bytes=target.stat().st_size,video_sha256=sha(target),
           first_frame_sha256=sha(out/'first_frame.png'),distinct_unlabelled_frame_hashes=len(set(pixel_hashes)),
           joint_name_mapping_verified=True,output_writers_closed=True,elapsed_seconds=time.monotonic()-start,
           final_reference_endpoint_comparison_performed=False,control_or_generation_benchmark=False)
  write(out/'receipt.json',r);print(json.dumps(r))
 except BaseException as error:
  r.update(status='RENDER_FAILED',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
  write(out/'receipt.json',r);raise
 finally:
  if writer is not None:writer.close()
  if renderer is not None:renderer.close()
if __name__=='__main__':main()
