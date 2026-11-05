"""Produce provisional direct 29DoF Omni/OMOMO text-motion artifacts in batches.

No robot re-retargeting or physical-action filtering.  Text association and
nominal subject grouping are unverified, especially for crops and variants.
"""
import argparse,hashlib,io,json,os,shutil,sys,time,zipfile
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
T=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
sys.path[:0]=[str(R),str(R/'ssm_new6_20hz_resume_20261101'),str(R/'antialias_v3_20261030/src'),str(T)]
from g1_fk import URDFFK
from feature263_signfix import encode_signfixed
from resample_antialias_omniretarget import resample_antialias
names=tuple(n+'_joint' for n in ('left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,required=True);parser.add_argument('--count',type=int,required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
 if a.start<0 or a.count<1 or not a.output.replace('_','').isalnum():raise ValueError('invalid selection')
 out=R/a.output
 if out.exists():raise FileExistsError(out)
 group=R/'omomo_group_split_20261101';manifest=group/'candidates.jsonl';grp=json.loads((group/'report.json').read_text())
 archive=R/'data/OmniRetarget_Dataset/robot-object.zip';textzip=R/'omomo_text_pairing_20261101/omomo_text_anno.zip'
 pins={str(manifest):grp['candidate_manifest_sha256'],str(archive):'93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3',str(textzip):'2d11fb37fef294b7f6816c74740694d12f28d62e0f638e348c678dc51ee1ad7c',str(U):'572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95'}
 if any(sha(Path(p))!=v for p,v in pins.items()):raise ValueError('input hash mismatch')
 rows=[json.loads(s) for s in manifest.read_text().splitlines()]
 if len(rows)!=1826 or a.start+a.count>len(rows):raise ValueError('selection outside pinned candidate set')
 fk=URDFFK(U)
 if len(fk.actuated)!=29 or set(fk.actuated)!=set(names):raise ValueError('joint layout mismatch')
 out.mkdir(exist_ok=False);failures=0;started=time.monotonic()
 with zipfile.ZipFile(archive) as z,zipfile.ZipFile(textzip) as tz,(out/'records.jsonl').open('x',buffering=1,encoding='utf8') as log:
  for idx in range(a.start,a.start+a.count):
   if shutil.disk_usage(R).free<8*1024**3+256*1024**2:raise RuntimeError('disk reserve guard')
   item=rows[idx];record={'index':idx,'source_member':item['motion_member'],'training_ready':False}
   try:
    raw=z.read(item['motion_member']);annraw=tz.read(item['annotation_member'])
    rh,ah=hashlib.sha256(raw).hexdigest(),hashlib.sha256(annraw).hexdigest()
    if rh!=item['motion_member_sha256'] or ah!=item['annotation_sha256']:raise ValueError('member hash mismatch')
    caption=json.loads(annraw)[item['nominal_action_key']]
    if caption!=item['caption'] or not isinstance(caption,str) or not caption.strip():raise ValueError('caption member mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:q=np.asarray(f['qpos']);fps=float(f['fps'])
    if q.ndim!=2 or q.shape[1]!=43 or fps!=30:raise ValueError('unexpected source format')
    sampled=resample_antialias(q,fps,20.,max_seconds=10.)
    points=fk.forward(sampled[:,4:7],sampled[:,[1,2,3,0]],sampled[:,7:36],names)
    feat=encode_signfixed(points,foot_velocity_squared_threshold=.002)
    if feat.shape!=(len(sampled)-1,263):raise ValueError('encoding shape')
    target=out/('sample_%04d.npz'%idx)
    with target.open('xb') as f:np.savez_compressed(f,qpos29_source_layout_20hz=sampled,points_z_up=points,features=feat,sample_times=np.arange(len(sampled))/20.,joint_names=np.asarray(names))
    record.update({'status':'technical_preview','source_sha256':rh,'annotation_member':item['annotation_member'],'annotation_sha256':ah,'caption':caption,'caption_accepted':False,'caption_origin':'source_text_nominal_filename_match','variant':item['variant'],'nominal_subject':item['nominal_subject'],'nominal_action_key':item['nominal_action_key'],'auxiliary_split':item['auxiliary_split'],'fps_declared':fps,'output_fps':20.,'source_frames':len(q),'state_frames':len(sampled),'feature_frames':len(feat),'cropped_for_max_10s':bool((len(q)-1)/fps >= 10.),'artifact':target.name,'artifact_sha256':sha(target)})
   except (ValueError,KeyError,IndexError,TypeError) as e:
    failures+=1;record.update({'status':'technical_error','error':repr(e),'caption_accepted':False})
   log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
   if (idx-a.start+1)%10==0:print('processed',idx-a.start+1,'of',a.count,'errors',failures,flush=True)
 if any(sha(Path(p))!=v for p,v in pins.items()):raise RuntimeError('input changed')
 result={'start':a.start,'count':a.count,'failures':failures,'elapsed_s':time.monotonic()-started,'input_sha256':pins,'records_sha256':sha(out/'records.jsonl'),'script_sha256':sha(Path(__file__)),'training_ready':False,'limitations':['No physical-action screening and no retargeting of Omni trajectories.','Name-associated captions are not semantically accepted, especially for crops or variants.','FIR quaternion component filtering and feature encoding are approximations.','Nominal subject splits, license scope and real clocks remain unverified.']}
 with (out/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
