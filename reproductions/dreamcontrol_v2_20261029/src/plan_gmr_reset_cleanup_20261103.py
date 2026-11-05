"""Inventory unequivocally derived AMASS/KIT/GRAB NPZ files, never raw inputs.

Only explicitly named, project-owned output locations are considered. Unknown
locations, archives, models, OmniRetarget and checkpoints are not touched.
This program only writes a manifest; it does not delete files.
"""
import hashlib,json,os,stat,zipfile
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029').resolve(strict=True)
DIRS={
 'human':['bmlmovi_full_human_20261102','bmlmovi_stream_full_20261102','bmlmovi_stream_resume_20261102','bmlmovi_stream_smoke_20261102','eyesjapan_stream_smoke_20261103','eyesjapan_stream_batch_20261103','eyesjapan_stream_resume_20261103','eyesjapan_stream_batch2_20261103','grab_full_human_source_batch_20261102']+[f'grab_full_human_source_batch_v{i}_{"20261102" if i in (2,3,4) else "20261103"}' for i in range(2,11)],
 'windows':['bmlmovi_full_robot_work_20261102/robot_windows','grab_full_robot_work_20261102/robot_windows','grab_full_robot_work_v2_20261102/robot_windows','grab_robot_batch_v2_work_20261103/robot_windows'],
 'features':['bmlmovi_full_robot_263_preview_20261102','grab_full_robot_263_preview_20261102','grab_full_robot_263_preview_v2_20261102','kit_dart_hypothesis_263_preview_20261102'],
 'kit_human':['kit_dart_hypothesis_human_20261102'],
 'gmr_pilot':['grab_human_robot_pilot_20261030']}
REQUIRED={'human':{'human_joints','source_frames','declared_fps'},'windows':{'sample_indices','source_frames','source_times','root_wxyz_xyz','joint_pos','joint_names'},'features':{'features','root_wxyz_xyz','joint_pos','sample_times'},'kit_human':{'human','source_frames','source_times'},'gmr_pilot':{'g1_qpos_120_wxyz_xyz_29','g1_27_joint_pos','human_joints'}}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 rows=[];missing=[];paths=set()
 for category,dirs in DIRS.items():
  for name in dirs:
   d=ROOT/name
   if not d.exists():missing.append(name);continue
   if d.is_symlink() or not d.is_dir() or not d.resolve().is_relative_to(ROOT):raise ValueError('unsafe directory: '+name)
   for p in sorted(d.rglob('*.npz')):
    if p.is_symlink() or any(x.is_symlink() for x in p.parents if x!=ROOT and x.is_relative_to(ROOT)):raise ValueError('symlink encountered')
    s=p.lstat();relative=p.relative_to(ROOT).as_posix()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or relative in paths:raise ValueError('unsafe file')
    with zipfile.ZipFile(p) as z:keys={x[:-4] for x in z.namelist() if x.endswith('.npy')}
    if not REQUIRED[category].issubset(keys):raise ValueError('unexpected NPZ signature: '+relative)
    rows.append({'path':relative,'category':category,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)});paths.add(relative)
 before=ROOT/'data/GRAB'
 raw=[before/f'grab__s{i}.zip' for i in range(1,11)]
 if not all(p.is_file() and not p.is_symlink() for p in raw):raise ValueError('one or more raw GRAB archives missing')
 report={'files':rows,'counts':{k:sum(r['category']==k for r in rows) for k in DIRS},'bytes':sum(r['bytes'] for r in rows),'missing_dirs':missing,'raw_grab_archives':[{'path':str(p),'size':p.stat().st_size} for p in raw],'training_ready':False,'scope':'Derived NPZ in explicitly named project output locations only; other categories require separate classification.'}
 target=ROOT/'gmr_reset_cleanup_plan_20261103.json'
 with target.open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps({'files':len(rows),'bytes':report['bytes'],'counts':report['counts'],'missing_dirs':missing}))
if __name__=='__main__':main()
