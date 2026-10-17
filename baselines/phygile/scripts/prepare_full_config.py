"""Prepare path-corrected upstream Phase A/B configs; does not launch training.

Requires author-compatible data/labels. Passing this file check does not
establish that the dataset or its labels match the paper.
"""
import argparse, pathlib, yaml
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['a','b'],required=True)
p.add_argument('--difficulty-csv',type=pathlib.Path,required=True);p.add_argument('--motion-dir',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
assert a.difficulty_csv.is_file(),a.difficulty_csv
assert a.motion_dir.is_dir() and any(a.motion_dir.glob('*.npz')),a.motion_dir
root=pathlib.Path(__file__).resolve().parents[1]
src=root/f'upstream/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_moe_phase_{a.phase}.yaml'
cfg=yaml.safe_load(src.read_text());cfg['runner']['upload_models_to_wandb']=False
motion=cfg['commands']['motion'];motion['preload_device']='same_as_env'
motion['curriculum_csv_path']=str(a.difficulty_csv.resolve());motion['curriculum_mocap_dir']=str(a.motion_dir.resolve())
motion['curriculum_resume_level_override']=0
if a.phase=='b':motion['phase_b_pseudo_level_csv']=str(a.difficulty_csv.resolve())
a.output.parent.mkdir(parents=True,exist_ok=True)
if a.output.exists():raise FileExistsError(a.output)
a.output.write_text(yaml.safe_dump(cfg,sort_keys=False))
print('Prepared upstream configuration; data/label equivalence and full training are NOT validated:',a.output)
