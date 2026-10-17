"""Four PPO iterations exercise the actual scheduler/evaluator call path.

Two arbitrary fixture IDs are explicitly not research difficulty annotations.
Their purpose is to prove a low-success evaluation holds the current level.
No resulting checkpoint is promoted to a research baseline.
"""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
motions=out/'fixture_motions';motions.mkdir()
selected=sorted((ROOT/'data/train').glob('*.npz'))[:2]
if len(selected)!=2:raise ValueError('Need two real motion references for integration fixture')
for source in selected:(motions/source.name).symlink_to(source.resolve())
csv_path=out/'fixture_ids_not_research_labels.csv'
with csv_path.open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['file','rating'])
 for i,source in enumerate(selected):writer.writerow([source.name.replace('.npz','.txt'),i+1])
cfg=yaml.safe_load((ROOT/'configs/moe_pilot.yaml').read_text())
cfg['runner']['experiment_name']='phygile_curriculum_interface_validation'
cfg['runner']['save_interval']=1
cfg['commands']['motion'].update(curriculum_enabled=True,curriculum_csv_path=str(csv_path),
    curriculum_mocap_dir='',curriculum_max_difficulty=2,curriculum_iteration_per_level=2,
    curriculum_eval_runs=2,curriculum_success_threshold=.8,
    curriculum_auto_upgrade_per_level=[0,0],curriculum_auto_upgrade_after_iters=0,
    curriculum_metric_convergence_min_iters=3000,moe_supervised_until_level=2)
config=out/'fixture_config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
command=['bash','scripts/run_isaac.sh','scripts/train_with_curriculum_evaluator.py',
 '--task','Tracking-Flat-G1-Wo-State-Estimation-v0','--headless','--device','cuda:0',
 '--num_envs','32','--seed','42','--max_iterations','4','--run_name',out.parent.name,
 '--logger','tensorboard','--g1_training_cfg',str(config),'--motion_dir',str(motions),
 '--no_wandb_model_upload']
before=set((ROOT/'logs/rsl_rl/phygile_curriculum_interface_validation').glob('*'))
with (out/'training.log').open('w') as log:
 result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
if result.returncode:raise RuntimeError(f'Curriculum training validation failed; code {result.returncode}')
after=set((ROOT/'logs/rsl_rl/phygile_curriculum_interface_validation').glob('*'))
new=after-before
if len(new)!=1:raise AssertionError('Expected exactly one new training log directory')
run=new.pop();decisions=sorted(run.glob('independent_curriculum/*/decision.json'))
if len(decisions)!=1:raise AssertionError(('Expected one actual independent evaluation',decisions))
decision=json.loads(decisions[0].read_text())
assert decision['iteration']==3 and decision['old_level']==1 and decision['new_level']==1
assert not decision['promote']
evaluation=decisions[0].parent/'assessment/evaluation.json'
report=json.loads(evaluation.read_text())
assert report['valid'] and report['attempts_per_file']==2
log=(out/'training.log').read_text()
assert 'PHYGILE_CURRICULUM_HOOK_ACTIVE' in log and 'PHYGILE_INDEPENDENT_CURRICULUM_DECISION' in log
records={'status':'COMPLETED','purpose':'scheduler integration fixture only; not difficulty curriculum reproduction',
 'fixture_labels_are_research_annotations':False,'ppo_iterations':4,'adam_optimizer_steps_expected':80,
 'num_envs':32,'training_run':str(run),'decision_path':str(decisions[0]),
 'decision':decision,'evaluation_path':str(evaluation),'evaluation':report,
 'motion_sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in selected]}
(out/'verification.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
print('PHYGILE_CURRICULUM_TRAINING_VALIDATED',json.dumps({k:v for k,v in records.items() if k!='evaluation'}))
