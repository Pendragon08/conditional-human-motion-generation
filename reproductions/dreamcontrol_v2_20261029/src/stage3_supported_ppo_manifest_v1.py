"""Select distinct scene-admitted references for a separate task PPO bank."""
import argparse,json,sys
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103'
sys.path[:0]=[str(B/'src'),str(S/'src')]
from stage3_supported_ppo_core_v1 import TaskReferenceBank,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True)
    p.add_argument('--count',type=int,default=500);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    assert a.count in [2,500] and a.output_dir.resolve().is_relative_to(B.resolve())
    rows=[json.loads(s) for s in (B/'bulk_v1/library_records.jsonl').read_text().splitlines(keepends=True) if s.endswith('\n')]
    refs=sorted([r for r in rows if r['task']==a.task],key=lambda r:r['candidate_id'])[:a.count]
    assert len(refs)==a.count
    kin=json.loads(Path(refs[0]['kinematic_report']).read_text())
    scene=json.loads((Path(kin['sample']).parent/'report.json').read_text())['scene']
    a.output_dir.mkdir(exist_ok=False);manifest=a.output_dir/'reference_bank.json'
    asset=S/'assets/g1_29dof_with_hands_urdf_aligned.usd'
    m={'task':a.task,'reference_count':a.count,'reference_fps':20,'body_dof':29,
       'scene':scene,'scene_sha256':refs[0]['scene_sha256'],'asset_sha256':sha(asset),
       'references':refs,'scope':'independent PPO task reference bank; no task-success claim',
       'integration_only':a.count<500,'object_pose_policy':'per_reference_fixed'}
    manifest.open('x').write(json.dumps(m,indent=2))
    bank=TaskReferenceBank(manifest,a.task,sha(asset),'cpu',minimum=a.count)
    report={'status':'task_reference_bank_cpu_verified','task':a.task,'reference_count':bank.count,
            'manifest_sha256':sha(manifest),'source_admitted_record_count':sum(r['task']==a.task for r in rows),
            'reference_state_and_source_dedup_passed':True,'formal_training_started':False}
    (a.output_dir/'report.json').open('x').write(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
