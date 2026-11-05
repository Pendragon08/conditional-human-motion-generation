"""All2000 body-tracking rollouts through the unchanged pinned native evaluator."""
import argparse
import hashlib
import json
import math
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'stage3_supermimic29_20261103';N=D/'native_universal_20261104'
sys.path.insert(0,str(D/'src'))
from native_tracker2000_worker_v1 import dataset,sha,read,atomic,TASKS,LAST

def main():
    p=argparse.ArgumentParser();p.add_argument('--training-output',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert a.output.resolve().is_relative_to(N.resolve())
    assert a.training_output.resolve().is_relative_to(N.resolve())
    a.output.mkdir(exist_ok=False);began=time.monotonic()
    state={'status':'initializing','pid':os.getpid(),'optimizer_updates':0,
           'object_task_success_measured':False,'scope':'native29-body reference tracking only'}
    def save():
        state['elapsed_seconds']=time.monotonic()-began;atomic(a.output/'status.json',state)
    save()
    try:
        receipt,manifest,msha=dataset();train=read(a.training_output/'status.json')
        assert train['status']=='native2000_training_budget_completed_evaluation_pending'
        assert train['manifest_sha256']==msha and train['real_vector_steps']==72000
        checkpoint=a.training_output/'logs/last.pt';before=sha(checkpoint)
        assert before==train['checkpoint']['sha256'] and train['native_iteration']==LAST
        source=N/'source/supermimic/evaluate.py'
        assert sha(source)=='5b8923a8c46dcbf2b19e048729ab0039b9ffe1a45777f83eb355df14b245bb36'
        spec=read(N/'environment_spec.json')['spec'];sys.path[:0]=spec['module_paths']
        import torch
        torch.set_num_threads(2);torch.manual_seed(20261105)
        torch.cuda.set_per_process_memory_fraction(min(.22,10*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
        logs=a.output/'logs';logs.mkdir();name='ckpt_'+str(LAST).zfill(6)
        (logs/(name+'.pt')).symlink_to(checkpoint)
        command=[str(source),'--run_dir',str(logs),'--motion_dir',receipt['native_motion_dir'],
                 '--visualizer','none','--device','cuda:0']
        state.update(status='native2000_evaluation_running',manifest_sha256=msha,
                     checkpoint_sha256=before,command=command);save()
        sys.argv=command;g=runpy.run_path(str(source),run_name='__main__')
        result=read(logs/(name+'_eval.json'))
        assert len(result['per_clip'])==2000 and len(result['failed'])==result['n_failed']
        assert len(g['motion'].names)==2000 and g['env'].N==2000
        assert all(math.isfinite(v) for v in result['per_clip'].values())
        assert math.isfinite(result['norm_return']) and sha(checkpoint)==before
        records=read(manifest)['records'];mapping={Path(x['csv']).stem:x['task'] for x in records}
        assert len(mapping)==2000
        per_task={t:{'references':0,'failed':0,'return_sum':0.} for t in TASKS}
        failed={Path(x).stem for x in result['failed']}
        for clip,value in result['per_clip'].items():
            key=Path(clip).stem;task=mapping[key];r=per_task[task]
            r['references']+=1;r['failed']+=int(key in failed);r['return_sum']+=value
        for task,row in per_task.items():
            assert row['references']==500
            row['body_tracking_survival_rate']=(500-row['failed'])/500
            row['mean_normalized_tracking_return']=row.pop('return_sum')/500
        report={**state,'status':'native2000_body_tracking_evaluation_completed',
                'reference_count':2000,'num_envs':2000,'per_task':per_task,
                'n_failed':result['n_failed'],'body_tracking_survival_rate':(2000-result['n_failed'])/2000,
                'normalized_tracking_return':result['norm_return'],'checkpoint_unchanged':True,
                'per_clip_report':str(logs/(name+'_eval.json')),
                'per_clip_report_sha256':sha(logs/(name+'_eval.json')),
                'elapsed_seconds':time.monotonic()-began,
                'evaluation_distribution':'all training reference clips once from first frame; native friction1/no observation-action noise/no domain randomization',
                'not_held_out_reference_generalization':True}
        atomic(a.output/'report.json',report);state.update(report);save();print(json.dumps(report),flush=True)
    except BaseException as error:
        state.update(status='failed_preserved',error=repr(error),traceback=traceback.format_exc());save();raise

if __name__=='__main__':main()
