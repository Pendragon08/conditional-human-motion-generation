"""Durable Pick completion/evaluation handoff to the authorized eight-task queue.

Other task adapters are explicitly pending, to be implemented by the scheduled
agent. This controller does not launch old27DoF tasks or relabel Pick as them.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';Q=R/'stage3_skill_sequence_20261103'
RUN=S/'runs/pick29_8912_5000_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pick_transition(training,evaluation,checkpoint_exists,guard_released,total_updates=5000):
    """Do not confuse pending/failed training, a final evaluation, or task success."""
    if training is None:return 'training_status_missing'
    if training['status']=='failed':return 'training_failed_needs_repair'
    if training['status']!='ppo_completed':return 'waiting_for_pick_training'
    if training['policy_updates']!=total_updates or not checkpoint_exists:return 'completion_integrity_needs_repair'
    if evaluation is not None:
        if evaluation.get('final_evaluation') and evaluation.get('status')=='evaluation_completed' and evaluation.get('num_envs')==100:
            return 'archive_pick_outcome_and_prepare_next'
        return 'evaluation_integrity_needs_repair'
    return 'ready_to_evaluate_pick' if guard_released else 'waiting_for_pick_cleanup'


def read(path):return json.loads(path.read_text()) if path.exists() else None


def next_after_pick(spec):
    tasks=spec['tasks']
    if not tasks or tasks[0]['id']!='pick' or len(tasks)<2:
        raise ValueError('Pick must be followed by an explicit configured step')
    next_task=tasks[1]
    if next_task['id']=='pick_supermimic':
        action=('scheduled agent first implements the SuperMimic backend using the existing TextOp environment, '
                'preserving Pick29 live observations/actions, references, scene and matched simulation-step budget; '
                'validate, train and evaluate it before continuing the remaining skills')
    else:
        action=('scheduled agent implements and validates '+next_task['id']+
                '29DoF using the existing diffusion prior, then trains/evaluates it')
    return next_task['id'],action


def align_task_states(state,spec):
    """Insert newly authorized queue steps without losing any existing outcomes."""
    previous={task['id']:task for task in state['tasks']}
    return [{'id':task['id'],'status':'pending_implementation',**previous.get(task['id'],{})}
            for task in spec['tasks']]


def save(state):
    temp=Q/'progress.tmp';temp.write_text(json.dumps(state,indent=2,allow_nan=False));temp.replace(Q/'progress.json')


def probe(allow_launch):
    spec=read(Q/'sequence.json')
    state=read(Q/'progress.json') or {'current_index':0,'current_task':'pick','tasks':[
        {'id':task['id'],'status':'running' if k==0 else 'pending_implementation'} for k,task in enumerate(spec['tasks'])]}
    if state['current_index']>0:return state
    state['tasks']=align_task_states(state,spec)
    active=read(S/'active_pick_run.json')
    run=Path(active['run_dir']) if active else RUN
    target=int(active['total_updates']) if active else 5000
    guard_path=Path(active['guard_status']) if active else S/'formal_guard_status.json'
    training=read(run/'status.json');guard=read(guard_path)
    checkpoint=run/'ppo'/('model_%d.pt'%(target-1))
    suffix='_'+str(target) if active else ''
    output=Q/'evaluations'/('pick_final'+suffix+'_v1');evaluation=read(output/'report.json')
    launch_path=Q/('pick_evaluation'+suffix+'_launch.json')
    phase=pick_transition(training,evaluation,checkpoint.exists(),bool(guard and guard['status']=='training_completed'),target)
    state.update(status=phase,current_task='pick',controller_pid=os.getpid(),last_checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    state['tasks'][0].update(status=phase,policy_updates=training.get('policy_updates',0) if training else 0,
                           total_updates=target,run_dir=str(run))
    if phase=='ready_to_evaluate_pick':
        failure=read(output/'failure.json')
        launch=read(launch_path)
        if failure:
            state.update(status='pick_evaluation_failed_needs_repair',failure=failure)
        elif launch:
            proc=Path('/proc/%s/cmdline'%launch['pid'])
            state['status']='evaluating_pick' if proc.exists() else 'pick_evaluation_process_ended_check_logs'
        elif allow_launch:
            free=shutil.disk_usage('/data').free
            query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            inventory={int(line.split(',')[0]):int(line.split(',')[1]) for line in query.stdout.splitlines() if ',' in line}
            if free<8*1024**3+456*1024**2 or inventory.get(1,0)<8192:
                state['status']='waiting_for_pick_evaluation_resources'
            else:
                envspec=read(S/'env_spec_v2.json')
                command=[envspec['smoke_command'][0],'-u',str(Q/'src/stage3_pick29_evaluate.py'),
                    '--headless','--device','cuda:1','--checkpoint',str(checkpoint),'--num-envs','100',
                    '--steps','500','--output-dir',str(output)]
                with (Q/('pick_evaluation'+suffix+'.log')).open('x') as log:
                    child=subprocess.Popen(['timeout','--kill-after=20s','180']+command,
                        env={**os.environ,**envspec['env']},cwd=envspec['cwd'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with launch_path.open('x') as f:
                    json.dump({'pid':child.pid,'command':command,'checkpoint_sha256':sha(checkpoint),
                               'entrypoint_sha256':sha(Q/'src/stage3_pick29_evaluate.py')},f,indent=2)
                state['status']='evaluating_pick'
    elif phase=='archive_pick_outcome_and_prepare_next':
        launch=read(launch_path)
        if not launch or sha(checkpoint)!=evaluation['checkpoint_sha256'] or launch['checkpoint_sha256']!=evaluation['checkpoint_sha256']:
            state['status']='evaluation_checkpoint_binding_needs_repair'
        else:
            completion=Q/'completions/pick.json'
            if not completion.exists():
                with completion.open('x') as f:json.dump({'baseline_training_budget_completed':True,
                    'training_status_sha256':sha(run/'status.json'),'total_training_updates':target,
                    'evaluation_report':str(output/'report.json'),
                    'evaluation_sha256':sha(output/'report.json'),'checkpoint_sha256':sha(checkpoint),
                    'observed_success_ratio':evaluation['success_ratio'],'episodes':100,
                    'claim':'baseline budget and evaluation finished; observed success is reported without inventing a pass threshold'},f,indent=2)
            state['tasks'][0].update(status='baseline_evaluated',completion=str(completion),observed_success_ratio=evaluation['success_ratio'])
            next_task,action=next_after_pick(spec)
            state.update(current_index=1,current_task=next_task,status='next_task_preparation_requested',
                         next_action=action)
    save(state);return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');p.add_argument('--refresh',action='store_true')
    args=p.parse_args()
    with (Q/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            state=probe(allow_launch=args.watch)
            if not args.watch or state['current_index']>0:
                print(json.dumps(state),flush=True);return
            time.sleep(30)


if __name__=='__main__':main()
