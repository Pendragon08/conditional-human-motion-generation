"""Nine remaining official baselines; independent telemetry, no live renderer in PPO."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MON = ROOT / 'artifacts/monitoring'
TRAIN = ROOT / 'third_party/DreamControl/Training'
JOBS = [('Jump', 'Jump_sim2', 1), ('Sit', 'Sit_sim2', 1),
        ('Pick', 'Pick_sim2', 67), ('Bimanual-Pick', 'Bimanual_Pick_sim2', 1),
        ('Ground-Pick', 'Ground_Pick_sim2', 1), ('Ground-Pick-Top', 'Ground_Pick_Top_sim2', 1),
        ('Button-Press', 'Button_Press_sim2', 96), ('Open-Drawer', 'Open_Drawer_sim2', 602),
        ('Open-Door', 'Open_Door_sim2', 602)]
PREFIX = 'r3_dreamcontrol_remaining_20261001'

def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n' if isinstance(obj, dict) else obj + '\n')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def environment():
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): env[key] = '4'
    env.update(WANDB_MODE='disabled', WANDB_SILENT='true', HYDRA_FULL_ERROR='1',
               TMPDIR=str(ROOT/'cache/r3_tmp'), XDG_CACHE_HOME=str(ROOT/'cache/r3_xdg'),
               MPLCONFIGDIR=str(ROOT/'cache/r3_matplotlib'))
    env['PYTHONPATH'] = ':'.join(str(p) for p in [ROOT/'scripts', ROOT/'envs/dreamcontrol-r3/site-packages',
        TRAIN/'source/isaaclab', TRAIN/'source/isaaclab_assets', TRAIN/'source/isaaclab_tasks',
        TRAIN/'source/isaaclab_rl', TRAIN/'isaac_utils', TRAIN/'scripts/reinforcement_learning/rsl_rl'])
    for key in ('CUDA_VISIBLE_DEVICES', 'DREAMCONTROL_KICK_REF_PATH', 'DREAMCONTROL_PUNCH_REF_PATH'):
        env.pop(key, None)
    return env

def execute(cmd, log, env, seconds):
    with log.open('wb') as f:
        proc = subprocess.Popen(cmd, cwd=TRAIN, env=env, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try: return proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            return 124

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); MON.mkdir(parents=True, exist_ok=True)
    manifest = {'tasks': [], 'envs': 8192, 'iterations': 2000, 'seed': 42,
                'gpu': 1, 'eval_envs': 1000, 'eval_frames': 500, 'excluded_completed': ['Kick','Punch']}
    for task, ref, count in JOBS:
        files = sorted((ROOT/'third_party/DreamControl/TrajGen/sample'/ref).glob('*.pkl'))
        assert len(files) == count, (task, len(files), count)
        assert all(p.stat().st_size > 130 for p in files), task
        manifest['tasks'].append({'task': task, 'ref': ref, 'count': count,
                                'files': {p.name: digest(p) for p in files}})
    gate = json.loads((MON/'r3_i1_live_training_view_gate.json').read_text())
    assert gate['user_confirmed_local_stream'] and gate['mode'] == 'sampled_actual_training_simulation'
    assert (ROOT/'third_party/isaacsim-4.5.0/python.sh').is_file()
    write(ROOT/'results/r3_dreamcontrol_remaining_manifest.json', manifest)
    print('REMAINING_QUEUE_PREFLIGHT_PASS tasks=9 envs=8192 seed=42', flush=True)
    if args.check: return
    with (MON/(PREFIX+'.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = environment(); python = str(ROOT/'third_party/isaacsim-4.5.0/python.sh')
        queue = MON/(PREFIX+'.state')
        try:
            for task, _, _ in JOBS:
                state = MON/f'{PREFIX}_{task}.state'
                if state.is_file() and state.read_text().startswith('COMPLETE'): continue
                log = MON/f'{PREFIX}_{task}_train.log'
                if log.exists(): raise RuntimeError(f'Existing attempt must be audited before resume: {log}')
                task_id = f'Isaac-Motion-Tracking-{task}-v0'
                write(queue, f'RUNNING task={task} stage=train remaining_tasks=9')
                write(state, f'RUNNING task={task} stage=train envs=8192 iterations=2000 seed=42 gpu=1')
                write(MON/'r3_active_dreamcontrol_training.json', {'task':task,'task_id':task_id,
                    'queue_state':str(queue),'log':str(log),'stage':'train','budget':2000,'gpu':1})
                rc = execute([python,'-u',str(ROOT/'scripts/r3_i1_train_matched.py'),
                    '--task='+task_id,'--device=cuda:1','--num_envs=8192','--max_iterations=2000',
                    '--seed=42','--headless','--telemetry-udp=127.0.0.1:40123'],log,env,172800)
                text = log.read_text(errors='replace')
                if rc or not re.search(r'Learning iteration\s+1999/2000', text):
                    raise RuntimeError(f'{task} train_exit={rc}; final iteration missing or failure')
                run = re.findall(r'Exact experiment name requested from command line: (.+)',text)[-1].strip()
                model = TRAIN/'logs/rsl_rl/g1'/run/'model_1999.pt'
                assert model.is_file() and model.stat().st_size > 0
                write(queue,f'RUNNING task={task} stage=eval')
                write(state,f'RUNNING task={task} stage=eval checkpoint={model}')
                eval_log = MON/f'{PREFIX}_{task}_eval.log'; eid=f'r3_remaining_{task}_8192_eval'
                rc = execute([python,'-u','scripts/reinforcement_learning/rsl_rl/play_eval.py',
                    '--task='+task_id,'--device=cuda:1','--num_envs=1000','--path='+str(model),
                    '--headless','--video','--video_length=500','--id='+eid],eval_log,env,7200)
                matches=re.findall(r'Success rate:\s*([\d.]+)\s*%',eval_log.read_text(errors='replace'))
                if rc or not matches: raise RuntimeError(f'{task} eval_exit={rc}; no success rate')
                videos=list((model.parent/'videos/play').glob('*'+eid+'*.mp4')); assert videos
                video=videos[0]
                duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                    'format=duration','-of','default=nw=1:nk=1',str(video)],text=True)); assert duration>0
                result={'task':task,'status':'complete','success_percent':float(matches[-1]),
                    'checkpoint':str(model),'checkpoint_sha256':digest(model),'video':str(video),
                    'video_sha256':digest(video),'duration_seconds':duration,'train_exit':0,'eval_exit':0,
                    'num_envs':8192,'iterations':2000,'seed':42,'eval_envs':1000,'eval_frames':500,
                    'train_log':str(log),'eval_log':str(eval_log)}
                write(ROOT/f'results/r3_dreamcontrol_official_{task}_8192_eval.json',result)
                write(state,f'COMPLETE task={task} success_percent={result["success_percent"]} checkpoint={model}')
                print(json.dumps(result),flush=True); time.sleep(10)
            write(queue,'COMPLETE remaining_tasks=9 training_and_evaluation')
        except BaseException as e:
            write(queue,f'BLOCKED task={task} reason={type(e).__name__}:{e}')
            write(state,f'BLOCKED task={task} reason={type(e).__name__}:{e}')
            raise

if __name__ == '__main__': main()
