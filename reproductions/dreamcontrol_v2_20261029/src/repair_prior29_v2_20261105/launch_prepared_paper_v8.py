"""Launch a fully audited prepared continuation exactly once."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

S = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
D = S / 'paper_pipeline_v8_launch'
N = S / 'src/paper_protocol_v8'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def create(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


with (D / 'launch.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (D / 'launch.json').exists():
        print((D / 'launch.json').read_text())
        raise SystemExit(0)
    assert not list(D.glob('*_launch.json')), 'Partial launch; inspect before continuing'
    for p, expected in read(D / 'source_binding.json').items():
        assert sha(p) == expected, p
    assert sha(D / 'config.json') == 'eb3c79702a08589d7eb73a40605e1f0d1515254420bdfbd20bfbf9de72d2a0c2'
    assert sha(D / 'independent_pick_readback.json') == '6e3b994e665ff10c69810953c4b2327fc916789e0ba14a1d66c7d3781a3891a5'
    assert read(D / 'initializer_cpu_witness.json')['status'] == 'passed'
    assert not (S / 'new_references_paper_v8').exists()
    sys.path.insert(0, str(N))
    import pipeline_controller_v2 as p
    plan, cfg, output = p.setup_plan(D / 'config.json')
    snapshot = p.resource_snapshot(S.parent)
    p.resource_guard(snapshot, plan['cohort_write_budget_bytes'])
    for pid in (1899660, 1899674, 1899675, 1899676, 1914254, 1914522):
        cmd = Path(f'/proc/{pid}/cmdline')
        assert not cmd.exists() or str(S).encode() not in cmd.read_bytes()
    create(D / 'actual_prelaunch_resources.json', snapshot)
    commands = {'generation':[plan['sample_python'],'-B','-u',str(N / 'pipeline_controller_v2.py'),
        '--plan',str(D / 'config.json'),'--plan-sha256',sha(D / 'config.json')]}
    for key, name in [('evaluation','evaluation_controller_paper_v8.py'),
                      ('supermimic','supermimic_repaired_controller_paper_v8.py'),
                      ('videos','report_video_sources_paper_v8.py')]:
        commands[key] = [plan['sample_python'],'-B','-u',str(S / 'src' / name)]
    processes = {}
    for key, command in commands.items():
        env = {**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
        if key == 'videos':
            env.update(plan['postopt_env']);env['JAX_PLATFORMS']='cpu';env['CUDA_VISIBLE_DEVICES']=''
        with (D / (key + '.log')).open('xb') as log:
            child = subprocess.Popen(command,cwd=S,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[key] = {'pid':child.pid,'argv':command}
        create(D / (key + '_launch.json'),processes[key])
    create(D / 'launch.json', {'status':'launched','plan_sha256':sha(D / 'config.json'),
        'launcher_sha256':sha(__file__),'processes':processes})
    print((D / 'launch.json').read_text())
