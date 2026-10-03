"""Bounded official Open-Drawer diagnostic; never a formal baseline result."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

from r3_dreamcontrol_remaining_queue import ROOT, TRAIN, MON, environment

EXPECTED = '30fddfc98008e380fe37d947e995c7875ead6113544b810d0254dc23450b6a7c'


def main():
    cabinet = TRAIN / 'assets/cabinet.usd'
    assert cabinet.stat().st_size == 17599
    assert hashlib.sha256(cabinet.read_bytes()).hexdigest() == EXPECTED
    # No running formal trainer or scheduler may be interrupted or duplicated.
    ps = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    assert not any('r3_dreamcontrol_remaining_queue.py' in x and '--check' not in x
                   for x in ps.splitlines() if not x.startswith(('sh ', 'bash ')))
    assert not any('r3_i1_train_matched.py' in x for x in ps.splitlines()
                   if not x.startswith(('sh ', 'bash ')))
    log = MON / 'r3_cabinet_asset_witness_20261003.log'
    assert not log.exists(), 'Do not repeat a completed/failed witness automatically'
    command = [str(ROOT/'third_party/isaacsim-4.5.0/python.sh'), '-u',
               str(ROOT/'scripts/r3_i1_train_matched.py'),
               '--task=Isaac-Motion-Tracking-Open-Drawer-v0', '--device=cuda:1',
               '--num_envs=16', '--max_iterations=1', '--seed=42', '--headless',
               '--telemetry-udp=127.0.0.1:40123']
    with log.open('wb') as f:
        p = subprocess.run(command, cwd=TRAIN, env=environment(), stdout=f,
                           stderr=subprocess.STDOUT, timeout=300)
    text = log.read_text(errors='replace')
    assert p.returncode == 0
    assert re.search(r'Learning iteration\s+0/1', text), text[-4000:]
    assert 'Traceback (most recent call last)' not in text
    run = re.findall(r'Exact experiment name requested from command line: (.+)', text)[-1].strip()
    model = TRAIN/'logs/rsl_rl/g1'/run/'model_0.pt'
    assert model.is_file() and model.stat().st_size > 0
    result = dict(status='DIAGNOSTIC_PASS', task='Open-Drawer', seed=42,
                  num_envs=16, iterations=1, exit_code=p.returncode,
                  cabinet_sha256=EXPECTED, model=str(model),
                  model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  log=str(log), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                  scope='Asset initialization and one PPO update only; no formal score or full-budget completion')
    (MON/'r3_cabinet_asset_witness_20261003.json').write_text(json.dumps(result, indent=2))
    print('CABINET_ASSET_WITNESS_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
