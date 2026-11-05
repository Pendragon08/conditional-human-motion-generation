"""Run the pilot audit after the specifically identified solver finishes."""
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    pid=int((HERE/'pid').read_text().strip())
    while True:
        proc=Path('/proc')/str(pid)
        if not proc.exists(): break
        try:
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ')
            status=(proc/'stat').read_text().split()[2]
        except FileNotFoundError:
            break
        if status=='Z': break
        if b'pyroki_grab_legacy_pilot.py' not in cmd:
            raise RuntimeError('PID no longer belongs to the expected solver')
        time.sleep(15)
    if not (HERE/'report.json').is_file() or not (HERE/'raw_pilot.npz').is_file():
        raise RuntimeError('solver ended without a complete report and artifact')
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=['nice','-n','15','taskset','-c','19',
             '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python',
             '-u',str(HERE/'audit_pyroki_grab_pilot.py')]
    print('solver finished; starting audit',flush=True)
    subprocess.run(command,check=True,cwd=HERE,env=env)
    print('audit completed',flush=True)


if __name__=='__main__':main()
