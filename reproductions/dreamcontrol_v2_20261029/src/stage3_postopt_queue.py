"""Run the two named Pick candidate fits once, sequentially, with a5% GPU quota."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','JAX_PLATFORMS':'cuda,cpu',
     'XLA_FLAGS':'--xla_gpu_autotune_level=0','XLA_PYTHON_CLIENT_PREALLOCATE':'true',
     'XLA_PYTHON_CLIENT_MEM_FRACTION':'0.05','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
     'PATH':'/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
     'LD_LIBRARY_PATH':'/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu',
     'PYTHONPATH':':'.join(map(str,[R/'jax_cuda12_plugin_0_6_2_20261031',
         R/'pyroki_cpu_deps_20261030/extra-packages',R/'pyroki_cpu_deps_20261030/site-packages',
         Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/pyroki/src'),
         Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/jaxls/src')]))}
status={'pid':os.getpid(),'status':'running','completed':[],'PPO_started':False}


def save():
    p=S/'postopt_queue_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(status,indent=2));tmp.replace(p)


try:
    save()
    for item in json.loads((S/'postopt_queue.json').read_text())['items']:
        assert shutil.disk_usage('/data').free>8*1024**3+356*1024**2+100*1024**2
        output=Path(item['output']);assert not output.exists()
        status['current']=item;save()
        command=['timeout','300','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python',
                 '-u',str(S/'src/stage3_postopt_pick29.py'),'--sample',item['sample'],'--output-dir',item['output']]
        with output.with_suffix('.log').open('x') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('postopt exit '+str(result.returncode)+'; '+str(output.with_suffix('.log')))
        report=json.loads((output/'report.json').read_text())
        status['completed'].append({'output':str(output),'passed_kinematic_checks':report['passed_requested_kinematic_checks']})
        save()
    status.update(status='completed',native_scene_replay='pending');save()
except Exception as exc:
    status.update(status='failed_preserved_outputs',error=repr(exc));save();raise
