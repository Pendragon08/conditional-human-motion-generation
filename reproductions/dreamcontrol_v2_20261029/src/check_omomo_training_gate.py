"""Exercise the actual training entry's fail-closed gate on the schema preview.

The child receives no visible GPU. It must reject unaccepted data before any
resource check or training setup. No checkpoint or model update is allowed.
"""
import hashlib,json,os,subprocess,sys
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT=ROOT/'omomo_training_schema_preview_20261101'
SOURCE=ROOT/'training_disk_guard_v2_20261101/train_robot_prior.py'
INIT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen/save/omnicontrol_ckpt/model_humanml3d.pt')
HERE=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    run=HERE/'unused_run';run.mkdir(exist_ok=False)
    config={'dataset_root':str(ROOT),'manifest':str(INPUT/'manifest.json'),
            'stats':str(ROOT/'omomo_mixed_encoding_preview_20261101/stats.npz'),'initial_checkpoint':str(INIT),
            'run_dir':str(run),'min_std':.001,'batch_size':2,'seed':1,'max_steps':1,'checkpoint_every':1,
            'guidance_steps':1,'guidance_scale':.5,'learning_rate':1e-5,'weight_decay':0.,
            'device':'cuda:0','checkpoint_size_estimate':1024**3}
    with (HERE/'config.json').open('x') as f:json.dump(config,f,indent=2)
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    proc=subprocess.run([sys.executable,str(SOURCE),'--config',str(HERE/'config.json'),'--run'],env=env,
                        capture_output=True,text=True,timeout=120)
    with (HERE/'stdout.log').open('x') as f:f.write(proc.stdout)
    with (HERE/'stderr.log').open('x') as f:f.write(proc.stderr)
    expected='training blocked: source and applicable quality acceptance not yet implemented'
    if proc.returncode==0 or expected not in proc.stderr or list(run.iterdir()):
        raise RuntimeError('unexpected gate result; inspect retained logs')
    result={'entry_sha256':digest(SOURCE),'config_sha256':digest(HERE/'config.json'),
            'manifest_sha256':digest(INPUT/'manifest.json'),'stderr_sha256':digest(HERE/'stderr.log'),
            'stdout_sha256':digest(HERE/'stdout.log'),'script_sha256':digest(Path(__file__)),
            'exit_code':proc.returncode,'expected_block_observed':True,'run_dir_empty':True,
            'training_ready':False,'limitations':['This is a negative test of the current gate, not a complete acceptance implementation.',
                'State preflight is omitted here; caption, source, time and physical quality remain unaccepted.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))

if __name__=='__main__':main()
