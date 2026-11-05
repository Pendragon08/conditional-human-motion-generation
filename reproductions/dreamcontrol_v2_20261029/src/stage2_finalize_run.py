"""One finite experiment continuation: wait for this run, then sample and fit.

Never restarts training. Does not create a recurring automation, launch RL,
delete rejected outputs, or treat an uncalibrated smoke scene as accepted.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE = ROOT / 'stage2_prior29_20261103'
PYTHON = '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'


def write_status(data):
    path = STAGE / 'finalization_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(temp, path)


def run_child(name, command, env, timeout):
    with (STAGE / (name + '.log')).open('x') as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if process.returncode:
        raise RuntimeError(name + ' exit ' + str(process.returncode) + '; inspect preserved log')


def main():
    import hashlib
    config_path = STAGE / 'config.json'
    cfg = json.loads(config_path.read_text())
    run = STAGE / 'runs' / cfg['run_name']
    write_status({'status': 'waiting_for_existing_training_run', 'training_run': str(run), 'pid': os.getpid()})
    while True:
        if (run / 'failure.json').exists():
            raise RuntimeError('existing training run failed: ' + (run / 'failure.json').read_text())
        if (run / 'status.json').exists():
            status = json.loads((run / 'status.json').read_text())
            if status['status'] == 'completed':
                break
            try:
                os.kill(status['pid'], 0)
            except ProcessLookupError:
                raise RuntimeError('training process ended without completed status')
        time.sleep(30)
    if status['cursor']['step'] != cfg['max_steps']:
        raise RuntimeError('training did not reach configured final step')
    checkpoint = Path(status['last_checkpoint']['path'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != status['last_checkpoint']['sha256']:
        raise RuntimeError('final checkpoint hash mismatch')
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    samples = STAGE / 'reports/trained_final_sampling'
    write_status({'status': 'sampling_final_trained_checkpoint', 'checkpoint': str(checkpoint), 'pid': os.getpid()})
    run_child('trained_final_sampling', ['nice', '-n', '15', PYTHON, '-u', str(STAGE / 'src/stage2_sample_prior.py'),
                                      '--config', str(config_path), '--checkpoint', str(checkpoint), '--output-dir', str(samples)], env, 900)
    import numpy as np
    sampling = json.loads((samples / 'report.json').read_text())
    template = json.loads((STAGE / 'reports/postopt_smoke_scene.json').read_text())
    extra = [str(ROOT / 'jax_cuda12_plugin_0_6_2_20261031'), str(ROOT / 'pyroki_cpu_deps_20261030/extra-packages'),
             str(ROOT / 'pyroki_cpu_deps_20261030/site-packages'),
             '/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/pyroki/src',
             '/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/jaxls/src']
    post_env = {**env, 'CUDA_VISIBLE_DEVICES': '0', 'JAX_PLATFORMS': 'cuda,cpu',
                'XLA_FLAGS': '--xla_gpu_autotune_level=0', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'true',
                'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.05', 'OMP_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'PATH': '/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
                'LD_LIBRARY_PATH': '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu', 'PYTHONPATH': ':'.join(extra)}
    post_reports = []
    for index, item in enumerate(sampling['samples']):
        sample = samples / item['sample']
        scene = json.loads(json.dumps(template))
        with np.load(sample, allow_pickle=False) as f:
            goal = f['target_points'][item['right_wrist_target_frame'], 21]
        world_goal = goal @ np.asarray(scene['canonical_to_world_rotation']).T + np.asarray(scene['canonical_to_world_translation'])
        scene['hand_goals'] = [{'frame': item['right_wrist_target_frame'], 'point_index': 21, 'position': world_goal.tolist()}]
        scene_file = STAGE / 'reports' / ('final_scene_%d.json' % index)
        with scene_file.open('x') as f:
            json.dump(scene, f, indent=2)
        output = STAGE / 'reports' / ('trained_final_postopt29_%d' % index)
        write_status({'status': 'postoptimizing_generated_candidate', 'candidate': index, 'pid': os.getpid()})
        run_child('trained_final_postopt29_%d' % index,
                  ['nice', '-n', '15', PYTHON, '-u', str(STAGE / 'src/stage2_postopt29.py'),
                   '--sample', str(sample), '--sample-sha256', item['sha256'], '--scene', str(scene_file), '--output-dir', str(output)],
                  post_env, 300)
        post_reports.append(json.loads((output / 'report.json').read_text()))
    result = {'status': 'completed', 'training_steps': status['cursor']['step'], 'checkpoint': status['last_checkpoint'],
              'final_sampling_report': str(samples / 'report.json'), 'postoptimized_candidates': len(post_reports),
              'passed_requested_kinematic_filters': sum(x['passed_requested_kinematic_filters'] for x in post_reports),
              'calibrated_scene_references': sum(x['eligible_for_calibrated_scene_reference'] for x in post_reports),
              'source_dataset_filtering': False, 'RL_started': False, 'real_robot_deployment': False,
              'scope': 'finite baseline training/sampling/29DoF postoptimization workflow; quality is recorded, not guaranteed',
              'postoptimization_reports': post_reports}
    with (STAGE / 'reports/final_workflow_report.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    write_status({**result, 'pid': os.getpid()})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_status({'status': 'failed_preserved_outputs', 'error': type(exc).__name__ + ': ' + str(exc), 'pid': os.getpid()})
        raise
