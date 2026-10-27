"""Paper-aligned asynchronous scheduling over the pinned official online runner.

Uses the existing native PyTorch ScaleBFM compatibility adapter. It is not a
TensorRT/real-time reproduction. Does not alter training or vendor sources.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import threading
import traceback

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--prompt', default='walk forward')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--compile-planner', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.16, 0)
    upstream = a.code_root / 'upstream'
    sys.path[:0] = [str(a.code_root/'scripts'), str(upstream), str(upstream/'deploy')]
    source_paths = [Path(__file__), a.code_root/'scripts/torch_bfm_agent.py']
    source_paths += sorted(upstream.rglob('*.py'))
    pins = {str(f): digest(f) for f in source_paths}
    record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status='RUNNING', prompt=a.prompt, seed=a.seed,
                  checkpoint=str(a.checkpoint), checkpoint_sha256=digest(a.checkpoint),
                  official_commit='5a79d03cab824eb5917271ba720294e8a1b9d374',
                  source_hashes=pins, paper_benchmark_reproduced=False,
                  execution_backend='native_pytorch_compatibility',
                  timing='official wall-clock history and asynchronous scheduling',
                  changes=['applied_frame=0 enables official asynchronous branch',
                           'real observed prefix', 'buffer threshold=10 control frames',
                           'official five-frame trajectory blending'],
                  limitations=['TensorRT not installed; no 50Hz real-time claim',
                               'partial planner checkpoint; text-only branch; no target-reaching data'])
    (a.output/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    spec = importlib.util.spec_from_file_location('paper_faithful_official_runner', upstream/'deploy/run_online_generation.py')
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    compile_witness = []
    if a.compile_planner:
        original_compile = torch.compile

        def compile_without_cudagraphs(module, *args, **kwargs):
            if kwargs.get('mode') == 'reduce-overhead':
                kwargs.pop('mode')
                kwargs['options'] = {'triton.cudagraphs': False}
            return original_compile(module, *args, **kwargs)

        # The official DiT also retains text KV tensors between forwards.
        # These cached tensors outlive CUDA-graph output storage in Torch 2.5.
        # Keep Inductor kernels, disable only graph replay in this process.
        torch.compile = compile_without_cudagraphs
        from reactivebfm.utils.training.sampling import TextConditionCFGSampleModel

        def cloned_cfg(self, x, timesteps, y=None):
            # Torch 2.5 reduce-overhead reuses CUDA-graph output storage across
            # the conditional and unconditional calls. Clone outside compiled
            # forward before the second call; CFG arithmetic is unchanged.
            conditioned = self.model(x, timesteps, y).clone()
            baseline = dict(y)
            baseline['text_uncond'] = True
            without_text = self.model(x, timesteps, baseline).clone()
            scale = y['scale'].view(-1, 1, 1, 1)
            result = without_text + scale * (conditioned - without_text)
            if not compile_witness:
                eager = self.model._orig_mod
                original_c = eager(x, timesteps, y)
                original_u = eager(x, timesteps, baseline)
                expected = original_u + scale * (original_c-original_u)
                error = float((result-expected).abs().max())
                torch.testing.assert_close(result, expected, atol=2e-4, rtol=2e-4)
                compile_witness.append({'eager_vs_compiled_cfg_max_abs': error,
                                        'atol': 2e-4, 'rtol': 2e-4,
                                        'scope': 'first actual planner CFG invocation'})
            return result

        TextConditionCFGSampleModel.forward = cloned_cfg
        record['changes'].append('Torch2.5 compiled CFG output clones prevent CUDA-graph storage reuse; eager parity asserted')
        record['changes'].append('Inductor compilation without CUDA graphs preserves persistent text KV cache lifetime')
    from scalebridge.simulator.mujoco_simulator import MujocoSimulator
    original_apply = MujocoSimulator.apply_action
    rows, prefixes, calls = [], [], []
    original_job = runner._run_planner_job
    planner_idle = threading.Event()
    planner_idle.set()

    def apply(simulator, *args, **kwargs):
        answer = original_apply(simulator, *args, **kwargs)
        rows.append({'time': float(simulator.mujoco_data.time),
                     'qpos': simulator.mujoco_data.qpos.copy(),
                     'body_pos': simulator.mujoco_data.xpos.copy()})
        if not np.isfinite(rows[-1]['qpos']).all():
            raise FloatingPointError('Nonfinite executed simulator state')
        return answer

    def job(planner, prefix, *args, **kwargs):
        planner_idle.clear()
        begin = len(rows)
        try:
            value = original_job(planner, prefix, *args, **kwargs)
            prefixes.append(prefix.copy())
            calls.append({'physics_calls_before': begin, 'physics_calls_after': len(rows),
                          'elapsed_s': value.elapsed_s, 'prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest()})
            return value
        finally:
            planner_idle.set()

    MujocoSimulator.apply_action = apply
    runner._run_planner_job = job
    args = ['--model_path', str(a.checkpoint),
            '--policy_checkpoint', str(a.runtime_root/'weights/ScaleBFM_m_model_22200.pt'),
            '--text_prompt', a.prompt, '--init_pose', 'A_pose',
            '--simulator', 'mujoco_simulator', '--headless',
            '--planner_device', 'cuda', '--deploy_device', 'cuda',
            '--planner_data_dir', str(a.runtime_root/'data'),
            '--planner_stats_dataset', 'kungfu_g1_36dim',
            '--no-reference_forcing', '--replan_prefix_mode', 'real',
            '--applied_frame', '0', '--regenerate_threshold', '10',
            '--reference_blend_frames', '5', '--max_policy_steps', str(a.steps),
            '--warmup_steps', '20', '--seed', str(a.seed),
            '--output_root', str(a.output/'session'),
            '--scalebridge_override', 'agent._target_=torch_bfm_agent.TorchBFMAgent']
    record['official_arguments'] = args
    if a.compile_planner:
        args.append('--planner_compile')
    sys.argv = [str(upstream/'deploy/run_online_generation.py'), *args]
    started = time.monotonic()
    code = 0
    try:
        runner.main()
        if not planner_idle.wait(60):
            raise TimeoutError('Final planner job did not finish within 60 seconds')
        summaries = []
        events = []
        for f in (a.output/'session').rglob('*.json'):
            obj = json.loads(f.read_text())
            if isinstance(obj, dict) and 'stop_reason' in obj:
                summaries.append({'file': str(f), 'stop_reason': obj['stop_reason']})
                events.extend(obj.get('events', []))
        record['session_summaries'] = summaries
        from collections import Counter
        record['event_counts'] = dict(Counter(x.get('kind', 'unknown') for x in events))
        record['effective_replans'] = sum(x.get('kind')=='planner_activated' for x in events)
        if len(rows) != a.steps or any(x['stop_reason']=='exception' for x in summaries):
            raise RuntimeError('Official loop did not finish the requested simulation calls')
        if len(calls) < 2:
            raise RuntimeError('No feedback replan completed')
        record['status'] = ('ASYNC_FEEDBACK_APPLIED' if record['effective_replans'] else 'ASYNC_EXECUTED_NO_FRESH_PLAN')
    except Exception as exc:
        code = 1
        record.update(status='FAILED', error=str(exc), traceback=traceback.format_exc())
    if rows:
        np.savez_compressed(a.output/'executed_states.npz',
                            time=np.array([x['time'] for x in rows]),
                            qpos=np.stack([x['qpos'] for x in rows]),
                            body_pos=np.stack([x['body_pos'] for x in rows]))
        record['minimum_root_height_m'] = float(min(x['qpos'][2] for x in rows))
    if prefixes:
        np.savez_compressed(a.output/'observed_prefixes.npz', prefix=np.stack(prefixes))
    record.update(exit_code=code, policy_calls=len(rows), planner_calls=calls,
                  compile_parity_witness=compile_witness,
                  elapsed_s=time.monotonic()-started,
                  sources_unchanged=all(digest(f)==v for f,v in pins.items()),
                  checkpoint_unchanged=digest(a.checkpoint)==record['checkpoint_sha256'],
                  ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (a.output/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'}), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
