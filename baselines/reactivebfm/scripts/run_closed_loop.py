"""Run the official online planner/MuJoCo loop using the native tracker adapter.

No vendor file is edited. Records feedback-prefix digests, actual physics steps
and posthoc simulator state diagnostics. Completion never implies upright success.
"""
import argparse
from collections import deque
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
UPSTREAM = BASE / 'upstream'
sys.path[:0] = [str(Path(__file__).parent), str(UPSTREAM), str(UPSTREAM / 'deploy')]
p = argparse.ArgumentParser()
p.add_argument('--runtime-root', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--steps', type=int, default=400)
p.add_argument('--prompt', default='stand still')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--video', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
state = {'status': 'RUNNING', 'seed': a.seed, 'prompt': a.prompt,
    'checkpoint': str(a.checkpoint), 'reference_forcing': False, 'replan_prefix_mode': 'real',
    'tracker_backend': 'supplementary native PyTorch using official weights/core',
    'feedback_timebase': 'MuJoCo simulation time; actual wall latency remains separately measured',
    'paper_benchmark_reproduced': False, 'physics_steps': 0, 'mujoco_integrator_steps': 0,
    'simulation_seconds': 0., 'legacy_physics_steps_field_means_policy_calls': True, 'planner_calls': [],
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
start = time.monotonic(); code = 0; recorded_prefixes = []; prefix_times = []
try:
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    spec = importlib.util.spec_from_file_location('official_online_runner', UPSTREAM / 'deploy/run_online_generation.py')
    runner = importlib.util.module_from_spec(spec); sys.modules[spec.name] = runner; spec.loader.exec_module(runner)
    # In a serial offline simulator, planning pauses physics. Using wall time
    # would stretch the observed motion prefix when planning takes >one tick.
    # Keep the official resampling math but timestamp actual observations with
    # simulation time. Do not replace the planner's wall-clock timing meter.
    original_seed = runner._seed_motion_history
    original_resample = runner._resample_history
    class SimulationHistory(deque):
        def append(self, item):
            super().append((state['simulation_seconds'], item[1]))
    def seed_history(context, anchor_s, history_len, fps):
        seeded = original_seed(context, 0., history_len, fps)
        result = SimulationHistory(maxlen=history_len)
        for item in seeded: deque.append(result, item)
        return result
    def resample_history(history, anchor_s, context_len, fps):
        return original_resample(history, state['simulation_seconds'], context_len, fps)
    runner._seed_motion_history = seed_history
    runner._resample_history = resample_history
    from scalebridge.simulator.mujoco_simulator import MujocoSimulator
    original_apply = MujocoSimulator.apply_action
    def apply(self, *args, **kwargs):
        before = float(self.mujoco_data.time)
        result = original_apply(self, *args, **kwargs)
        state['physics_steps'] += 1
        duration = float(self.mujoco_data.time) - before
        state['simulation_seconds'] += duration
        state['mujoco_integrator_steps'] += int(round(duration / self.mujoco_model.opt.timestep))
        z = float(self.mujoco_data.qpos[2])
        state['minimum_root_height_m'] = min(state.get('minimum_root_height_m', z), z)
        if not np.isfinite(self.mujoco_data.qpos).all():
            raise ValueError('Nonfinite physical state')
        return result
    MujocoSimulator.apply_action = apply
    original_job = runner._run_planner_job
    def job(planner, prefix, *args, **kwargs):
        row = {'call_index': len(state['planner_calls']), 'prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
            'prefix_shape': list(prefix.shape), 'mean_prefix_root_height_m': float(prefix[:, 2].mean()),
            'physics_steps_before': state['physics_steps'], 'prefix_anchor_sim_seconds': state['simulation_seconds']}
        result = original_job(planner, prefix, *args, **kwargs)
        recorded_prefixes.append(prefix.copy()); prefix_times.append(state['simulation_seconds'])
        row['elapsed_s'] = result.elapsed_s; state['planner_calls'].append(row)
        return result
    runner._run_planner_job = job
    sys.argv = ['official_online_runner', '--model_path', str(a.checkpoint),
        '--policy_checkpoint', str(a.runtime_root / 'weights/ScaleBFM_m_model_22200.pt'),
        '--text_prompt', a.prompt, '--init_pose', 'A_pose', '--simulator', 'mujoco_simulator',
        '--headless', '--planner_device', 'cuda', '--deploy_device', 'cpu',
        '--planner_data_dir', str(a.runtime_root / 'data'),
        '--planner_stats_dataset', 'kungfu_g1_36dim',
        '--no-reference_forcing', '--replan_prefix_mode', 'real', '--applied_frame', '10',
        '--max_policy_steps', str(a.steps), '--warmup_steps', '5', '--seed', str(a.seed),
        '--output_root', str(a.output / 'session'),
        '--scalebridge_override', 'agent._target_=torch_bfm_agent.TorchBFMAgent']
    # A single-component checkpoint expects the component directory directly.
    args_json = json.loads((a.checkpoint.parent / 'args.json').read_text())
    if ',' not in args_json['dataset'] and args_json['dataset'] != 'amass_100style_kungfu_g1_36dim':
        sys.argv[sys.argv.index('--planner_data_dir') + 1] = str(a.runtime_root / 'data' / args_json['dataset'])
    if a.video:
        sys.argv += ['--record_video', '--video_path', str(a.output / 'rollout.mp4')]
    state['official_arguments'] = sys.argv[1:]
    runner.main()
    # The official runner catches exceptions; inspect its saved stop event too.
    json_files = list((a.output / 'session').rglob('*.json'))
    reasons = []
    for f in json_files:
        payload = json.loads(f.read_text())
        if isinstance(payload, dict) and 'stop_reason' in payload:
            reasons.append(payload['stop_reason'])
    state['recorded_stop_reasons'] = reasons
    state['policy_steps'] = state['physics_steps']
    if state['physics_steps'] != a.steps or len(state['planner_calls']) < 2 or 'exception' in reasons:
        raise RuntimeError('Requested physics/replanning loop did not complete')
    state['status'] = 'PHYSICAL_CLOSED_LOOP_EXECUTED'
    state['upright_entire_rollout_height_proxy'] = state['minimum_root_height_m'] >= .4
    state['scope'] = 'Real MuJoCo PD control and state-feedback replanning; height threshold is a diagnostic proxy, not paper success metric.'
except Exception as exc:
    code = 1; state.update(status='FAILED', error=str(exc), traceback=traceback.format_exc())
state.update(exit_code=code, elapsed_seconds=time.monotonic() - start,
    ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
if recorded_prefixes:
    np.savez_compressed(a.output / 'planner_prefixes.npz', prefix=np.stack(recorded_prefixes),
        anchor_sim_seconds=np.asarray(prefix_times), fps=np.asarray(60.))
(a.output / 'receipt.json').write_text(json.dumps(state, indent=2) + '\n')
print(json.dumps(state), flush=True)
sys.exit(code)
