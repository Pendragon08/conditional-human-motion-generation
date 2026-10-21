"""Official full schedule with memory recomputation, local evidence and safe saves.

Run with torchrun (2 ranks). --probe-updates creates a separate short execution
with the declared full schedule settings but stops before full training completes.
Vendor sources and optimization loss remain unchanged. Checkpoint saves wait for
disk room rather than deleting any unverified artifact.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import torch
from torch.utils.checkpoint import checkpoint

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'upstream'))
p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, default=BASE / 'configs/full_training_v1.json')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--probe-updates', type=int, default=0)
p.add_argument('--resume', default='')
a = p.parse_args()
c = json.loads(a.config.read_text())
rank = int(os.getenv('RANK', 0)); local_rank = int(os.getenv('LOCAL_RANK', 0))
world = int(os.getenv('WORLD_SIZE', 1))
assert world == c['world_size']
torch.set_num_threads(c['cpu_threads_per_rank'])
torch.cuda.set_device(local_rank)
torch.cuda.set_per_process_memory_fraction(c['memory_fraction_per_gpu'], local_rank)
a.output.mkdir(parents=True, exist_ok=True)
state = {'rank': rank, 'world_size': world, 'status': 'STARTING',
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'configuration': c, 'probe_updates': a.probe_updates,
    'full_training_completed': False, 'physics_evaluated': False,
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
started = time.monotonic()
def write_status():
    state.update(elapsed_seconds=time.monotonic() - started,
        max_allocated_bytes=torch.cuda.max_memory_allocated(local_rank),
        max_reserved_bytes=torch.cuda.max_memory_reserved(local_rank))
    dest = a.output / f'status_rank{rank}.json'; tmp = dest.with_suffix('.json.partial')
    tmp.write_text(json.dumps(state, indent=2) + '\n'); os.replace(tmp, dest)
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
code = 0
try:
    from reactivebfm.model.motion_planner.architectures.dit import DiTBlock
    block_forward = DiTBlock.forward
    def memory_forward(self, *args, **kwargs):
        if c['activation_checkpointing'] and self.training and torch.is_grad_enabled():
            return checkpoint(block_forward, self, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
        return block_forward(self, *args, **kwargs)
    DiTBlock.forward = memory_forward
    from reactivebfm.train import run_utils
    from reactivebfm.train.logging_platforms import NoPlatform
    from reactivebfm.train.scheduled_forcing_loop import ScheduledForcingLoop
    from reactivebfm.train import scheduled_forcing_loop as loop_module
    # Disable the per-batch terminal progress bar; numeric logs remain unchanged.
    loop_module.tqdm = lambda iterable, **kwargs: iterable
    class LocalPlatform(NoPlatform):
        def report_scalars(self, values, iteration, group_name=None):
            if rank: return
            row = {'official_step': iteration, 'optimizer_updates': state['optimizer_updates'],
                'elapsed_seconds': time.monotonic() - started,
                **{k: float(v) for k, v in values.items()}}
            with (a.output / 'metrics.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
    run_utils.TRAIN_PLATFORMS['NoPlatform'] = LocalPlatform
    init_original = ScheduledForcingLoop.__init__
    run_original = ScheduledForcingLoop.run_loop
    save_original = ScheduledForcingLoop.save
    avg_original = ScheduledForcingLoop.update_average_model
    def initialize(self, *args, **kwargs):
        init_original(self, *args, **kwargs)
        state.update(status='TRAINING', batch_size_global=self.batch_size_global,
            trainable_parameters=sum(p.numel() for p in self.model.parameters_wo_clip()),
            dataset_component_lengths=self.data.dataset.component_lengths,
            use_bf16=self.use_bf16, optimizer_updates=max(
                [int(v['step']) for v in self.opt.state.values() if 'step' in v] or [0]),
            official_step_at_start=self.total_step())
        self._last_published_updates = -1
        def after_step(opt, args, kwargs):
            state['optimizer_updates'] += 1
        self.opt.register_step_post_hook(after_step)
        if a.probe_updates:
            self.num_steps = self.total_step() + a.probe_updates
        write_status()
    def save(self):
        if a.probe_updates: return
        # Every rank must enter the guard: the official loop has a barrier after save.
        required = c['minimum_free_output_bytes'] + c['estimated_checkpoint_pair_bytes']
        while shutil.disk_usage(a.output).free < required:
            state['status'] = 'WAITING_FOR_VERIFIED_BACKUP_SPACE'; write_status(); time.sleep(30)
        state['status'] = 'SAVING'; write_status()
        save_original(self)
        model = Path(self.save_dir) / self.ckpt_file_name()
        opt = Path(self.save_dir) / f'opt{self.total_step():09d}.pt'
        rows = [{'file': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in [model, opt]]
        manifest = {'official_step': self.total_step(), 'optimizer_updates': state['optimizer_updates'],
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': rows,
            'configuration_sha256': hashlib.sha256(a.config.read_bytes()).hexdigest()}
        dest = a.output / f'checkpoint_ready_{self.total_step():09d}.json'
        tmp = dest.with_suffix('.json.partial'); tmp.write_text(json.dumps(manifest, indent=2) + '\n'); os.replace(tmp, dest)
        self._last_published_updates = state['optimizer_updates']
        state.update(status='TRAINING', latest_checkpoint_manifest=str(dest)); write_status()
    def average(self):
        avg_original(self)
        # Early recovery checkpoint, after the normal optimizer and EMA update.
        if not a.probe_updates and state['optimizer_updates'] == c['first_checkpoint_updates']:
            from reactivebfm.utils.runtime import distributed as dist
            dist.barrier()
            if rank == 0: self.save()
            dist.barrier()
        if state['optimizer_updates'] % 10 == 0:
            state['official_step'] = self.total_step(); write_status()
    def run(self):
        run_original(self)
        if not a.probe_updates:
            from reactivebfm.utils.runtime import distributed as dist
            dist.barrier()
            if rank == 0 and self._last_published_updates != state['optimizer_updates']: self.save()
            dist.barrier()
        state.update(status='PROBE_COMPLETED' if a.probe_updates else 'FULL_SCHEDULE_COMPLETED',
            full_training_completed=not a.probe_updates and state['optimizer_updates'] >= c['num_steps'],
            final_official_step=self.total_step())
    ScheduledForcingLoop.__init__ = initialize
    ScheduledForcingLoop.save = save
    ScheduledForcingLoop.update_average_model = average
    ScheduledForcingLoop.run_loop = run
    from reactivebfm.train.train_planner_scheduled_forcing import main
    env_spec = json.loads((BASE / 'configs/env_spec.json').read_text())
    sys.argv = ['official_full_training', '--save_dir', str(a.output / 'checkpoints'),
        '--dataset', c['dataset'], '--data_dir', c['runtime_root'] + '/data', '--model_type', 'flow',
        '--text_encoder', env_spec['weight_dirs']['text_encoder']['path'],
        '--num_steps', str(c['num_steps']), '--num_warmup_steps', str(c['num_warmup_steps']),
        '--batch_size_local', str(c['batch_size_local']), '--seed', str(c['seed']),
        '--num_workers', str(c['num_workers']), '--train_platform_type', 'NoPlatform',
        '--save_interval', str(c['save_interval']), '--log_interval', str(c['log_interval']),
        '--cross_prob', str(c['cross_prob']), '--lr', str(c['learning_rate'])]
    if c['use_bf16']: sys.argv += ['--use_bf16']
    if not c['use_ema']: sys.argv += ['--no-use_ema']
    if a.resume: sys.argv += ['--resume_checkpoint', a.resume]
    state['official_arguments'] = sys.argv[1:]
    main()
except Exception as exc:
    code = 1; state.update(status='FAILED', error=str(exc), traceback=traceback.format_exc())
finally:
    state['exit_code'] = code
    state['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_status(); print(json.dumps(state), flush=True)
sys.exit(code)
