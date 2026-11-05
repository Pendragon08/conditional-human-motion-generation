"""Guarded entry into the unchanged native trainer; never substitutes its loop.

Preparation is CPU-only. GPU phases require the actual final2000 receipt.
The two training phases deliberately rebuild replay once and account for it.
"""
import argparse
import hashlib
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

R = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D = R / 'stage3_supermimic29_20261103'
N = D / 'native_universal_20261104'
FINAL = R / 'stage3_supported_references_20261104/mixed2000_final_v1'
TRAIN_SHA = 'b94569605bd3ff96a49d4903bb791bb54bc4695a03f1914c8447c0ec7dd1e6bd'
TASKS = ['pick', 'pour', 'bimanual_box_lift', 'box_push_pull']
ENVS, WARM, EARLY, LAST, BUDGET = 8912, 65, 600, 28748, 72000
HEADER = ('iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,'
          'imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,'
          'term_frac,it_per_s,lr\n')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(p, value):
    t = p.with_suffix('.worker.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def native_collection_steps(iterations):
    # Match the pinned trainer's floating point accumulator, including resets.
    accumulator = 0.0
    result = 0
    for _ in range(iterations):
        accumulator += 1 / .4
        n = int(accumulator)
        accumulator -= n
        result += n
    return result


def budget_plan():
    early = WARM + native_collection_steps(EARLY)
    final = early + WARM + native_collection_steps(LAST - EARLY)
    assert EARLY % 2 == 0 and final == BUDGET
    assert WARM + native_collection_steps(28774) == BUDGET
    return {
        'num_envs': ENVS, 'total_real_vector_steps': final,
        'total_real_environment_transitions': final * ENVS,
        'early_native_iteration': EARLY, 'last_native_iteration': LAST,
        'early_real_vector_steps': early, 'warmup_steps_per_launch': WARM,
        'native_launches': 2, 'replay_rebuild_steps_counted': WARM,
        'world_model_only_native_iterations': 500, 'utd': .4,
        'nominal_world_model_adam_updates': LAST * 10,
        'policy_update_attempts': LAST - 500,
        'nominal_critic_adam_updates': (LAST - 500) * 20,
        'actual_policy_adam_updates': 'read from optimizer; native skips nonfinite gradient steps',
        'buffer_steps': 128, 'policy_batch': 2048, 'horizon': 32,
        'differences_from_upstream_defaults': {
            'num_envs': [4096, ENVS], 'buffer_steps': [1000, 128],
            'pi_batch': [4096, 2048], 'warmup_steps': [64, WARM],
            'native_iterations': [320000, LAST],
            'checkpoint_schedule': 'early600 then resume to28748, native main interval2000',
        },
        'resume_scope': 'models and optimizers only; replay rebuilt, RNG not restored',
        'same_real_budget_single_launch_alternative': 28774,
        'policy_attempts_reduced_by_replay_rebuild': 26,
        'imagined_steps_in_real_budget': False,
    }


def dataset():
    receipt_path = FINAL / 'completion_receipt.json'
    receipt = read(receipt_path)
    assert receipt['status'] == 'four_task2000_reference_dataset_completed'
    assert receipt['reference_count'] == 2000 and receipt['per_task'] == {t: 500 for t in TASKS}
    assert receipt['data_ready_for_native_body_tracker']
    assert receipt['source_fps'] == 20 and receipt['body_dof'] == 29
    manifest = FINAL / 'native_motion_manifest.json'
    assert Path(receipt['manifest']) == manifest and sha(manifest) == receipt['manifest_sha256']
    m = read(manifest)
    assert not m['preview'] and m['reference_count'] == 2000 and len(m['records']) == 2000
    assert m['per_task'] == receipt['per_task'] and len(m['body_joint_order']) == 29
    report_path = FINAL / 'native_loader_check_v1/report.json'
    assert Path(receipt['native_loader_report']) == report_path
    assert sha(report_path) == receipt['native_loader_report_sha256']
    report = read(report_path)
    assert report['manifest_sha256'] == sha(manifest) and report['reference_count'] == 2000
    assert report['all_clips_future_goal_and_boundary_passed']
    assert Path(receipt['native_motion_dir']) == FINAL / 'motion_csv20'
    return receipt, manifest, sha(manifest)


def formal_gate(manifest_sha):
    gate = read(N / 'all_four_reference_gate.json')
    assert gate['training_allowed'] is True and gate['manifest_sha256'] == manifest_sha
    capacity = Path(gate['capacity_report'])
    assert capacity.resolve().is_relative_to(N.resolve())
    assert sha(capacity) == gate['capacity_report_sha256']
    c = read(capacity)
    assert c['status'] == 'native2000_capacity_passed' and c['manifest_sha256'] == manifest_sha
    assert c['reference_clips'] == 2000 and c['num_envs'] == ENVS
    assert c['real_environment_transitions'] == WARM * ENVS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'capacity', 'early', 'main'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--early-output', type=Path)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(N.resolve())
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    state = {'status': 'initializing', 'pid': os.getpid(), 'phase': args.phase,
             'formal_training_started': False, 'task_success_claimed': False,
             'worker_sha256': sha(__file__), 'budget': budget_plan()}

    def save():
        state['elapsed_seconds'] = time.monotonic() - started
        atomic(args.output / 'status.json', state)

    save()
    try:
        document = read(N / 'environment_spec.json')
        spec = document['spec']
        assert document['spec_hash'] == '32a40546'
        assert sha(spec['full_native_train']) == spec['full_native_train_sha256'] == TRAIN_SHA
        state.update(environment_spec_hash=document['spec_hash'], full_native_train_sha256=TRAIN_SHA)
        if args.phase == 'prepare':
            assert 'torch' not in sys.modules
            try:
                dataset()
                state['final2000_data_gate'] = 'receipt_verified'
            except FileNotFoundError:
                state['final2000_data_gate'] = 'waiting_actual_final_receipt'
            state.update(status='native2000_entry_prepared_not_runtime_validated', GPU_work=False,
                         training_allowed=False,
                         pending=['actual_final2000_receipt', 'new2000_library8912_capacity',
                                  'formal_guard_and_disk_budget', 'full2000_tracking_evaluation'])
            save()
            return
        receipt, manifest, manifest_sha = dataset()  # Must pass before importing torch.
        state.update(manifest=str(manifest), manifest_sha256=manifest_sha,
                     completion_receipt_sha256=sha(FINAL / 'completion_receipt.json'))
        if args.phase != 'capacity':
            formal_gate(manifest_sha)
        sys.path[:0] = spec['module_paths']
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(20261104)
        # Keep at least2GiB of the12GiB job cap for simulator/native allocations.
        fraction = min(.22, (10 * 1024 ** 3) / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        early_state = None
        resume = None
        initial_sim = 0
        start_iteration = 0
        if args.phase == 'main':
            assert args.early_output and args.early_output.resolve().is_relative_to(N.resolve())
            early_state = read(args.early_output / 'status.json')
            assert early_state['status'] == 'native2000_early_checkpoint_verified'
            assert early_state['manifest_sha256'] == manifest_sha
            resume = args.early_output / 'logs/last.pt'
            assert sha(resume) == early_state['checkpoint']['sha256']
            checkpoint = torch.load(resume, weights_only=False, map_location='cpu')
            assert checkpoint['iteration'] == EARLY
            assert checkpoint['sim_steps'] == state['budget']['early_real_vector_steps'] * ENVS
            start_iteration, initial_sim = EARLY, checkpoint['sim_steps']
            del checkpoint
            # Upstream intentionally writes its header only when start_it==0.
            (args.output / 'logs').mkdir()
            (args.output / 'logs/train_log.csv').write_text(HEADER)
        iterations = {'capacity': 0, 'early': EARLY, 'main': LAST}[args.phase]
        command = [spec['full_native_train'], '--visualizer', 'none', '--device', 'cuda:0',
                   '--seed', '20261104', '--motion_dir', receipt['native_motion_dir'],
                   '--log_dir', str(args.output / 'logs'), '--amp', 'bf16',
                   '--num_envs', str(ENVS), '--iterations', str(iterations),
                   '--utd', '.4', '--buffer_steps', '128', '--warmup_steps', str(WARM),
                   '--wm_only_iters', '500', '--wm_batches', '10', '--wm_batch', '2048',
                   '--pi_batch', '2048', '--horizon', '32', '--log_every', '50',
                   '--ckpt_every', '600' if args.phase == 'early' else '2000']
        if resume:
            command += ['--resume', str(resume)]
        (args.output / 'command.json').open('x').write(json.dumps(command, indent=2))
        state.update(status='native2000_trainer_running', command=command,
                     formal_process_started=args.phase != 'capacity',
                     formal_training_started=args.phase == 'main',
                     torch_allocator_fraction=fraction,
                     native_output_scope='29 body reference tracking; hand/object execution not certified')
        save()
        sys.argv = command
        g = runpy.run_path(spec['full_native_train'], run_name='__main__')
        torch.cuda.synchronize()
        assert g['motion'].clip_len.numel() == 2000
        expected_sim = initial_sim + (WARM + native_collection_steps(iterations - start_iteration)) * ENVS
        assert g['sim_steps'] == expected_sim
        state.update(num_envs=ENVS, reference_clips=2000,
                     real_environment_transitions=expected_sim,
                     real_vector_steps=expected_sim // ENVS,
                     native_motion_frames=g['motion'].num_frames,
                     torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if args.phase == 'capacity':
            assert g['buffer'].size == WARM
            assert all(torch.isfinite(v[:WARM]).all() for v in g['buffer'].data.values())
            state.update(status='native2000_capacity_passed', optimizer_updates=0)
            save()
            return
        def opt_steps(opt):
            values = [int(s['step']) for s in opt.state.values() if 'step' in s]
            assert values and min(values) == max(values)
            return max(values)
        updates = {k: opt_steps(g[k]) for k in ['opt_wm', 'opt_pi', 'opt_v']}
        attempts = iterations - 500
        assert updates['opt_wm'] == iterations * 10 and updates['opt_v'] == attempts * 20
        assert 0 < updates['opt_pi'] <= attempts
        checkpoint_path = args.output / 'logs/last.pt'
        stored = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert stored['iteration'] == iterations and stored['sim_steps'] == expected_sim
        for key in ['wm', 'pi', 'critic']:
            assert all(torch.equal(v.cpu(), stored[key][k]) and bool(torch.isfinite(v).all())
                       for k, v in g[key].state_dict().items())
        def same(left, right):
            if isinstance(left, torch.Tensor):
                return (isinstance(right, torch.Tensor)
                        and bool(torch.isfinite(left).all())
                        and torch.equal(left.cpu(), right.cpu()))
            if isinstance(left, dict):
                return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
            if isinstance(left, (list, tuple)):
                return len(left) == len(right) and all(same(x, y) for x, y in zip(left, right))
            return left == right
        assert all(same(g[k].state_dict(), stored[k]) for k in updates)
        import torch._inductor.metrics as metrics
        assert metrics.generated_kernel_count > 0
        state.update(status='native2000_early_checkpoint_verified' if args.phase == 'early'
                     else 'native2000_training_budget_completed_evaluation_pending',
                     formal_training_started=True,
                     native_iteration=iterations, optimizer_updates=updates,
                     policy_update_attempts=attempts,
                     policy_attempts_without_adam_step=attempts - updates['opt_pi'],
                     generated_inductor_kernels=metrics.generated_kernel_count,
                     checkpoint={'path': str(checkpoint_path), 'sha256': sha(checkpoint_path),
                                 'bytes': checkpoint_path.stat().st_size,
                                 'model_and_optimizer_serialization_verified': True,
                                 'replay_and_rng_saved': False})
        save()
    except BaseException as error:
        state.update(status='failed_preserved', error=repr(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
