"""Read-only CPU audit of the safe83703 emergency; performs zero updates."""
import argparse
import hashlib
import json
import os
import pathlib
from ldm100k_verified_input import ROOT, owned, read, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert os.getuid() == 1013 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
    out = pathlib.Path(a.output)
    out.mkdir(mode=0o700, exist_ok=False)
    sp = owned(ROOT / 'configs/tmr69d-evaluator-train-spec.json')
    assert sha(sp) == '2196a765dbaa61ba7265b9d798f99fe5c094a502fd34bd47869f0fe141360b13'
    spec = read(sp)
    for name, value in spec['bound_sources'].items():
        assert sha(owned(ROOT / name)) == value
    rp = owned(ROOT / 'evaluator_tmr_20261029/tmr69d_evaluator500_train_v1/receipt.json')
    assert sha(rp) == 'd5a43ad794f6aaa3014f302a425a56b33332692ee5c3b4d3d4618295bee3d361'
    rec = read(rp)
    item = rec['emergency_checkpoint']
    ck = owned(item['path'], pathlib.Path('/home/ouyanghao'))
    assert ck == pathlib.Path('/home/ouyanghao/textop-tmr69d-evaluator-20261029/checkpoints/tmr_emergency.pth')
    before = ck.stat()
    assert before.st_nlink == 1 and before.st_size == 192951813
    assert sha(ck) == item['sha256'] == '8169a44260d0e238505e3a42c8cc859b214482e102d35241e054931d470f5d05'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    state = torch.load(ck, map_location='cpu')
    assert state['optimizer_updates'] == 83703 and state['completed_epochs'] == 82
    assert state['pending_validation'] is False
    assert state['request_path'] == str(sp) and state['request_sha256'] == sha(sp)
    assert state['runtime_binding_sha256'] == spec['runtime_binding_sha256']
    assert state['input_signature_sha256'] == spec['accepted_input_signature_sha256']
    sampler = state['sampler']
    assert sampler['count'] == 32462 and sampler['epoch'] == 82 and sampler['cursor'] == 15136 and sampler['steps'] == 83703
    assert sampler['order'].dtype == torch.int64 and sorted(sampler['order'].tolist()) == list(range(32462))
    generator = torch.Generator(device='cpu')
    generator.set_state(sampler['generator'])
    assert torch.equal(generator.get_state(), sampler['generator'])
    assert set(state['all_rng']) == {'python', 'numpy', 'torch', 'cuda'}
    assert len(state['all_rng']['cuda']) == 1
    for t in [state['all_rng']['torch']] + state['all_rng']['cuda']:
        assert t.device.type == 'cpu' and t.dtype == torch.uint8 and t.ndim == 1 and t.numel() > 0
    assert state['optimizer']['param_groups'] and state['optimizer']['state']
    for group in state['optimizer']['param_groups']:
        assert group['lr'] == .0001 and group['weight_decay'] == .01
    for value in state['optimizer']['state'].values():
        assert int(value['step']) == 83703
        assert set(value) == {'step', 'exp_avg', 'exp_avg_sq'}
        assert value['exp_avg'].shape == value['exp_avg_sq'].shape
    tensors = []
    def visit(value):
        if torch.is_tensor(value):
            assert value.device.type == 'cpu' and torch.isfinite(value).all()
            tensors.append(value)
        elif isinstance(value, dict):
            for x in value.values():
                visit(x)
        elif isinstance(value, (list, tuple)):
            for x in value:
                visit(x)
        elif isinstance(value, np.ndarray):
            assert np.isfinite(value).all()
    visit(state)
    assert sum(v.numel() for v in state['model'].values()) == 16031301
    after = ck.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    assert sha(ck) == item['sha256'] and not torch.cuda.is_initialized()
    result = dict(status='TMR100_CAPACITY_EMERGENCY_CPU_METADATA_FINITE_VERIFIED',
        run=out.name, source_sha256=sha(__file__),
        original_failure_receipt_sha256=sha(rp), selected_checkpoint=item,
        optimizer_updates=83703, completed_epochs=82, pending_validation=False,
        sampler_epoch=82, sampler_cursor=15136, sampler_steps=83703,
        finite_cpu_tensors=len(tensors), optimizer_step_min=83703, optimizer_step_max=83703,
        original_fullstate_unchanged=True, original_dirty_step83704_discarded=True,
        actual_optimizer_updates=0, new_cuda_contexts=0, model_constructions=0,
        saved_cuda_rng_restored=False, actual_source_arrays_loaded=0,
        saved_rng_bytes=dict(torch=state['all_rng']['torch'].numel(), cuda=[x.numel() for x in state['all_rng']['cuda']]),
        full_production_restore_and_nextbatch_witness_still_required=True, output_writers_closed=True)
    with (out / 'receipt.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
