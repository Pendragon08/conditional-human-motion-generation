"""Read actual closed filtered VAE200k provenance before a new LDM handoff.

This module uses only the standard library. It neither loads a checkpoint nor
creates a model; the separately supervised CPU exporter checks tensor contents.
"""
import hashlib
import json
import os
import pathlib

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_bytes())


def owned(path, root=ROOT):
    p = pathlib.Path(path)
    assert p.is_absolute() and p.resolve() == p and (p == root or root in p.parents)
    here = p
    while True:
        assert here.stat().st_uid == os.getuid() == 1013
        if here == root:
            break
        here = here.parent
    return p


def closed_job(prefix, receipt_path, receipt_sha256, supervisor_sha256):
    rp = owned(receipt_path)
    assert sha(rp) == receipt_sha256
    paths = {k: owned(ROOT / 'receipts' / (prefix + suffix)) for k, suffix in
             [('exit', '_exit.json'), ('process', '_process_exit.json'),
              ('outer', '_outer_exit.json')]}
    ex, pr, outer = [read(paths[k]) for k in ['exit', 'process', 'outer']]
    assert outer['actual_returncode'] == 0
    for e in [ex, pr]:
        assert e['supervision_passed'] and e['child_reaped'] and e['child_exit_code'] == 0
        assert not e['timed_out'] and e['supervisor_error'] is None
        assert e['sample_errors'] == [] and not e.get('cleanup_error')
    key = 'evaluation_receipt_sha256' if 'evaluation_receipt_sha256' in ex else 'receipt_sha256'
    assert ex[key] == receipt_sha256 and ex['process_exit_sha256'] == sha(paths['process'])
    if 'source_sha256' in ex:
        assert ex['source_sha256'] == supervisor_sha256
    resources = owned(ROOT / 'logs' / (prefix + '_resources.jsonl'))
    samples = [json.loads(x) for x in resources.read_text().splitlines()]
    rk = 'resources_sha256' if 'resources_sha256' in ex else 'resource_sha256'
    assert sha(resources) == ex[rk]
    assert len(samples) == ex['samples_written'] == pr['samples_written']
    assert samples[-1]['phase'] == 'after_exit'
    return {k + '_sha256': sha(v) for k, v in paths.items()}


def completed_vae(binding):
    """Require the actual200k run and exact current data/source binding."""
    assert isinstance(binding, dict) and binding['run'].isascii() and binding['run'].isidentifier()
    run = binding['run']
    sp = owned(ROOT / 'configs/filtered15000-vae-spec.json')
    s = read(sp)
    assert s['schema'] == 'textop.filtered15000_vae.v1' and s['train_authorized']
    assert s['vae']['stages'] == [100000, 50000, 50000]
    assert sha(ROOT / 'scripts/run_filtered15000_vae.py') == s['entrypoint_sha256']
    assert sha(ROOT / 'scripts/supervise_filtered15000_vae.py') == s['supervisor_sha256']
    for n, h in s['bound_sources'].items():
        assert sha(ROOT / n) == h, n
    rp = owned(ROOT / 'filtered15000_20261027/vae_runs' / run / 'receipt.json')
    assert str(rp) == binding['receipt_path'] and sha(rp) == binding['receipt_sha256']
    r = read(rp)
    assert r['status'] == 'FILTERED15000_VAE_200000_TRAINING_FINISHED'
    assert r['mode'] == 'train' and r['output_writers_closed']
    assert r['spec_sha256'] == sha(sp) and r['source_sha256'] == s['entrypoint_sha256']
    assert r['completed_primitive_attempts'] == 200000
    assert r['optimizer_updates'] + r['skipped_updates'] == 200000
    assert r['vae_quality_gate_passed'], 'Retain real result; investigate reconstruction before LDM'
    for k in ['corpus_receipt_sha256', 'data_acceptance_sha256', 'strict_audit_receipt_sha256']:
        assert r[k] == s[k] and r[k]
    keys = ['vae', 'seed', 'corpus_receipt_sha256', 'data_acceptance_sha256',
            'strict_audit_receipt_sha256', 'bound_sources', 'entrypoint_sha256',
            'checkpoint_dir', 'checkpoint_allowed_root', 'checkpoint_policy']
    signature = hashlib.sha256(json.dumps({k: s[k] for k in keys}, sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()
    assert r['runtime_binding_sha256'] == signature
    assert all(r['latest_file_resume_verification'][k] for k in
               ['model_restored_exactly', 'optimizer_restored_exactly',
                'clock_restored_exactly', 'rng_and_generator_restored_exactly'])
    steps = [x['manager_step'] for x in r['checkpoints']]
    assert steps == [50000, 100000, 150000, 200000]
    ck = r['checkpoints'][-1]
    allowed = pathlib.Path(s['checkpoint_allowed_root'])
    dest = owned(s['checkpoint_dir'], allowed)
    cp = owned(ck['path'], allowed)
    assert cp.parent == dest and cp.name == 'vae_state_200000.pth'
    assert cp.stat().st_size == ck['bytes'] <= s['checkpoint_policy']['checkpoint_upper_bound_bytes']
    proof = closed_job('filtered15000_vae_' + run, rp, sha(rp), s['supervisor_sha256'])
    assert proof['exit_sha256'] == binding['exit_sha256']
    return dict(spec=s, receipt=r, checkpoint=ck, closure=proof,
                receipt_sha256=sha(rp), runtime_binding_sha256=signature)
