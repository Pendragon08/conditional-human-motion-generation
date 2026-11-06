"""Verified poses initialize optimization; every output is refit to its own raw motion."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(spec):
    assert sha(spec['path']) == spec['sha256'], spec['path']
    return json.loads(Path(spec['path']).read_text())


def aligned(row, contact, frames, names):
    path = row.get('path', row.get('source'))
    expected = row.get('sha256', row.get('source_sha256'))
    assert sha(path) == expected
    old_contact = round(row['grasp_time_s'] * 20)
    clock = np.arange(frames)
    # Match the shared contact/lift/tilt clock, with the unchanged end frame.
    offsets = [0, 40] if frames == 120 else [0, 30, 50, 70, 90, 100, 120]
    knots = [0] + [contact + x for x in offsets] + [frames - 1]
    old_knots = [0] + [old_contact + x for x in offsets] + [frames - 1]
    assert np.min(np.diff(knots)) > 0 and np.min(np.diff(old_knots)) > 0
    phase = np.interp(clock, knots, old_knots)
    with np.load(path, allow_pickle=False) as z:
        assert z['joint_names'].tolist() == names and z['q'].shape == (frames, 29)
        assert float(z['fps']) == 20.
        q = np.column_stack([np.interp(phase, clock, x) for x in z['q'].T])
        pos = np.column_stack([np.interp(phase, clock, x) for x in z['root_pos'].T])
        rot = Slerp(clock, Rotation.from_quat(z['root_quat'][:, [1, 2, 3, 0]]))(phase)
    return q, np.c_[rot.as_quat()[:, [3, 0, 1, 2]], pos]


def prepare(plan, row, solver, sample):
    task = row['task']
    if task == 'pick':
        bank = read(plan['sources']['legacy_pick_bank'])
        center = np.asarray(sample['metadata']['task_object_position'])
        seeds = sorted(bank['references'], key=lambda r: np.linalg.norm(
            np.asarray(r['task_object_position']) - center))[:12]
    elif task == 'pour':
        receipt = read(plan['pour_initialization_receipt'])
        assert receipt['status'] == 'existing_sample_validation_passed'
        seeds = receipt['records']
    elif task == 'bimanual_box_lift':
        receipt = read(plan['box_initialization_receipt'])
        assert receipt['native_pass'] == 1
        seeds = receipt['records']
        assert len(seeds) == 1 and seeds[0]['candidate_id'] == 20200004
        assert int(sample['metadata']['grasp_frame']) == 40 and sample['frames'] == 180
        with np.load(seeds[0]['source'], allow_pickle=False) as z:
            metadata = json.loads(z['metadata_utf8'].tobytes())
        assert metadata['task_plan']['phases'] == sample['metadata']['task_plan']['phases']
        assert metadata['scene_sha256'] == sample['metadata']['scene_sha256']
    else:
        return None
    compared = []
    for seed in seeds:
        proof_path = seed.get('postopt_report', seed.get('native_proof'))
        proof_sha = seed.get('postopt_report_sha256', seed.get('native_proof_sha256'))
        assert sha(proof_path) == proof_sha
        proof = json.loads(Path(proof_path).read_text())
        assert proof['eligible_for_calibrated_scene_reference'] and all(proof['native_checks'].values())
        if task == 'bimanual_box_lift':
            # This task has the exact same 180-frame phase clock. Preserve the
            # successful diagnostic start without an unnecessary Slerp roundtrip.
            with np.load(seed['source'], allow_pickle=False) as z:
                assert z['joint_names'].tolist() == solver.names
                assert z['q'].shape == (sample['frames'], 29) and float(z['fps']) == 20.
                q = z['q'].copy()
                root = np.c_[z['root_quat'], z['root_pos']]
        else:
            q, root = aligned(seed, int(sample['metadata']['grasp_frame']), sample['frames'], solver.names)
        pts = solver.fk.forward(root[:, 4:], root[:, [1, 2, 3, 0]], q, solver.names, links=solver.all_links)[:, solver.kp]
        error = float(np.sqrt(np.mean(np.sum((pts - sample['decoded_world']) ** 2, axis=-1))))
        compared.append((error, seed.get('sha256', seed.get('source_sha256')), seed, q, root))
    error, state_sha, seed, q, root = min(compared, key=lambda x: x[:2])
    q = np.clip(q, solver.low + 1e-6, solver.high - 1e-6).astype(np.float32)
    root = root.astype(np.float32)
    return q, root, {
        'scope': 'optimizer initialization only; original raw, conditions, losses and paper filters unchanged',
        'seed_path': seed.get('path', seed.get('source')), 'seed_sha256': state_sha,
        'phase_aligned_seed_raw_rmse_m': error,
        'matched_contact_frame': int(sample['metadata']['grasp_frame']),
        'initial_state_only': True, 'raw_or_target_modified': False,
        'candidates_considered': len(compared), 'initializer_source_sha256': sha(__file__)}


def process(plan, row, solver, solver_module, directory):
    sample = solver_module.load_sample(row['source_sample'], row['source_sample_sha256'])
    seed = prepare(plan, row, solver, sample)
    if seed is None:
        return solver.process(row['source_sample'], directory, row['source_sample_sha256'])
    q, root, provenance = seed
    original_initial_state = solver.initial_state
    try:
        solver.initial_state = lambda decoded: (q.copy(), root.copy())
        report = solver.process(row['source_sample'], directory, row['source_sample_sha256'])
    finally:
        solver.initial_state = original_initial_state
    report['initialization'] = provenance
    if report.get('output'):
        with np.load(report['output'], allow_pickle=False) as z:
            report['initialization']['output_vs_seed_max_joint_rad'] = float(np.max(np.abs(z['q'] - q)))
            report['initialization']['output_vs_seed_max_root_m'] = float(np.linalg.norm(z['root_pos'] - root[:, 4:], axis=-1).max())
    path = Path(directory) / 'report.json'
    temporary = path.with_suffix('.init.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(path)
    return report
