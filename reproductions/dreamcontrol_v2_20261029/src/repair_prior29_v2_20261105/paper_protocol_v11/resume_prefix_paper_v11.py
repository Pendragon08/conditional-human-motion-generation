"""Verify preserved prefixes and finish only their pending native replay."""
from collections import Counter
from pathlib import Path
import os


def resume(p, plan_path, plan_sha, plan, cfg, output, checkpoint_sha, native_spec):
    data = p.read(p.bound(plan['resume_manifest']))
    rows = data['records']
    cohorts = data['cohorts']
    for row in rows:
        verify(p, row, checkpoint_sha)
    for entry in data['pending_native']:
        task, number = entry['task'], entry['cohort_number']
        directory = output / task / f'cohort_{number:04d}'
        directory.mkdir(parents=True, exist_ok=False)
        batch = p.read(p.bound(entry['raw_bindings']))
        for row in batch:
            p.verify_hash(row['source_sample'], row['source_sample_sha256'])
            proof = entry['postopt_reports'][str(row['candidate_id'])]
            p.bound(proof)
        staged = p.stage_native(plan, task, Path(entry['postopt_dir']), directory / 'native_input', batch)
        native = directory / 'native'
        if staged:
            script = p.bound(plan['sources']['native_supported'])
            env = p.runtime_env(native_spec['env'], remove_visibility=True)
            env['PYTHONPATH'] = plan['native_import_dir'] + os.pathsep + str(script.parent) + os.pathsep + env.get('PYTHONPATH', '')
            argv = [native_spec['python'], '-B', '-u', str(script), '--headless', '--device', plan['native_device'],
                    '--output-dir', str(native), '--task', task]
            for _, candidate in staged:
                argv += ['--candidate-dir', str(candidate)]
            p.phase(plan_path, plan_sha, plan, cfg, output, directory, 'native', argv, env, native_spec['cwd'],
                    (native / 'report.json', 'native_task_replay_completed'))
            p.native_proofs(task, staged, native)
        result = {'status': 'cohort_review_completed', 'task': task, 'cohort': number, 'records': batch,
                  'generated': len(batch), 'kinematic_pass': len(staged), 'native_pass': sum(x['admitted'] for x in batch),
                  'reused_raw_and_postopt': True, 'original_cohort': entry['original_cohort'], 'ppo_started': False}
        rp = directory / 'result.json'
        p.save(rp, result, exclusive=True)
        cohorts.append({'path': str(rp), 'sha256': p.sha(rp), 'task': task, 'generated': len(batch),
                        'kinematic_pass': len(staged), 'native_pass': result['native_pass'], 'allocated_bytes': p.disk_measure(directory)})
        rows.extend(batch)
    offsets, numbers = Counter(), Counter(data['next_cohort_numbers'])
    pilots = {}
    for task in p.TASKS:
        task_rows = sorted((r for r in rows if r['task'] == task), key=lambda r: r['candidate_id'])
        ids = [r['candidate_id'] for r in task_rows]
        start = 20000000 + 100000 * p.TASKS.index(task)
        p.require(ids == list(range(start, start + len(ids))), 'Resume candidate prefix is not contiguous')
        offsets[task] = len(ids)
        pilots[task] = {'passed': any(r['admitted'] for r in task_rows), 'attempts': [
            {'candidate_id': r['candidate_id'], 'passed': r['admitted'], 'initial_noise_seed': r['initial_noise_seed'],
             'rejection_reason': r.get('rejection_reason'), 'preserved_prefix': True} for r in task_rows]}
    p.assert_no_duplicates(rows)
    p.save(output / 'resume_receipt.json', {'status': 'existing_prefix_verified', 'new_samples': 0,
           'candidate_counts': dict(offsets), 'verified_admitted_counts': p.reference_counts(rows),
           'resume_manifest_sha256': plan['resume_manifest']['sha256']}, exclusive=True)
    return rows, cohorts, offsets, numbers, pilots


def verify(p, row, checkpoint_sha):
    p.require(row['checkpoint_sha256'] == checkpoint_sha, 'Prefix checkpoint mismatch')
    for key in ('source_sample', 'kinematic_report'):
        p.verify_hash(row[key], row[key + '_sha256'])
    report = p.read(Path(row['kinematic_report']))
    p.require(report['candidate_id'] == row['candidate_id'] and report['source_sample_sha256'] == row['source_sample_sha256'],
              'Preserved postoptimizer binding mismatch')
    if row['admitted']:
        for key in ('source', 'native_proof'):
            p.verify_hash(row[key], row[key + '_sha256'])
        proof = p.read(Path(row['native_proof']))
        p.require(proof['eligible_for_calibrated_scene_reference'] and all(proof['native_checks'].values()), 'Native prefix rejected')
        p.require(report['passed_requested_kinematic_checks'] and all(report['checks'].values()), 'Kinematic prefix rejected')
