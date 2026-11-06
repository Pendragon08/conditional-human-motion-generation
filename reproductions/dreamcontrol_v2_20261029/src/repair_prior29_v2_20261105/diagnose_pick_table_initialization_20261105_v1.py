"""One candidate: test whether staged collision weighting avoids the desk-edge minimum."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE = ROOT / 'repair_prior29_v2_20261105'
OUT = STAGE / 'pick_table_initialization_diagnostic_v1'
PREVIOUS_PLAN = STAGE / 'generation_pipeline_v5_launch/config.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def modify(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def setup():
    assert sha(PREVIOUS_PLAN) == '5d2cfca7b6da4cc9f6b47fb36c63db05ddb6cf8f26244a8189f2486b7a81403a'
    previous = json.loads(PREVIOUS_PLAN.read_text())
    assert json.loads((STAGE / 'new_references_v5/pilot_receipt.json').read_text())['status'] == 'pilot_quality_budget_exhausted'
    if OUT.exists():
        print((OUT / 'status.json').read_text() if (OUT / 'status.json').exists() else str(OUT))
        return
    OUT.mkdir()
    code = OUT / 'source'
    code.mkdir()
    for binding in previous['sources'].values():
        path = Path(binding['path'])
        if path.parent.name != 'generation_retry_v5':
            continue
        assert sha(path) == binding['sha256']
        body = path.read_text()
        if path.name == 'postopt_v2.py':
            body = modify(body, 'ground_z, penalty):\n            body = pose',
                          'ground_z, penalty, table_scale):\n            body = pose')
            body = modify(body, 'ground[None], foot, table[None]',
                          'ground[None], foot, table[None] * table_scale')
            body = modify(body, 'ground_z, stage, penalty):',
                          'ground_z, stage, penalty, table_scale=1.):')
            body = modify(body, 'jnp.full((frames,), 4.)),',
                          'jnp.full((frames,), 4.), jnp.full((frames,), table_scale)),')
            body = modify(body,
                '            qa, ra = self.solve(target, goals, orient, mask, jnp.asarray(initial_q), jnp.asarray(initial_root),',
                '''            # A collision-unconstrained seed is never eligible for output.
            seed_q, seed_root = self.solve(target, goals, orient, mask, jnp.asarray(initial_q),
                                          jnp.asarray(initial_root), target, jnp.asarray(initial_q),
                                          jnp.asarray(initial_root), ground, stage="A", penalty=0.,
                                          table_scale=0.)
            with (output / "unconstrained_seed_not_admitted.npz").open("xb") as f:
                np.savez_compressed(f, q=np.asarray(seed_q), root=np.asarray(seed_root))
            report["initialization_only"] = "relax table residual for seed, then restore it for A and B"
            qa, ra = self.solve(target, goals, orient, mask, seed_q, seed_root,''')
        target = code / path.name
        compile(body, str(target), 'exec')
        target.write_text(body)
        binding.update(path=str(target), sha256=sha(target))
    previous['output_dir'] = str(OUT)
    plan = OUT / 'plan.json'
    plan.write_text(json.dumps(previous, indent=2))
    controller = OUT / 'diagnostic_controller.py'
    controller.write_bytes(Path(__file__).read_bytes()) if Path(__file__).is_file() else controller.write_text(sys.stdin.read())
    assert controller.stat().st_size > 0
    env = {**os.environ, **previous['postopt_env'], 'PYTHONPATH': str(code) + os.pathsep + previous['postopt_env'].get('PYTHONPATH',''),
           'PYTHONDONTWRITEBYTECODE': '1'}
    with (OUT / 'controller.log').open('xb') as log:
        proc = subprocess.Popen([previous['postopt_python'], '-B', '-u', str(controller), '--worker'],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=OUT, env=env)
    (OUT / 'launch.json').write_text(json.dumps({'pid': proc.pid, 'plan_sha256': sha(plan)}, indent=2))
    print((OUT / 'launch.json').read_text())


def work():
    import pipeline_controller_v2 as p

    plan_path = OUT / 'plan.json'
    plan = json.loads(plan_path.read_text())
    cfg = p.load_config(p.bound(plan['training_config']))
    original = json.loads((STAGE / 'new_references_v5/pick/cohort_0007/postopt/candidate_10000007/report.json').read_text())
    raw = Path(original['source_sample'])
    assert original['candidate_id'] == 10000007 and sha(raw) == original['source_sample_sha256']
    script = p.bound(plan['sources']['postopt'])
    result_path = OUT / 'candidate_10000007'
    p.save(OUT / 'status.json', {'status':'starting_single_saved_candidate','candidate_id':10000007})
    command = [plan['postopt_python'], '-B', '-u', str(script), '--sample', str(raw),
               '--sample-sha256', original['source_sample_sha256'], '--output-dir', str(result_path),
               '--urdf', plan['urdf']['path'], '--fk-module', plan['fk']['path']]
    try:
        p.phase(plan_path, sha(plan_path), plan, cfg, OUT, OUT, 'postopt', command, os.environ.copy(), str(OUT))
    except ValueError:
        if not (result_path / 'report.json').is_file():
            raise
    finally:
        report_path = result_path / 'report.json'
        result = json.loads(report_path.read_text()) if report_path.exists() else {'status':'diagnostic_failed_preserved'}
        p.save(OUT / 'status.json', {'status': result['status'], 'candidate_id': 10000007,
                                    'report_sha256': sha(report_path) if report_path.exists() else None,
                                    'admitted_references': 0, 'new_samples': 0})


if __name__ == '__main__':
    work() if sys.argv[1:] == ['--worker'] else setup()
