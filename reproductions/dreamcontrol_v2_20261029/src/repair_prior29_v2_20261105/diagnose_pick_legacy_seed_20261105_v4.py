"""One candidate: test a physically checked, phase-aligned initialization without reusing reference data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE = ROOT / 'repair_prior29_v2_20261105'
OUT = STAGE / 'pick_legacy_seed_diagnostic_v4'
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
    prior = STAGE / 'pick_legacy_warm_start_readonly_v1.json'
    assert sha(prior) == '8f32511895f0a816f1a3ff49730afdb14cd68b4999c2443548ea3f062e76fa4c'
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
                '''            # The old, immutable state supplies only an optimizer start.
            from scipy.spatial.transform import Slerp
            legacy = Path("/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103/reference_scale500_20261104/final_library_v1/reference_bank.json")
            assert sha_file(legacy) == "01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572"
            rows = json.loads(legacy.read_text())["references"]
            matching = [row for row in rows if json.loads(Path(row["kinematic_report"]).read_text())["candidate_id"] == 5]
            assert len(matching) == 1 and self.frames == 120
            row = matching[0]
            legacy_state = Path(row["path"])
            assert sha_file(legacy_state) == row["sha256"]
            assert sha_file(Path(row["postopt_report"])) == row["postopt_report_sha256"]
            clock = np.arange(self.frames)
            contact = int(m["grasp_frame"])
            assert 40 <= contact <= 50
            phase = np.interp(clock, [0,contact,contact+40,119], [0,40,80,119])
            with np.load(legacy_state, allow_pickle=False) as z:
                assert z["joint_names"].tolist() == self.names and float(z["fps"]) == 20.
                seed_q = np.column_stack([np.interp(phase,clock,q) for q in z["q"].T])
                pos = np.column_stack([np.interp(phase,clock,v) for v in z["root_pos"].T])
                rot = Slerp(clock,self.Rotation.from_quat(z["root_quat"][:,[1,2,3,0]]))(phase)
                seed_root = np.c_[rot.as_quat()[:,[3,0,1,2]],pos]
            seed_q = jnp.asarray(np.clip(seed_q,self.low+1e-6,self.high-1e-6), dtype=jnp.float32)
            seed_root = jnp.asarray(seed_root,dtype=jnp.float32)
            report["initialization_only"] = "phase-aligned archived state is a seed; A refits the current raw with full table residual"
            report["legacy_seed_sha256"] = row["sha256"]
            report["legacy_seed_candidate_id"] = 5
            qa, ra = self.solve(target, goals, orient, mask, seed_q, seed_root,''')
            body = modify(body, '        def solve(target, goals, orientations, mask,',
                          '''        @jaxls.Cost.create_factory
        def wrist_gauge_cost(vals, q, r, orientation, mask):
            current = jaxlie.SO3(pose(vals, q, r).rotation().wxyz[wrists])
            angle = (jaxlie.SO3(orientation).inverse() @ current).log()
            return (angle * mask[:, None] * 2.5).flatten()

        def solve(target, goals, orientations, mask,''')
            body = modify(body, '                # Only a tiny joint gauge prior',
                          '''                costs.append(wrist_gauge_cost(q, r, orientations, mask))
                # Only a tiny joint gauge prior''')
            body = modify(body, '            return jnp.concatenate([dq, slip])',
                          '''            current = jaxlie.SO3(pose(vals, q, r).rotation().wxyz[wrists])
            previous = jaxlie.SO3(pose(vals, qp, rp).rotation().wxyz[wrists])
            delta = (previous.inverse() @ current).log()
            speed = jnp.sqrt(jnp.maximum(jnp.sum(delta**2, axis=-1), 1e-12))
            wrist_step = jnp.maximum(speed - .20, 0.) / .05
            return jnp.concatenate([dq, slip, wrist_step])''')
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
