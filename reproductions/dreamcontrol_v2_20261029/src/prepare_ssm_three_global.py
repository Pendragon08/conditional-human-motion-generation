"""Prepare sequential full-trajectory PyRoki solves for three audited sources."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE = ROOT / 'ssm_three_human_candidates_20261031'
OLD = ROOT / 'pyroki_ssm_000008_global_20261031'
HERE = Path(__file__).resolve().parent


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1: raise ValueError('unexpected template: ' + old)
    return text.replace(old, new)


def main():
    human = json.loads((SOURCE / 'report.json').read_text())
    audit = json.loads((SOURCE / 'independent_audit.json').read_text())
    old = json.loads((OLD / 'report.json').read_text())
    if sha(SOURCE / 'report.json') != audit['witness_report_sha256']:
        raise ValueError('human source report audit mismatch')
    if sha(OLD / 'run.py') != old['script_sha256']:
        raise ValueError('solver template differs from prior run')
    items = {x['id']: x for x in human['witnesses']}
    checks = {x['id'] for x in audit['checks']}
    if set(items) != {'000069', '003017', '013303'} or set(items) != checks:
        raise ValueError('unexpected source or audit coverage')
    template = (OLD / 'run.py').read_text(encoding='utf8')
    launcher = (OLD / 'run.sh').read_text(encoding='utf8')
    records = []
    for name in ('000069', '003017', '013303'):
        item = items[name]
        artifact = SOURCE / (name + '_joint_witness.npz')
        if sha(artifact) != item['artifact_sha256']: raise ValueError('source artifact changed')
        n = item['input_samples']; text = template
        text = replace_once(text, "ROOT / 'humanml_mirror_time_witness_v2_20261030/000008_joint_witness.npz'",
                            "ROOT / 'ssm_three_human_candidates_20261031/" + name + "_joint_witness.npz'")
        text = replace_once(text, '8354f592da84aa9e62b87f4ab467465cd8a43120e39d5c8ae346966a8883a0b7', item['artifact_sha256'])
        text = replace_once(text, 'p.shape != (43, 22, 3)', f'p.shape != ({n}, 22, 3)')
        text = replace_once(text, 'q.shape != (42, 27) or rq.shape != (42, 7)',
                            f'q.shape != ({n-1}, 27) or rq.shape != ({n-1}, 7)')
        text = replace_once(text, 'sample_indices=np.arange(1, 43)', f'sample_indices=np.arange(1, {n})')
        folder = HERE / name; folder.mkdir()
        with (folder / 'run.py').open('x', encoding='utf8', newline='\n') as f: f.write(text)
        old_dir = 'pyroki_ssm_000008_global_20261031'
        new_dir = 'ssm_three_global_20261031/' + name
        with (folder / 'run.sh').open('x', encoding='utf8', newline='\n') as f:
            f.write(replace_once(launcher, old_dir, new_dir))
        records.append({'id': name, 'human_artifact_sha256': sha(artifact), 'input_points': n,
                        'script_sha256': sha(folder / 'run.py'), 'launcher_sha256': sha(folder / 'run.sh')})
    result = {'human_report_sha256': sha(SOURCE / 'report.json'),
              'human_audit_sha256': sha(SOURCE / 'independent_audit.json'),
              'template_report_sha256': sha(OLD / 'report.json'), 'records': records,
              'training_ready': False,
              'limitations': 'Experimental v1 helper/assumed floor, source model and license unaccepted; no training data.'}
    with (HERE / 'manifest.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
