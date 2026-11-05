"""Adapt the audited CPU window solver to a 120-Hz ACCAD candidate."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD=ROOT/'pyroki_ssm_000008_human_20261031'
HUMAN=ROOT/'accad_013267_fullrate_20261031'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(text,a,b):
    if text.count(a)!=1:raise ValueError('unexpected template: '+a)
    return text.replace(a,b)


def main():
    source=json.loads((HUMAN/'human_report.json').read_text())
    audited=json.loads((HUMAN/'human_audit.json').read_text())
    old=json.loads((OLD/'report.json').read_text())
    if sha(HUMAN/'human_report.json')!=audited['human_report_sha256'] or sha(HUMAN/'human_120hz.npz')!=source['artifact_sha256']:
        raise ValueError('human provenance changed')
    if sha(OLD/'run.py')!=old['script_sha256']:raise ValueError('solver template changed')
    text=(OLD/'run.py').read_text()
    edits=[("ROOT / 'humanml_mirror_time_witness_v2_20261030/000008_joint_witness.npz'", "ROOT / 'accad_013267_fullrate_20261031/human_120hz.npz'"),
           ('8354f592da84aa9e62b87f4ab467465cd8a43120e39d5c8ae346966a8883a0b7',source['artifact_sha256']),
           ("d['original_joints_humanml']","d['human']"),
           ('converted.shape != (43, 22, 3) or not np.isfinite(converted).all() or len(times) != 43',
            'converted.shape != (355, 22, 3) or not np.isfinite(converted).all() or len(times) != 355'),
           ('human = np.stack((-converted[..., 0], converted[..., 2], converted[..., 1]), axis=-1)','human = converted'),
           ('np.diff(source_frames) == 6','np.diff(source_frames) == 1'),
           ('abs(fps - 20.00000762939453) > 1e-6','abs(fps - 120.0) > 1e-6'),
           ('# Original v1 temporal weights rather than the prior 120 Hz sixfold variant.',
            '# Exploratory sixfold temporal scaling for a declared 120 Hz source; not author weights.'),
           ("'fps_reference_derived': fps","'fps_assuming_declared_source_rate': fps"),
           ("weights['root_smoothness'] = 2.0","weights['root_smoothness'] = 12.0"),
           ("weights['foot_skating'] = 2.0","weights['foot_skating'] = 12.0"),
           ("'one short SSM candidate, marker evidence inherited'","'ACCAD candidate path/FPS and official intermediate identity unverified'"),
           ("'no verified caption, split, license or physical acceptance'","'no accepted caption semantics, source identity, or physical quality'")]
    for a,b in edits:text=replace(text,a,b)
    with (HERE/'run.py').open('x',encoding='utf8',newline='\n') as f:f.write(text)
    shell=(OLD/'run.sh').read_text()
    shell=replace(shell,'pyroki_ssm_000008_human_20261031','accad_013267_pyroki_fullrate_20261031')
    shell=replace(shell,'ionice -c 3 nice -n 15 taskset -c 19',
                  'ulimit -v 33554432\nionice -c 3 nice -n 15 timeout 2400 taskset -c 19')
    with (HERE/'run.sh').open('x',encoding='utf8',newline='\n') as f:f.write(shell)
    result={'human_report_sha256':sha(HUMAN/'human_report.json'),'human_audit_sha256':sha(HUMAN/'human_audit.json'),
            'human_artifact_sha256':sha(HUMAN/'human_120hz.npz'),'template_sha256':sha(OLD/'run.py'),
            'script_sha256':sha(HERE/'run.py'),'launcher_sha256':sha(HERE/'run.sh'),
            'input_points':355,'expected_windows':88,'nominal_hz':120,
            'training_ready':False,'limitations':['old v1 weights and heuristic 120Hz scaling; not author method',
                'windows require audited stitching before 20Hz encoding',
                'caption, source model, actual fps and physical quality unaccepted']}
    with (HERE/'manifest.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
