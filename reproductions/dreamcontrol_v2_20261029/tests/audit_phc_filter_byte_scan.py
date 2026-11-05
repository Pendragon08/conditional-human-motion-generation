"""Independently audit literal PHC issue-list searches and source-path mapping."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report = json.loads((HERE/'byte_scan.json').read_text())
    data = (HERE/'source.pkl').read_bytes()
    assert hashlib.sha256(data).hexdigest() == report['source_sha256']
    assert len(data) == report['bytes'] and report['training_ready'] is False
    alignment = ROOT/'ssm_id_alignment_20261030/ids.jsonl'
    assert sha(alignment) == '575c9c008298dbf30d21317f258346a3dd11cc7efc6858d242a26c11c6333b46'
    rows = {r['motion_id']:r for r in map(json.loads,alignment.read_text().splitlines())}
    assert set(report['candidate_key_counts']) == {'000008','000069','003017','013303'}
    for name, found in report['candidate_key_counts'].items():
        path = rows[name]['source_path']
        assert path.startswith('./pose_data/') and path.endswith('.npy')
        key = '0-' + path[len('./pose_data/'):-len('.npy')].replace('/','_')
        assert key == found['literal'] and data.count(key.encode('ascii')) == found['byte_count']
    for key,count in report['controls'].items():
        assert data.count(key.encode('ascii')) == count and count > 0
    assert data.count(b'SSM_synced') == report['ssm_synced_literal_count']
    convert = HERE/'phc_convert.py'; download = HERE/'phc_download.sh'
    assert sha(convert) == '63e507cbd3c9fffa862175a814433997614c0ce812f72b921fae2573a7dd638b'
    assert sha(download) == 'dfc68bc0fa58de1a846160ef3f38283c85a2370b2dd268e918cc3c28315136ce'
    c, d = convert.read_text(),download.read_text()
    assert 'sample_data/amass_copycat_occlusion_v3.pkl' in c
    assert '1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc' in d
    assert "'test': ['Transitions_mocap', 'SSM_synced']" in c
    result = {'report_sha256':sha(HERE/'byte_scan.json'),'source_sha256':sha(HERE/'source.pkl'),
              'alignment_sha256':sha(alignment),'convert_sha256':sha(convert),'download_sha256':sha(download),
              'candidate_names_verified':4,'controls_verified':len(report['controls']),
              'training_ready':False,
              'limitations':'Literal byte audit only; joblib not decoded, authors exact filtering and candidate eligibility remain unknown.'}
    with (HERE/'byte_audit.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':main()
