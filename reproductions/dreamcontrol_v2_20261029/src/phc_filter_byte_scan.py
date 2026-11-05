"""Search selected PHC issue-list keys without deserializing the joblib file.

Byte absence is not proof a recording is clean or eligible for training.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source.pkl'
EXPECTED = '2c336550d452e5769bc12d199c285bb3717af651f3691ee18ac6b82f2de0a3d8'
CANDIDATES = {
    '000008': '0-SSM_synced_20161014_50033_ATUSquat_sync_poses',
    '000069': '0-SSM_synced_20160930_50032_running_on_the_spot_sync_poses',
    '003017': '0-SSM_synced_20161014_50033_jumping_jacks_sync_poses',
    '013303': '0-SSM_synced_20160930_50032_punch_kick_sync_poses',
}
CONTROLS = ['0-BioMotionLab_NTroje_rub020_0002_treadmill_slow_poses',
            '0-KIT_3_kneel_down_with_stool_right_hand08_poses']


def main():
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED: raise ValueError('source SHA mismatch')
    def count(text): return raw.count(text.encode('ascii'))
    controls = {x: count(x) for x in CONTROLS}
    if any(v == 0 for v in controls.values()): raise ValueError('control pattern absent')
    result = {'source_sha256': digest, 'bytes': len(raw),
              'phc_code_commit': '846988d433ce1f341e85ac6fbd2cd51911bb3341',
              'source_url': 'https://drive.google.com/uc?export=download&id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc',
              'code_url': 'https://raw.githubusercontent.com/ZhengyiLuo/PHC/846988d433ce1f341e85ac6fbd2cd51911bb3341/scripts/data_process/convert_amass_data.py',
              'candidate_key_counts': {k: {'literal': v, 'byte_count': count(v)} for k,v in CANDIDATES.items()},
              'ssm_synced_literal_count': count('SSM_synced'), 'controls': controls,
              'training_ready': False,
              'limitations': ['raw byte search, not pickle/joblib deserialization or a complete key extraction',
                  'PHC key spelling/path layout and this downloadable issue list may differ from v2 authors usage',
                  'no literal match does not certify feasibility, source identity, caption or license',
                  'the pinned PHC conversion places SSM_synced in its test group, not its train group']}
    with (HERE / 'byte_scan.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
