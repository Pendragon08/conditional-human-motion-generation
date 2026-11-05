"""Re-read KIT previews and source captions; report technical checks only."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from audit_robot_states import audit_robot_states

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW = ROOT / 'kit_dart_hypothesis_263_preview_20261102'
HUMANS = ROOT / 'kit_dart_hypothesis_human_20261102'
WINDOWS = ROOT / 'kit_dart_hypothesis_work_20261102'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
TEXTS = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
IDS = ('001199', '007754', '006923', '001171', '007259', '001893', '009236')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((PREVIEW / 'report.json').read_text())
    index = {r['id']: r for r in report['cases']}
    if len(index) != 7 or set(index) != set(IDS):
        raise ValueError('preview IDs mismatch')
    archive_hash = sha(TEXTS)
    cases = []
    with zipfile.ZipFile(TEXTS) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('duplicate caption archive members')
        for sid in IDS:
            human_dir = HUMANS / ('human_' + sid)
            human_report = json.loads((human_dir / 'report.json').read_text())
            human_file = human_dir / 'human_20hz_dart_hypothesis.npz'
            target = PREVIEW / (sid + '.npz')
            row = index[sid]
            raw = archive.read(human_report['caption_member'])
            if (sha(target) != row['output_sha256'] or sha(human_file) != row['human_sha256']
                or archive_hash != human_report['caption_archive_sha256']
                or hashlib.sha256(raw).hexdigest() != human_report['caption_sha256']
                or raw.decode('utf-8') != human_report['caption_raw_utf8']):
                raise ValueError('artifact or caption source hash mismatch')
            with np.load(human_file, allow_pickle=False) as h, np.load(target, allow_pickle=False) as t:
                source_frames, times = h['source_frames'], h['source_times']
                expected = np.arange(1, len(source_frames))
                if (not np.array_equal(t['sample_indices'], expected)
                    or not np.array_equal(t['source_frames'], source_frames[expected])
                    or not np.array_equal(t['sample_times'], times[expected])
                    or not np.all(np.diff(source_frames) == 5)
                    or not np.allclose(times, source_frames / 100., atol=1e-12, rtol=0)):
                    raise ValueError('hypothetical time mapping mismatch')
            evidence = {'sample_id': sid, 'feature_path': target.name, 'state_path': target.name,
                        'feature_sha256': row['output_sha256'], 'state_sha256': row['output_sha256'],
                        'foot_velocity_squared_threshold': .002}
            try:
                result = audit_robot_states([evidence], feature_root=PREVIEW, state_root=PREVIEW,
                    urdf_path=URDF, tolerance_rad=1e-6, fk_decode_tolerance_m=1e-4,
                    reencoding_tolerance=1e-5)
                technical = {'passed': result['state_technical_checks_passed'], 'details': result['cases'][0]}
            except (ValueError, RuntimeError) as exc:
                technical = {'passed': False, 'error': str(exc)}
            cases.append({'id': sid, 'preview_sha256': sha(target), 'human_report_sha256': sha(human_dir / 'report.json'),
                          'caption_member': human_report['caption_member'], 'caption_sha256': human_report['caption_sha256'],
                          'caption_line_count': len(raw.decode('utf-8').splitlines()),
                          'caption_semantics_accepted': False, 'nominal_timing_only': True,
                          'missing_initial_source_sample': True, 'technical': technical})
    result = {'cases': cases, 'caption_archive_sha256': archive_hash, 'source_preview_report_sha256': sha(PREVIEW / 'report.json'),
              'window_audit_sha256': sha(WINDOWS / 'robot_windows_audit_20261102.json'),
              'script_sha256': sha(Path(__file__)), 'training_ready': False,
              'limitations': ['100 Hz assumed, source header 120 Hz; not a measured clock.',
                 'Archive association is by ID and does not establish text/action identity or semantic agreement.',
                 'Uses existing experimental FK and encoder; contact and physical dynamics are not checked.']}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'cases': len(cases), 'technical_passed': sum(c['technical']['passed'] for c in cases),
                      'training_ready': False}))


if __name__ == '__main__':
    main()
