"""Recheck four complete video timelines and their human/robot point inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from g1_fk import URDFFK

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'five_conservative_preview_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
ARCHIVE = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report = json.loads((HERE / 'report.json').read_text())
    assert sha(INPUT / 'manifest.json') == report['manifest_sha256']
    assert sha(INPUT / 'state_manifest.json') == report['state_manifest_sha256']
    assert sha(URDF) == report['urdf_sha256']
    records = {x['sample_id']: x for x in json.loads((INPUT / 'manifest.json').read_text())['records']}
    states = {x['sample_id']: x for x in json.loads((INPUT / 'state_manifest.json').read_text())['records']}
    robot_fk = URDFFK(URDF); outputs = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for item in report['cases']:
            ident = item['sample_id']; rec = records[ident]; state = states[ident]
            assert sha(INPUT / state['state_path']) == item['state_sha256'] == state['state_sha256']
            assert sha(INPUT / rec['feature_path']) == item['feature_sha256'] == rec['feature_sha256']
            file = Path(item['source_evidence']['path']); assert sha(file) == item['source_evidence']['sha256']
            with np.load(INPUT / state['state_path'], allow_pickle=False) as f:
                roots, q, times, names = (f[x].copy() for x in ('root_wxyz_xyz','joint_pos','sample_times','joint_names'))
            with np.load(file, allow_pickle=False) as f:
                source_times = f['source_times'].copy()
                if ident.startswith('ACCAD_'):
                    points = f['human'].copy()
                else:
                    hm = f['original_joints_humanml'].copy()
                    points = np.stack([-hm[..., 0], hm[..., 2], hm[..., 1]], axis=-1)
            points = points.astype(np.float64)
            interpolated = np.stack([np.interp(times, source_times, points[:, i, j])
                                     for i in range(22) for j in range(3)], axis=1).reshape(-1,22,3)
            digest = hashlib.sha256(np.ascontiguousarray(interpolated, dtype='<f8').tobytes()).hexdigest()
            assert digest == item['human_aligned_points_sha256']
            arithmetic_error = 0.
            for k, t in enumerate(times):
                right = int(np.searchsorted(source_times, t, side='right'))
                if right == len(source_times):
                    assert abs(t - source_times[-1]) < 1e-9
                    expected = points[-1]
                else:
                    left = max(0, right-1)
                    weight = (t-source_times[left])/(source_times[right]-source_times[left])
                    expected = points[left] + weight * (points[right] - points[left])
                arithmetic_error = max(arithmetic_error, float(np.abs(expected - interpolated[k]).max()))
            assert arithmetic_error < 1e-10
            robot = robot_fk.forward(roots[:, 4:], roots[:, [1,2,3,0]], q, list(names))
            assert hashlib.sha256(np.ascontiguousarray(robot, dtype='<f8').tobytes()).hexdigest() == item['robot_points_sha256']
            clip = '004347' if ident == 'ACCAD_004347' else ident
            raw = archive.read('texts/' + clip + '.txt')
            assert hashlib.sha256(raw).hexdigest() == item['caption_member_sha256']
            rows = [line.split('#') for line in raw.decode('utf8').splitlines()]
            assert [{'caption': x[0], 'raw_tags': x[2:]} for x in rows] == item['caption_records']
            video = HERE / (ident + '.mp4'); assert sha(video) == item['video_sha256']
            stream = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(video)], text=True))['streams'][0]
            assert int(stream['nb_read_frames']) == item['frames'] == len(times) and stream['r_frame_rate'] == '20/1'
            outputs.append({'sample_id': ident, 'frames': len(times), 'interpolation_max_error': arithmetic_error,
                            'video_sha256': sha(video), 'caption_accepted': False})
    assert len(outputs) == 4 and sum(x['frames'] for x in outputs) == 193
    out = {'report_sha256': sha(HERE / 'report.json'), 'cases': outputs, 'frames_verified': 193,
           'training_ready': False, 'limitations': ['Same FK library; no independent full human reconstruction or video rendering.',
           'Sample timing is conditional; caption semantics and physical quality are not accepted.']}
    with (HERE / 'audit_report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'cases': len(outputs), 'frames_verified': 193, 'training_ready': False}))


if __name__ == '__main__':
    main()
