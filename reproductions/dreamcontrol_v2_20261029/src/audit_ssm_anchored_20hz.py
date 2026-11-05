"""Independently check candidate hashes, time grid, captions and selected blends.

This does not verify body identity, filter fidelity, semantic or physical quality.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'ssm_rawfps_anchor_gpu_20261031'
HUMAN = ROOT / 'ssm_rawfps_human_batch_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    batch = json.loads((HERE/'batch_report.json').read_text())
    cases = []
    for record in batch['cases']:
        sid = record['id']
        assert sid in ('007410', '011120')
        sample = HERE/(sid+'.npz')
        assert sha(sample) == record['output_sha256']
        human = json.loads((HUMAN/sid/'human_report.json').read_text())
        assert sha(HUMAN/sid/'human_report.json') == record['human_report_sha256']
        assert sha(RUN/sid/'audit_report.json') == record['anchored_audit_sha256']
        assert sha(RUN/'comparison.json') == record['comparison_sha256']
        with np.load(HUMAN/sid/'human_rawfps.npz', allow_pickle=False) as data:
            frames, source_times = data['source_frames'].copy(), data['source_times'].copy()
        with np.load(sample, allow_pickle=False) as data:
            x = {k: data[k].copy() for k in data.files}
        t = x['sample_times']; features = x['features']; names = x['joint_names'].tolist()
        np.testing.assert_array_equal(x['raw_sample_indices'], np.arange(1, len(frames)))
        np.testing.assert_array_equal(x['raw_source_frames'], frames[1:])
        np.testing.assert_allclose(t, source_times[1]+np.arange(len(t))/20, rtol=0, atol=1e-12)
        assert t[0] >= source_times[1] and t[-1] <= source_times[-1]
        assert features.shape == (len(t)-1, 263) and np.isfinite(features).all()
        np.testing.assert_allclose(np.linalg.norm(x['root_wxyz_xyz'][:,:4], axis=1), 1, rtol=0, atol=1e-10)
        assert all(np.isfinite(x[k]).all() for k in ('root_wxyz_xyz','joint_pos','raw_blended'))
        limits = {j.get('name'): (float(j.find('limit').get('lower')), float(j.find('limit').get('upper')))
                  for j in ET.parse(URDF).getroot().findall('joint') if j.get('type') == 'revolute'}
        over = max(max(0., limits[name][0]-float(v), float(v)-limits[name][1])
                   for row in x['joint_pos'] for name, v in zip(names, row))
        np.testing.assert_allclose(over, record['max_joint_limit_after_filter_rad'], atol=1e-12, rtol=0)
        original = human['original_caption_records']
        assert [(r['text'],r['tags']) for r in original] == [(r['caption'],r['raw_tags']) for r in record['caption_records']]
        for cap in record['caption_records']:
            assert cap['raw_tags'] == ['0.0','0.0']
            assoc = cap['association']
            assert assoc['matching_feature_intervals'] == list(range(len(features)))
            assert assoc['caption_accepted'] is False and assoc['feature_covers_full_annotation_sample_extent'] is False
        rows = [json.loads(row) for row in (RUN/sid/'windows.jsonl').read_text().splitlines()]
        weights = [1.,2.,3.,4.,3.,2.,1.]
        max_blend_error = 0.
        for pos in (0, len(frames)//2, len(frames)-2):
            index = int(x['raw_sample_indices'][pos]); entries = []
            for row in rows:
                with np.load(RUN/sid/row['artifact'], allow_pickle=False) as win:
                    hit = np.flatnonzero(win['sample_indices']==index)
                    if len(hit):
                        i = int(hit[0]); entries.append((weights[i], np.r_[win['root_wxyz_xyz'][i], win['joint_pos'][i]].astype(np.float64)))
            assert entries
            first = entries[0][1][:4]
            total = np.zeros(34); weight = 0.
            for w, v in entries:
                v = v.copy()
                if v[:4] @ first < 0: v[:4] *= -1
                total += w*v; weight += w
            total /= weight; total[:4] /= np.linalg.norm(total[:4])
            expected = x['raw_blended'][pos]
            if total[:4] @ expected[:4] < 0: total[:4] *= -1
            max_blend_error = max(max_blend_error, float(np.max(np.abs(total-expected))))
        assert max_blend_error < 1e-10
        assert record['training_ready'] is False and record['caption_accepted'] is False
        cases.append({'id':sid,'output_sha256':sha(sample),'sample_count':len(t),'feature_rows':len(features),
                      'caption_count':len(original),'max_limit_overrun_rad':over,'three_frame_blend_max_error':max_blend_error})
    assert len(cases)==2
    result={'audit_script_sha256':sha(Path(__file__)),'cases':cases,'training_ready':False,
            'limitations':'No independent full filter, FK or encoder rerun; source, semantics and physical quality are unaccepted.'}
    with (HERE/'independent_audit.json').open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
