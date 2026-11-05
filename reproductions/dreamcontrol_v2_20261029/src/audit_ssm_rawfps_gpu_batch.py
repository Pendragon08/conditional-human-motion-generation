"""Audit all saved source-rate G1 windows without rerunning optimization."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMAN = ROOT/'ssm_rawfps_human_batch_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--human-root',type=Path,default=HUMAN)
    parser.add_argument('--samples',nargs='+',default=['007410','011120','002941'])
    args=parser.parse_args()
    if len(args.samples)!=len(set(args.samples)) or any(not s.isdecimal() or len(s)!=6 for s in args.samples):
        raise ValueError('unique six-digit sample IDs required')
    human_root=args.human_root.resolve(strict=True)
    limits = {}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        if joint.get('type') == 'continuous':
            limits[joint.get('name')] = (-np.inf,np.inf)
        elif joint.get('type') == 'revolute':
            limit = joint.find('limit'); limits[joint.get('name')] = (float(limit.get('lower')),float(limit.get('upper')))
    results = []
    for sample in args.samples:
        out, folder = HERE/sample, human_root/sample
        report = json.loads((out/'report.json').read_text()); human_audit = json.loads((folder/'human_audit.json').read_text())
        if (sample != report['id'] or sha(folder/'human_rawfps.npz') != report['source_sha256'] or
            report['source_sha256'] != human_audit['artifact_sha256'] or sha(URDF) != report['urdf_sha256'] or
            sha(out/'windows.jsonl') != report['rows_sha256']):
            raise ValueError('source or output provenance changed')
        with np.load(folder/'human_rawfps.npz',allow_pickle=False) as f:
            src_frames, times = f['source_frames'].copy(), f['source_times'].copy()
        rows = [json.loads(line) for line in (out/'windows.jsonl').read_text().splitlines()]
        starts = list(range(0,len(src_frames)-7,4))
        if starts[-1] != len(src_frames)-8: starts.append(len(src_frames)-8)
        if len(rows) != len(starts) or len(rows) != report['window_count']:
            raise ValueError('missing or extra windows')
        names0 = None; previous = None; covered = set(); max_joint_limit = 0.; overlaps = []
        for i, (row, start) in enumerate(zip(rows, starts)):
            path = out/row['artifact']
            if row['window'] != i or sha(path) != row['artifact_sha256']:
                raise ValueError('window or hash mismatch')
            with np.load(path,allow_pickle=False) as f:
                indices, raw, t = f['sample_indices'].copy(), f['source_frames'].copy(), f['source_times'].copy()
                rq, q, names = f['root_wxyz_xyz'].copy(), f['joint_pos'].copy(), [str(x) for x in f['joint_names']]
            if (indices.tolist() != row['sample_indices'] or not np.array_equal(indices,np.arange(start+1,start+8)) or
                not np.array_equal(raw,src_frames[indices]) or not np.array_equal(t,times[indices]) or
                rq.shape != (7,7) or q.shape != (7,27) or len(set(names)) != 27 or
                not np.isfinite(rq).all() or not np.isfinite(q).all() or
                not np.allclose(np.linalg.norm(rq[:,:4],axis=1),1.,atol=1e-4,rtol=0)):
                raise ValueError('invalid window contents')
            if names0 is None:
                names0 = names
                if set(names) != set(limits): raise ValueError('joint identities differ from URDF')
                lower = np.array([limits[x][0] for x in names]); upper = np.array([limits[x][1] for x in names])
            elif names != names0:
                raise ValueError('joint ordering changed')
            max_joint_limit = max(max_joint_limit, float(np.maximum(np.maximum(lower-q,q-upper),0).max()))
            covered.update(indices.tolist())
            if previous is not None:
                common = np.intersect1d(previous['indices'],indices)
                a = np.searchsorted(previous['indices'],common); b = np.searchsorted(indices,common)
                rot1 = Rotation.from_quat(previous['root'][a][:,[1,2,3,0]])
                rot2 = Rotation.from_quat(rq[b][:,[1,2,3,0]])
                actual = {'root_m': float(np.linalg.norm(previous['root'][a,4:]-rq[b,4:],axis=1).max()),
                          'rotation_rad': float((rot1.inv()*rot2).magnitude().max()),
                          'joint_rad': float(np.abs(previous['q'][a]-q[b]).max())}
                if common.tolist() != row['overlap']['sample_indices'] or any(abs(actual[k]-row['overlap'][k])>1e-6 for k in actual):
                    raise ValueError('overlap metadata mismatch')
                overlaps.append(actual)
            previous = {'indices':indices,'root':rq,'q':q}
        if covered != set(range(1,len(src_frames))):
            raise ValueError('source coverage mismatch')
        result = {'id':sample,'report_sha256':sha(out/'report.json'),'source_sha256':sha(folder/'human_rawfps.npz'),
                  'windows_verified':len(rows),'covered_sample_indices':[1,len(src_frames)-1],
                  'raw_frame_first_last':[int(src_frames[1]),int(src_frames[-1])],
                  'max_joint_limit_violation_rad':max_joint_limit,
                  'max_overlap_root_m':max(x['root_m'] for x in overlaps),
                  'max_overlap_rotation_rad':max(x['rotation_rad'] for x in overlaps),
                  'max_overlap_joint_rad':max(x['joint_rad'] for x in overlaps),
                  'training_ready':False,'limitations':['Saved outputs only; no independent optimization.',
                  'No actual-time, caption semantics, real contact, collision or physical acceptance.']}
        with (out/'audit_report.json').open('x',encoding='utf8') as f:
            json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
        results.append(result); print(json.dumps({'audited':sample,'windows':len(rows)}),flush=True)
    with (HERE/'batch_audit.json').open('x',encoding='utf8') as f:
        json.dump({'cases':results,'training_ready':False},f,indent=2,allow_nan=False); f.write('\n')


if __name__ == '__main__': main()
