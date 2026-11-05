"""Compare all anchored and unanchored windows, including source keypoints."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from g1_fk import URDFFK

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMAN = ROOT/'ssm_rawfps_human_batch_20261031'
BASE = ROOT/'ssm_rawfps_gpu_batch_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent
LINKS = ['left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
         'left_ankle_pitch_link','right_ankle_pitch_link','left_shoulder_pitch_link','right_shoulder_pitch_link',
         'left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link']
HUMAN_IDS = [1,2,4,5,7,8,16,17,18,19,20,21]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def stats(values):
    x = np.asarray(values,dtype=float)
    return {'mean':float(x.mean()),'p95':float(np.quantile(x,.95)),'max':float(x.max())}


def main():
    fk = URDFFK(URDF); results = []
    for sample in ('007410','011120','002941'):
        source = HUMAN/sample/'human_rawfps.npz'; old = BASE/sample; new = HERE/sample
        old_report = json.loads((old/'report.json').read_text()); old_audit = json.loads((old/'audit_report.json').read_text())
        new_report = json.loads((new/'report.json').read_text()); new_audit = json.loads((new/'audit_report.json').read_text())
        if (sha(old/'report.json') != old_audit['report_sha256'] or sha(new/'report.json') != new_audit['report_sha256'] or
            sha(source) != old_report['source_sha256'] or sha(source) != new_report['source_sha256'] or
            sha(URDF) != old_report['urdf_sha256'] or sha(URDF) != new_report['urdf_sha256'] or
            sha(old/'windows.jsonl') != old_report['rows_sha256'] or sha(new/'windows.jsonl') != new_report['rows_sha256']):
            raise ValueError('comparison provenance changed')
        old_rows = [json.loads(x) for x in (old/'windows.jsonl').read_text().splitlines()]
        new_rows = [json.loads(x) for x in (new/'windows.jsonl').read_text().splitlines()]
        if len(old_rows) != len(new_rows): raise ValueError('window count differs')
        with np.load(source,allow_pickle=False) as f: human = f['human'].copy()
        errors = {'baseline':[],'anchored':[]}; diffs = []; old_overlaps=[]; new_overlaps=[]
        for a,b in zip(old_rows,new_rows):
            if a['window'] != b['window'] or a['sample_indices'] != b['sample_indices']:
                raise ValueError('window mismatch')
            motions=[]
            for label,folder,row in [('baseline',old,a),('anchored',new,b)]:
                path = folder/row['artifact']
                if sha(path) != row['artifact_sha256']: raise ValueError('window changed')
                with np.load(path,allow_pickle=False) as f:
                    indices = f['sample_indices'].copy(); root = f['root_wxyz_xyz'].copy()
                    q = f['joint_pos'].copy(); names = [str(x) for x in f['joint_names']]
                points = fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names,links=LINKS)
                errors[label].extend(np.linalg.norm(points-human[indices][:,HUMAN_IDS],axis=-1).ravel().tolist())
                motions.append((root,q))
            ar,aq = motions[0]; br,bq = motions[1]
            angle = (Rotation.from_quat(ar[:,[1,2,3,0]]).inv()*Rotation.from_quat(br[:,[1,2,3,0]])).magnitude()
            diffs.append({'root_m':float(np.linalg.norm(ar[:,4:]-br[:,4:],axis=1).max()),
                          'rotation_rad':float(angle.max()),'joint_rad':float(np.abs(aq-bq).max())})
            if a['overlap'] is not None:
                old_overlaps.append(a['overlap']); new_overlaps.append(b['overlap'])
        result = {'id':sample,'windows_compared':len(old_rows),'source_sha256':sha(source),
                  'baseline_audit_sha256':sha(old/'audit_report.json'),'anchored_audit_sha256':sha(new/'audit_report.json'),
                  'baseline_overlap':{k:stats([x[k] for x in old_overlaps]) for k in ('root_m','rotation_rad','joint_rad')},
                  'anchored_overlap':{k:stats([x[k] for x in new_overlaps]) for k in ('root_m','rotation_rad','joint_rad')},
                  'baseline_vs_anchor':{k:stats([x[k] for x in diffs]) for k in ('root_m','rotation_rad','joint_rad')},
                  'unscaled_world_keypoint_error_m':{k:stats(v) for k,v in errors.items()},
                  'joint_limit_violation_rad':{'baseline':old_audit['max_joint_limit_violation_rad'],
                                               'anchored':new_audit['max_joint_limit_violation_rad']},
                  'training_ready':False}
        results.append(result); print(json.dumps({'compared':sample,'windows':len(old_rows)}),flush=True)
    out = {'urdf_sha256':sha(URDF),'script_sha256':sha(Path(__file__)),'cases':results,'training_ready':False,
           'limitations':['Same G1 FK code; human/body morphology makes world error an imperfect target.',
                           'No independent optimizer, actual ground, caption or physical dynamics acceptance.',
                           'Soft anchor weight is an exploration, not the paper author method.']}
    with (HERE/'comparison.json').open('x',encoding='utf8') as f:
        json.dump(out,f,indent=2,allow_nan=False); f.write('\n')


if __name__ == '__main__': main()
