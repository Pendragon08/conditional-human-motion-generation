"""Compare anchored and baseline estimates at the same S3 frames."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p,expected):
    if sha(p)!=expected:raise ValueError('artifact hash mismatch')
    with np.load(p,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def main():
    base=ROOT/'pyroki_grab_s3_windowed_20261030'
    bdoc=json.loads((base/'report.json').read_text());doc=json.loads((HERE/'report.json').read_text())
    audit=json.loads((HERE/'audit_report.json').read_text())
    if sha(base/'report.json')!=doc['baseline_report_sha256'] or sha(base/'windows.jsonl')!=bdoc['rows_sha256'] or sha(HERE/'report.json')!=audit['report_sha256']:
        raise ValueError('audit provenance mismatch')
    rows=[json.loads(x) for x in (HERE/'windows.jsonl').read_text().splitlines()];baseline=[json.loads(x) for x in (base/'windows.jsonl').read_text().splitlines()]
    if len(rows)!=23:raise ValueError('expected 23 tail windows')
    result=[]
    for row in rows:
        i=row['window'];a=load(HERE/f'window_{i:02d}.npz',row['artifact_sha256']);b=load(base/f'window_{i:02d}.npz',baseline[i]['artifact_sha256'])
        if not np.array_equal(a['frames'],b['frames']) or not np.array_equal(a['joint_names'],b['joint_names']):raise ValueError('inconsistent source')
        p=a['root_wxyz_xyz'];q=b['root_wxyz_xyz'];qa=Rotation.from_quat(p[:,[1,2,3,0]]);qb=Rotation.from_quat(q[:,[1,2,3,0]])
        d=np.linalg.norm(p[:,4:]-q[:,4:],axis=1)
        result.append({'window':i,'frames':a['frames'].tolist(),'mean_root_difference_m':float(d.mean()),
                       'max_root_difference_m':float(d.max()),'max_rotation_difference_rad':float((qa.inv()*qb).magnitude().max()),
                       'max_joint_difference_rad':float(np.abs(a['joint_pos']-b['joint_pos']).max())})
    out={'source_sha256':doc['source_sha256'],'anchor_audit_sha256':sha(HERE/'audit_report.json'),
         'baseline_report_sha256':sha(base/'report.json'),'script_sha256':sha(__file__),
         'first_window':result[0],'last_window':result[-1],
         'max_root_difference_m':max(x['max_root_difference_m'] for x in result),
         'windows':result,
         'warning':'Differences between two approximate solutions, not deviation from physical ground truth; a larger endpoint difference is not proof of accumulated drift. One subject/action, unknown source ground, no dynamics.'}
    with (HERE/'cross_deviation.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({k:out[k] for k in ('first_window','last_window','max_root_difference_m')}),flush=True)
if __name__=='__main__':main()
