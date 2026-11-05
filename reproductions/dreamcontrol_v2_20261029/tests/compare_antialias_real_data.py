"""Compare experimental FIR and linear resampling on official archives.

Only known, hash-matching anomalies may use explicitly recorded object
quaternion normalization. This inspects differences, not motion validity.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

from normalize_object_quaternion import normalize_object_quaternions
from resample_antialias_omniretarget import resample_antialias
from resample_omniretarget import resample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archives',nargs='+',type=Path); p.add_argument('--repair-witness',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    witness=json.loads(args.repair_witness.read_text(encoding='utf-8'))
    repairs={r['member']:r['source_npz_sha256'] for r in witness['repairs']}
    if witness['refused']: raise ValueError('repair witness contains unresolved cases')
    rows=[]; errors=[]; repaired=[]
    for path in args.archives:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.endswith('.npz'): continue
                try:
                    payload=archive.read(member)
                    with np.load(io.BytesIO(payload),allow_pickle=False) as data:
                        q=data['qpos']; fps=float(data['fps'])
                    if member.filename in repairs:
                        if hashlib.sha256(payload).hexdigest()!=repairs[member.filename]:
                            raise ValueError('repair witness hash mismatch')
                        q,_=normalize_object_quaternions(q); repaired.append(member.filename)
                    a=resample(q,fps,max_seconds=None); b=resample_antialias(q,fps)
                    if a.shape!=b.shape: raise ValueError('resampling shape mismatch')
                    dot=np.abs(np.sum(a[:,:4]*b[:,:4],axis=1)).clip(0,1)
                    root_delta=np.linalg.norm(a[:,4:7]-b[:,4:7],axis=1)
                    joint_delta=np.abs(a[:,7:36]-b[:,7:36])
                    root_frame=int(np.argmax(root_delta))
                    joint_frame=int(np.unravel_index(np.argmax(joint_delta),joint_delta.shape)[0])
                    rows.append({'archive':path.name, 'member':member.filename, 'frames':len(a),
                                 'root_translation_max_m':float(root_delta.max()),
                                 'root_max_frame':root_frame,
                                 'root_max_near_boundary':bool(root_frame<10 or root_frame>=len(a)-10),
                                 'root_rotation_max_rad':float((2*np.arccos(dot)).max()),
                                 'joint_delta_mean_rad':float(joint_delta.mean()),
                                 'joint_delta_max_rad':float(joint_delta.max()),
                                 'joint_max_frame':joint_frame,
                                 'joint_max_near_boundary':bool(joint_frame<10 or joint_frame>=len(a)-10)})
                except (ValueError,KeyError,OSError,zipfile.BadZipFile) as e:
                    errors.append({'archive':path.name,'member':member.filename,'error':str(e)})
    if set(repaired)!=set(repairs):
        raise ValueError('not all known repaired cases were encountered')
    summaries={}
    for name in sorted({r['archive'] for r in rows}):
        subset=[r for r in rows if r['archive']==name]
        summaries[name]={'count':len(subset)}
        for key in ('root_translation_max_m','root_rotation_max_rad','joint_delta_mean_rad','joint_delta_max_rad'):
            x=np.array([r[key] for r in subset]); summaries[name][key]={'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'max':float(x.max())}
    report={'valid_motions':len(rows),'known_repairs':len(repaired),'errors':errors,'summaries':summaries,
            'largest_joint_differences':sorted(rows,key=lambda r:r['joint_delta_max_rad'],reverse=True)[:10],
            'largest_root_differences':sorted(rows,key=lambda r:r['root_translation_max_m'],reverse=True)[:10],
            'root_max_near_boundary_count':sum(r['root_max_near_boundary'] for r in rows),
            'joint_max_near_boundary_count':sum(r['joint_max_near_boundary'] for r in rows),
            'note':'FIR versus linear difference, not an error against ground truth; quaternion FIR is an approximation and physical validity is unverified.'}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'valid_motions':len(rows),'known_repairs':len(repaired),'error_count':len(errors)}))


if __name__=='__main__':
    main()
