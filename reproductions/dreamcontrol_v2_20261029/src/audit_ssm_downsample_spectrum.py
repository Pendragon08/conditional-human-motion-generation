"""Flag high-frequency energy before SSM 30→20Hz; diagnostic, not ground truth."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RECEIPTS=BASE/'ssm_receipt_audit_20261030/rows.jsonl'
OUT=BASE/'ssm_downsample_spectrum_20261030'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def powers(data,fps):
    x=np.asarray(data,dtype=np.float64)
    if x.ndim!=2 or len(x)<8 or not np.isfinite(x).all():raise ValueError('invalid spectrum signal')
    x=x-x.mean(axis=0,keepdims=True)
    spectrum=np.abs(np.fft.rfft(x*np.hanning(len(x))[:,None],axis=0))**2
    f=np.fft.rfftfreq(len(x),1/fps); power=spectrum.sum(axis=0)
    valid=power>1e-12
    if not valid.any():return {'pooled_high_fraction':0.0,'maximum_channel_fraction':0.0,'valid_channels':0}
    high=spectrum[f>=10].sum(axis=0)
    return {'pooled_high_fraction':float(high.sum()/power.sum()),
            'maximum_channel_fraction':float(np.max(high[valid]/power[valid])),
            'valid_channels':int(valid.sum())}

def main():
    receipts=[r for r in map(json.loads,RECEIPTS.open()) if not r['within_0_01_hz_of_20']]
    rows=[]
    for r in receipts:
        doc=json.loads(Path(r['receipt']).read_text()); target=Path(doc['smplx_file']); ref=Path(doc['reference_file'])
        if sha(target)!=doc['smplx_sha256'] or sha(ref)!=doc['reference_sha256']:
            raise ValueError('source changed')
        with np.load(target,allow_pickle=False) as x:
            trans=np.asarray(x['trans'],dtype=np.float64)[::r['stride']]
            pose=np.asarray(x['pose_body'],dtype=np.float64)[::r['stride']]
        if len(trans)!=len(pose) or pose.shape[1]!=63:raise ValueError('unexpected dimensions')
        # Relative rotations avoid naive differentiation through axis-angle wrap.
        n=len(pose); rot=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(n,21,3,3)
        delta=np.matmul(np.swapaxes(rot[:-1],-1,-2),rot[1:])
        omega=Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(n-1,63)*r['effective_fps']
        rows.append({'source_group':r['source_group'],'samples':n,'actual_sample_fps':r['effective_fps'],
                     'translation':powers(trans,r['effective_fps']), 'body_angular_velocity':powers(omega,r['effective_fps'])})
    if len(rows)!=13:raise ValueError('unexpected number of ~30Hz SSM sources')
    OUT.mkdir(mode=0o700)
    with (OUT/'rows.jsonl').open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    summary={'recordings':len(rows),
        'worst_translation_pooled_fraction':max(r['translation']['pooled_high_fraction'] for r in rows),
        'worst_translation_channel_fraction':max(r['translation']['maximum_channel_fraction'] for r in rows),
        'worst_body_omega_pooled_fraction':max(r['body_angular_velocity']['pooled_high_fraction'] for r in rows),
        'worst_body_omega_channel_fraction':max(r['body_angular_velocity']['maximum_channel_fraction'] for r in rows),
        'limitations':['Hann-window FFT energy at or above 10 Hz on legacy sampled data; leakage and finite-length effects apply.',
                       'High-frequency fractions are not true resampling errors, physics/contact outcomes or acceptance thresholds.',
                       'Only 13 SSM recordings; marker, caption, frame identity evidence inherited from previous audits.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
