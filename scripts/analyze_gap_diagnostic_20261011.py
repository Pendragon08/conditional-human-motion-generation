"""CPU analysis of frozen first-attempt diagnostic; descriptive, not benchmark superiority."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
args=p.parse_args()
base=Path(args.input)
rows=[]
sens=[]

def val(x):
    return float(x) if np.isfinite(x) else None

for seed in (11,12,13):
    folder=base/('seed_'+str(seed))
    r=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
    file=folder/'first_attempts.npz'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==r['sha256']
    d=np.load(file)
    assert 'v3: generated observation reference' in r['alignment']
    actual=d['actual']; ref=d['reference']; mask=d['valid']
    force=d['contact_force']; dt=r['dt']; names=r['body_names']
    feet=[names.index(x) for x in ['L_Ankle','L_Toe','R_Ankle','R_Toe']]
    track=np.linalg.norm(actual-ref,axis=-1).mean(axis=-1)*1000
    local=np.linalg.norm((actual-actual[:,:,0:1])-(ref-ref[:,:,0:1]),axis=-1).mean(axis=-1)*1000
    av=np.linalg.norm(np.diff(actual,axis=0),axis=-1).mean(axis=-1)/dt
    rv=np.linalg.norm(np.diff(ref,axis=0),axis=-1).mean(axis=-1)/dt
    foot_speed=np.linalg.norm(np.diff(actual[:,:,feet,:2],axis=0),axis=-1)/dt
    foot_f=np.linalg.norm(force[:,:,feet],axis=-1)
    valid_pair=mask[1:] & mask[:-1]
    # Boundary frame k compares ref[k] with ref[k-1]; excludes first prefix.
    b=d['boundary'][1:]
    for i in range(r['environments']):
        good=mask[:,i].copy(); good[:30]=False
        pair=valid_pair[:,i].copy(); pair[:30]=False
        t=track[good,i]; l=local[good,i]
        if len(t)<10:
            row=dict(seed=seed,environment=i,prompt=r['prompts'][i],
                terminated=bool(d['terminated'][i]),frames=int(d['lengths'][i]),
                quality_frames=len(t),mean_gmpjpe_mm=None,p95_gmpjpe_mm=None,
                mean_lmpjpe_mm=None,contact_proxy_rate=None,quality_gate=False,
                diagnostic_insufficient=True)
            rows.append(row)
            continue
        persistent=(foot_f[:-1,i]>10)&(foot_f[1:,i]>10)&pair[:,None]
        exceed=persistent&(foot_speed[:,i]>0.5)
        denom=int(persistent.sum())
        rate=float(exceed.sum()/denom) if denom else None
        bm=pair&b; im=pair&~b
        mean=val(t.mean()); tail=val(np.quantile(t,.95))
        survives=not bool(d['terminated'][i])
        tracking_gate=survives and mean<=100 and tail<=200
        quality_gate=tracking_gate and rate is not None and rate<=.05
        row=dict(seed=seed,environment=i,prompt=r['prompts'][i],
            terminated=not survives,frames=int(d['lengths'][i]),quality_frames=len(t),
            mean_gmpjpe_mm=mean,p95_gmpjpe_mm=tail,mean_lmpjpe_mm=val(l.mean()),
            contact_proxy_rate=rate,contacted_body_frame_pairs=denom,
            ref_boundary_speed_mps=val(rv[bm,i].mean()) if bm.any() else None,
            ref_internal_speed_mps=val(rv[im,i].mean()) if im.any() else None,
            actual_boundary_speed_mps=val(av[bm,i].mean()) if bm.any() else None,
            actual_internal_speed_mps=val(av[im,i].mean()) if im.any() else None,
            boundary_gmpjpe_mm=val(track[1:,i][bm].mean()) if bm.any() else None,
            internal_gmpjpe_mm=val(track[1:,i][im].mean()) if im.any() else None,
            tracking_gate=tracking_gate,quality_gate=quality_gate,diagnostic_insufficient=False)
        rows.append(row)
        for fth in [5,10,20]:
            pm=(foot_f[:-1,i]>fth)&(foot_f[1:,i]>fth)&pair[:,None]
            for vth in [.25,.5,1.0]:
                proxy=float((pm&(foot_speed[:,i]>vth)).sum()/pm.sum()) if pm.any() else None
                for mth,pth in [(50,100),(100,200),(150,300)]:
                    sens.append(dict(seed=seed,environment=i,force_n=fth,speed_mps=vth,
                        mean_error_mm=mth,tail_error_mm=pth,
                        pass_gate=bool(survives and mean<=mth and tail<=pth and
                            proxy is not None and proxy<=.05)))

fields=list(dict.fromkeys(k for row in rows for k in row))
with (base/'per_clip.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
with (base/'threshold_sensitivity.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(sens[0]));w.writeheader();w.writerows(sens)

def summary(rs):
    adequate=[r for r in rs if not r['diagnostic_insufficient']]
    surv=[r for r in adequate if not r['terminated']]
    mean=lambda field,subset=adequate: val(np.mean([r[field] for r in subset if r.get(field) is not None]))
    paired=lambda a,b: val(np.mean([r[a]-r[b] for r in adequate if r.get(a) is not None and r.get(b) is not None]))
    return dict(n=len(rs),completed_without_termination=sum(not r['terminated'] for r in rs),
        terminated=sum(r['terminated'] for r in rs),adequate_quality_records=len(adequate),
        tracking_gate_accepted=sum(bool(r.get('tracking_gate')) for r in rs),
        quality_proxy_gate_accepted=sum(r['quality_gate'] for r in rs),
        survivors_rejected_by_tracking=sum(not r.get('tracking_gate',False) for r in surv),
        survivors_rejected_by_combined_gate=sum(not r['quality_gate'] for r in surv),
        mean_gmpjpe_mm=mean('mean_gmpjpe_mm'),mean_p95_gmpjpe_mm=mean('p95_gmpjpe_mm'),
        survivor_mean_gmpjpe_mm=mean('mean_gmpjpe_mm',surv),
        survivor_mean_contact_speed_proxy_rate=mean('contact_proxy_rate',surv),
        reference_boundary_minus_internal_speed_mps=paired('ref_boundary_speed_mps','ref_internal_speed_mps'),
        actual_boundary_minus_internal_speed_mps=paired('actual_boundary_speed_mps','actual_internal_speed_mps'),
        boundary_minus_internal_gmpjpe_mm=paired('boundary_gmpjpe_mm','internal_gmpjpe_mm'))

result=dict(protocol='frozen thresholds; v3 observation/action-aligned diagnostic; all first attempts; no source/model/policy changes',
    scope='CLoSD SMPL character T2M; not real robot certification or explicit target retention',
    thresholds=dict(mean_gmpjpe_mm=100,p95_gmpjpe_mm=200,force_n=10,speed_mps=.5,max_proxy_rate=.05),
    per_seed={str(s):summary([r for r in rows if r['seed']==s]) for s in (11,12,13)},
    pooled=summary(rows),
    warnings=['Contact rigid-body center speed is a proxy, not surface slip ground truth',
        'Operational thresholds are predeclared diagnostic choices, not a universal benchmark',
        'Boundary/internal association does not establish causality',
        'No measured text success, hard robot torque guarantee or downstream policy gain'])
(base/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False))
