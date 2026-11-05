"""Prepare full-timeline human/G1 curves for five caption reviews.

The curves aid manual semantic review; they do not automatically accept text.
Approximate fitted human and robot body frames are independently aligned at
each sample, so ground contact, absolute travel, and true clock are unknown.
"""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'ssm_new6_20hz_resume_20261101';HUMAN=ROOT/'ssm_new6_human_20261101';HERE=Path(__file__).resolve().parent
IDS=('004028','006749','007129','008894','009466')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def curves(p):
    across=(p[:,2]-p[:,1])+(p[:,17]-p[:,16]);across[:,2]=0
    norm=np.linalg.norm(across,axis=1)
    if np.any(norm<1e-6):raise ValueError('body lateral direction degenerates')
    lateral=across/norm[:,None];forward=np.stack([-lateral[:,1],lateral[:,0],np.zeros(len(p))],axis=1)
    def relative(a,b,axis):
        v=p[:,a]-p[:,b]
        return v[:,2] if axis=='z' else np.sum(v*(lateral if axis=='lat' else forward),axis=1)
    return np.column_stack([relative(20,16,'z'),relative(21,17,'z'),relative(18,16,'z'),relative(19,17,'z'),
        relative(7,1,'lat'),relative(8,2,'lat'),relative(7,1,'fwd'),relative(8,2,'fwd'),
        relative(7,1,'z'),relative(8,2,'z'),p[:,0,2]-p[0,0,2],
        relative(10,7,'z'),relative(11,8,'z'),relative(20,16,'lat'),relative(21,17,'lat'),
        relative(20,16,'fwd'),relative(21,17,'fwd')])
def main():
    report=json.loads((SOURCE/'report.json').read_text());audit=json.loads((SOURCE/'audit.json').read_text())
    assert sha(SOURCE/'report.json')==audit['source_report_sha256'] and sha(SOURCE/'records.jsonl')==report['records_sha256']
    records={r['id']:r for r in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())};cases=[]
    labels=('left wrist height','right wrist height','left elbow height','right elbow height',
      'left ankle lateral','right ankle lateral','left ankle forward','right ankle forward',
      'left ankle height','right ankle height','pelvis height change',
      'left foot-point height','right foot-point height','left wrist lateral','right wrist lateral',
      'left wrist forward','right wrist forward')
    groups=((0,1),(2,3),(4,5),(6,7),(8,9),(10,),(11,12),(13,14),(15,16))
    for sid in IDS:
        rec=records[sid]; path=SOURCE/rec['artifact']; human_path=HUMAN/sid/'human_rawfps.npz'
        assert rec['status']=='technical_preview' and sha(path)==rec['artifact_sha256']
        hr=json.loads((HUMAN/sid/'human_report.json').read_text());assert sha(human_path)==hr['artifact_sha256']
        with np.load(path,allow_pickle=False) as f:t=f['sample_times'];robot=f['points_z_up_world']
        with np.load(human_path,allow_pickle=False) as f:st=f['source_times'];raw=f['human']
        if t[0]<st[0] or t[-1]>st[-1]:raise ValueError('human interpolated outside source')
        human=np.column_stack([np.interp(t,st,raw.reshape(len(st),-1)[:,j]) for j in range(66)]).reshape(-1,22,3)
        h=curves(human);g=curves(robot)
        curvefile=HERE/(sid+'_curves.npz')
        with curvefile.open('xb') as f:np.savez_compressed(f,sample_times=t,human=h,robot=g,labels=np.asarray(labels))
        fig,axes=plt.subplots(3,3,figsize=(17,11));axes=axes.ravel()
        for ax,columns in zip(axes,groups):
            for k in columns:
                ax.plot(t,h[:,k],label='human '+labels[k],lw=1.1)
                ax.plot(t,g[:,k],label='robot '+labels[k],lw=1.1,ls='--')
            ax.axhline(0,color='black',lw=.3);ax.set_xlabel('nominal seconds');ax.set_ylabel('meters');ax.legend(fontsize=6)
        text=' | '.join(c['caption'] for c in rec['captions'])
        fig.suptitle(sid+'  '+text,fontsize=9,wrap=True);fig.tight_layout()
        img=HERE/(sid+'.png');
        if img.exists():raise FileExistsError(img)
        fig.savefig(img,dpi=100);plt.close(fig)
        cases.append({'id':sid,'state_sha256':sha(path),'human_source_sha256':sha(human_path),'curves_sha256':sha(curvefile),
            'image_sha256':sha(img),'sampled_frames':len(t),'caption_records':rec['captions'],
            'human_range_per_curve_m':np.ptp(h,axis=0).tolist(),'robot_range_per_curve_m':np.ptp(g,axis=0).tolist(),
            'semantic_reviewed':False,'training_ready':False})
    result={'input_report_sha256':sha(SOURCE/'report.json'),'input_audit_sha256':sha(SOURCE/'audit.json'),
        'script_sha256':sha(Path(__file__)),'labels':labels,'cases':cases,'training_ready':False,
        'limitations':['Body-frame axes derive from hips/shoulders; interpolation of fitted human points is approximate.',
           'Plots omit mesh, depth context, ground, contact and forces; curves alone do not prove semantics or true time.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'cases':len(cases),'frames':sum(x['sampled_frames'] for x in cases),'training_ready':False}))
if __name__=='__main__':main()
