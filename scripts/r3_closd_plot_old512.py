"""Plot existing 512-env training scalars; these are not new 3072 results."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
out=root/'results/closd_old512_training_curves_20261006'
with gzip.open(str(out/'all_scalars.json.gz'),'rt') as f:data=json.load(f)
tags=['rewards0/iter','episode_lengths/iter','a_loss','c_loss','clip_frac','disc_reward_mean']
fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
summary={}
for ax,tag in zip(axes.flat,tags):
    rows=data['scalars'][tag];x=np.array([r['step'] for r in rows]);y=np.array([r['value'] for r in rows])
    assert np.isfinite(y).all()
    window=100;rolling=np.convolve(y,np.ones(window)/window,mode='valid')
    ax.plot(x,y,color='#cad2de',lw=.6,label='Raw logged values')
    ax.plot(x[window-1:],rolling,color='#246fa8',lw=1.8,label='Trailing 100-point mean')
    ax.set_title(tag);ax.set_xlabel('Logged epoch');ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    summary[tag]={'points':len(rows),'first_epoch':int(x[0]),'last_epoch':int(x[-1]),
        'first':float(y[0]),'last':float(y[-1]),
        'first100_mean':float(y[:100].mean()),'last100_mean':float(y[-100:].mean()),
        'min':float(y.min()),'max':float(y.max())}
    with (out/(tag.replace('/','_')+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['step','wall_time','value']);w.writeheader();w.writerows(rows)
axes[0,0].legend(fontsize=8)
fig.suptitle('CLoSD: archived 512-env fine-tune (shutdown exit 139; final evaluation degraded)',fontsize=13)
for suffix in ['png','svg']:fig.savefig(out/('old512_training_curves.'+suffix),dpi=160)
plt.close(fig)
(out/'scalar_summary.json').write_text(json.dumps({'source_sha256':data['source_sha256'],
    'interpretation_boundary':'Training proxies; not 4096 official task success or new3072 training.',
    'statistics':summary},indent=2))
print(json.dumps(summary,indent=2))
