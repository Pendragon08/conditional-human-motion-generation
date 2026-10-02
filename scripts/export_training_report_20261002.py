"""Read-only export of completed baseline TensorBoard events for teacher report."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
R=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
O=Path('/data/code/ouyanghao/oyh/humanoid-combat-motion')
OUT=R/'artifacts/report_training_20261002';OUT.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache')
import json,hashlib,csv,gzip,shutil,datetime,statistics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

jobs={
'Kick':[(O/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a',0,1999)],
'Punch':[(R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_07-24-06-Punch-a',0,1599),(R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-29_12-37-46-Punch-a',1600,1999)],
'Jump':[(R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-10-01_14-01-43-Jump-a',0,1999)],
'Sit':[(R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-10-01_16-54-27-Sit-a',0,1999)],
'Pick':[(R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-10-01_19-59-36-Pick-a',0,1999)],
'Bimanual-Pick':[(R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-10-02_00-21-24-Bimanual-Pick-a',0,1999)]}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
all_summary=[]
for task,parts in jobs.items():
    td=OUT/task;td.mkdir(exist_ok=True);combined={};provenance=[];raw_rows=[]
    for run,lo,hi in parts:
        ea=EventAccumulator(str(run),size_guidance={'scalars':0});ea.Reload();tags=ea.Tags()['scalars']
        for source in sorted(run.glob('events.out.tfevents.*')):
            dst=td/'events'/run.name/source.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
            provenance.append({'source':str(source),'local_relative':str(dst.relative_to(OUT)),'bytes':source.stat().st_size,'sha256':sha(source),'included_iterations':[lo,hi],'raw_event_file_includes_rolled_back_steps':task=='Punch' and lo==0})
        for tag in tags:
            for x in ea.Scalars(tag):
                raw_rows.append([task,run.name,tag,x.step,x.wall_time,x.value])
                if tag.endswith('/time'):continue # these x axes are elapsed seconds, not iterations
                if lo<=x.step<=hi:combined.setdefault(tag,{})[x.step]=(x.wall_time,float(x.value))
    with gzip.open(td/'all_scalars_raw.csv.gz','wt',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['task','run','tag','step_or_time','wall_time','value']);w.writerows(raw_rows)
    with (td/'iteration_scalars.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['iteration','tag','value','wall_time'])
        for tag,rows in sorted(combined.items()):
            for step,(wall,value) in sorted(rows.items()):w.writerow([step,tag,value,wall])
    metrics={}
    for tag,rows in combined.items():
        seq=sorted(rows);vals=np.array([rows[s][1] for s in seq]);assert np.isfinite(vals).all()
        metrics[tag]={'points':len(seq),'first_step':seq[0],'last_step':seq[-1],'first':float(vals[0]),'last':float(vals[-1]),'first100_mean':float(vals[:100].mean()),'last100_mean':float(vals[-100:].mean()),'last100_std':float(vals[-100:].std(ddof=1)),'min':float(vals.min()),'max':float(vals.max())}
    panels=[('Train/mean_reward','Mean episode reward','linear'),('Loss/value_function','Value-function loss','log'),('Loss/surrogate','PPO surrogate loss','linear'),('Loss/entropy','Action-distribution entropy','linear'),('Train/mean_episode_length','Mean episode length','linear'),('Policy/mean_noise_std','Policy mean noise std','linear')]
    fig,axs=plt.subplots(2,3,figsize=(16,8),dpi=110);fig.patch.set_facecolor('#f5f8fb')
    for ax,(tag,title,scale) in zip(axs.flat,panels):
        rows=combined[tag];steps=np.array(sorted(rows));values=np.array([rows[int(s)][1] for s in steps]);assert len(steps)==2000 and steps[0]==0 and steps[-1]==1999
        ax.plot(steps,values,lw=.65,color='#9bbdc6',alpha=.65,label='Raw logger values')
        # Causal 50-iteration moving average; never smooth across restart boundary.
        for begin,end in ([(0,1599),(1600,1999)] if task=='Punch' else [(0,1999)]):
            keep=(steps>=begin)&(steps<=end);xs=steps[keep];vs=values[keep]
            avg=np.array([vs[max(0,i-49):i+1].mean() for i in range(len(vs))]);ax.plot(xs,avg,lw=1.6,color='#127b86',label='Trailing 50-iteration mean' if begin==0 else None)
        if task=='Punch':ax.axvline(1600,color='#c86543',ls='--',lw=1);ax.text(.72,.9,'Resume at 1600',transform=ax.transAxes,fontsize=8,color='#a74731')
        if scale=='log':ax.set_yscale('log')
        ax.set_title(title,fontsize=12);ax.set_xlabel('PPO iteration');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs.flat[0].legend(fontsize=8);fig.suptitle(f'DreamControl {task} | official baseline | 8192 environments | seed 42 | 2000 iterations',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(td/'training_curves.svg');fig.savefig(td/'training_curves.png');plt.close(fig)
    cfg=parts[-1][0]/'params'
    if cfg.exists():
        for p in cfg.glob('*.yaml'):shutil.copy2(p,td/p.name)
    summary={'task':task,'status':'completed_baseline_training','iterations':2000,'num_envs':8192,'steps_range':[0,1999],'seed':42,'source_events':provenance,'metrics':metrics,'smoothing':'Raw data retained; trailing 50-iteration mean shown separately, split at Punch iteration1600 resume','aggregation':'Punch initial steps0..1599 plus resumed1600..1999; discarded rolled-back initial1600..1618 from combined curves; original event and raw CSV retained','interpretation':'Training reward and loss are logger statistics, not evaluation success or physical quality; reward scales differ across tasks'}
    (td/'summary.json').write_text(json.dumps(summary,indent=2));all_summary.append(summary)
    print('LOGGER_EXPORT_COMPLETE',task,metrics['Train/mean_reward']['first100_mean'],metrics['Train/mean_reward']['last100_mean'],flush=True)
(OUT/'summary.json').write_text(json.dumps({'status':'COMPLETE','exported_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':all_summary,'scope':'six official completed baselines only, no research comparison or active training logs'},indent=2))
print('TRAINING_REPORT_EXPORT_COMPLETE',len(all_summary),flush=True)
