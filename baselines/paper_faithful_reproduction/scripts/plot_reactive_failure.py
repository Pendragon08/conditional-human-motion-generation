"""Plot preserved training diagnostics; loss is not a task-success metric."""
import argparse
import json
from pathlib import Path
import tarfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--archive',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with tarfile.open(a.archive) as tar:
    def read(name):
        return [json.loads(line) for line in tar.extractfile('training_fp32_v3/'+name).read().decode().splitlines()]
    metrics=read('metrics.jsonl')
    health=read('health_rank0.jsonl')
fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
loss=[r for r in metrics if 'loss' in r]
axs[0].plot([r['optimizer_updates'] for r in loss],[r['loss'] for r in loss],color='#23668c',linewidth=.9)
axs[0].set_ylabel('Logged training loss\n(interval aggregates)')
finite=[r for r in health if r['grad_norm_pre_clip_fp64'] is not None]
axs[1].semilogy([r['optimizer_updates'] for r in finite],[r['grad_norm_pre_clip_fp64'] for r in finite],color='#bd4f35',linewidth=.7)
axs[1].set_ylabel('Pre-clipping gradient norm\n(FP64 reduction, each update)')
axs[1].set_xlabel('Validated optimizer updates (not checkpoint availability)')
for ax in axs:
    ax.axvline(10041,color='#367b4b',linestyle='--',label='Last saved state: 10041')
    ax.axvline(16871,color='#a42e35',linestyle=':',label='16871 valid updates; next attempt rejected')
    ax.grid(alpha=.2)
axs[0].legend(fontsize=8,loc='upper left')
fig.suptitle('ReactiveBFM v3: preserved numerical failure, 2026-10-23\nUnsaved updates after 10041 cannot be restored',fontsize=13)
fig.savefig(a.output/'reactive_v3_failure_curve.png',dpi=180)
fig.savefig(a.output/'reactive_v3_failure_curve.pdf')
summary={'validated_updates_before_failure':health[-1]['optimizer_updates'],
         'last_saved_optimizer_updates':10041,'unsaved_updates':health[-1]['optimizer_updates']-10041,
         'failure_reason':health[-1]['reason'],'finite_gradient_points':len(finite),
         'max_finite_pre_clip_gradient_norm':max(r['grad_norm_pre_clip_fp64'] for r in finite),
         'metric_rows':len(metrics),'raw_record_source':a.archive.name}
(a.output/'failure_curve_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
