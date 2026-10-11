"""Locate changed initial/policy states after export-route intervention."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'results/contact_sensor_v10_20261011/raw/repeat_a'
B=ROOT/'results/contact_cpu_export_v11_20261012/raw/repeat_a'
report=dict(initial={},series={},metadata={})
with np.load(A/'initial_states.npz',allow_pickle=False) as a,np.load(B/'initial_states.npz',allow_pickle=False) as b:
    report['initial']={k:float(np.max(np.abs(a[k]-b[k]))) for k in a.files}
for filename in ['trajectories.npz','sensor_diagnostic.npz']:
    with np.load(A/filename,allow_pickle=False) as a,np.load(B/filename,allow_pickle=False) as b:
        report['series'][filename]={}
        for k in a.files:
            if k in ['valid','terminated','reference_frame']:continue
            delta=np.max(np.abs(a[k]-b[k]).reshape(a[k].shape[0],-1),axis=1)
            bad=np.flatnonzero(delta>1e-6)
            report['series'][filename][k]=dict(maximum=float(delta.max()),first_over_1e6=int(bad[0]) if len(bad) else None,first20_max=delta[:20].tolist())
ma=json.loads((A/'metadata.json').read_text());mb=json.loads((B/'metadata.json').read_text())
for k in ['original_goal_sha256','generation_noise_sha256','generation_prefix_sha256','motion_ids_before_first_action','motion_start_times_before_first_action','global_offset_before_first_action','foot_asset_sha256','full_reset_calls_before_first_action','upstream_warmup_diffusion_calls','reset_initial_root','manual_reset_state_sha256','sim_params']:
    report['metadata'][k]=dict(equal=ma.get(k)==mb.get(k),old=ma.get(k),new=mb.get(k))
report.update(measurement_only_invariance_supported=False,overleaf_results_sync=False)
out=ROOT/'results/contact_cpu_export_v11_20261012/route_divergence_diagnostic.json'
out.write_text(json.dumps(report,indent=2))
print(json.dumps(dict(initial=report['initial'],first_divergence={f:{k:x['first_over_1e6'] for k,x in v.items()} for f,v in report['series'].items()},metadata_equal={k:v['equal'] for k,v in report['metadata'].items()}),indent=2))
