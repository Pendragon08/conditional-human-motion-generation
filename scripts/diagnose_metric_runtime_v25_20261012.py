"""Read-only compare collector and verifier numerical runtimes, no simulation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import run_reset_alignment_checks_v4 as runtime

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/metric_runtime_diagnostic_v25_20261012'


def main():
    assert not OUT.exists(), 'Preserve diagnostic attempts'
    code="""import json,sys,numpy as np
from pathlib import Path
from physics_guided_metric_v23_20261012 import metrics
p=Path(sys.argv[1]); meta=json.loads((p/'metadata.json').read_text());thresholds=json.loads(Path(sys.argv[2]).read_text())['thresholds']
saved=json.loads((p/'metrics.json').read_text())
with np.load(p/'trajectories.npz',allow_pickle=False) as d:arrays={k:d[k] for k in d.files}
computed=[metrics(arrays,i,meta,thresholds) for i in range(len(saved))]
print(json.dumps(dict(numpy_version=np.__version__,numpy_path=np.__file__,maximum_mean_slip_difference_m_s=max(abs(a['surface_slip_proxy_mean_m_s']-b['surface_slip_proxy_mean_m_s']) for a,b in zip(computed,saved)),first_saved=saved[0]['surface_slip_proxy_mean_m_s'],first_computed=computed[0]['surface_slip_proxy_mean_m_s'],classifications_match=all(all(a[k]==b[k] for k in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')) for a,b in zip(computed,saved)))))"""
    args=[sys.executable,'-c',code,str(ROOT/'results/contact_com_validation_v24_20261012/raw/val_planned_s11'),str(ROOT/'results/contact_com_validation_v24_20261012/frozen_thresholds.json')]
    result={}
    for name,env in [('wrapper_default',os.environ.copy()),('collector_overlay',runtime.environment())]:
        env['PYTHONPATH']=str(ROOT/'scripts')+':'+env.get('PYTHONPATH','')
        process=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
        result[name]=json.loads(process.stdout)
    OUT.mkdir();(OUT/'diagnostic.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
