from pathlib import Path
import importlib.util,json,sys
import numpy as np
s=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
plan=json.loads((s/'generation_pipeline_v5_launch/config.json').read_text())
sys.path.insert(0,str(Path(plan['sources']['postopt']['path']).parent))
from postopt_v2 import load_sample
spec=importlib.util.spec_from_file_location('diagnostic_fk',plan['fk']['path'])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fk=module.URDFFK(plan['urdf']['path'])
from scipy.spatial.transform import Rotation
for rp in sorted((s/'new_references_v5/pick').glob('cohort_*/postopt/*/report.json')):
    r=json.loads(rp.read_text())
    ap=rp.parent/'stage_a_decoded_realization.npz'
    if not ap.exists():continue
    raw=load_sample(r['source_sample'],r['source_sample_sha256'])
    with np.load(ap) as a:
        q,root,d=a['q'],a['root'],a['decoded_world']
        x=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,fk.actuated)
        e=np.linalg.norm(x-d,axis=-1)
        w=int(np.argmax(e[:,21]))
        f=raw['metadata']['grasp_frame']
        print(r['candidate_id'],'root-yaw deg',np.round(np.rad2deg(Rotation.from_quat(root[:,[1,2,3,0]]).as_euler('zyx')[::30,0]),1).tolist(),
              'largest mean errors',[(int(i),round(float(e[:,i].mean()),3)) for i in np.argsort(e.mean(0))[-5:][::-1]],
              'peak/wrist frame',w,'solved',np.round(x[w,21],3).tolist(),'decoded',np.round(d[w,21],3).tolist(),
              'grasp solved',np.round(x[f,21],3).tolist(),'grasp goal',np.round(raw['targets_world'][f,21],3).tolist(),
              'rmse',np.sqrt(np.mean(e**2)))
