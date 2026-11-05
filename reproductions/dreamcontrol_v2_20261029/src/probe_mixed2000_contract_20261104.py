import json,pathlib
import numpy as np
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
N=R/'stage3_supermimic29_20261103/native_universal_20261104'
W=R/'stage3_supported_references_20261104/bulk_v1'
P=R/'stage3_pick29_20261103/reference_scale500_20261104/final_library_v1/reference_bank.json'
pick=json.loads(P.read_text());new=[json.loads(x) for x in (W/'library_records.jsonl').read_text().splitlines()]
for label,ref in [('pick',pick['references'][0]),('new',new[0])]:
 print(label+'_manifest_record',json.dumps(ref))
 with np.load(ref['path'],allow_pickle=False) as z:
  print(label+'_arrays',json.dumps({k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}))
  for k in ['task','caption','metadata_utf8','joint_names']:
   if k in z:
    print(label+'_'+k,z[k].tobytes().decode() if k=='metadata_utf8' else z[k].tolist())
 print(label+'_proof',pathlib.Path(ref.get('native_proof',ref.get('postopt_report'))).read_text()[:16000])
 print(label+'_kinematic',pathlib.Path(ref['kinematic_report']).read_text()[:16000])
print('native_manifest_first',json.dumps(json.loads((N/'native_motion_manifest.json').read_text())['records'][0]))
for name in ['kinematics.py','robot.py']:
 text=(N/'source/supermimic'/name).read_text()
 print('source_'+name,text[:18000] if name=='kinematics.py' else text[:7000])
