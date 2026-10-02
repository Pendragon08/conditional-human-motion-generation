"""All-reference numerical comparison against unchanged official motion library."""
import ast
import json
import torch
import numpy as np
from r3_sim2sim_completed_mujoco import TASKS, ROOT, RESULT, base

source=ROOT/'third_party/DreamControl/Training/source/isaaclab_tasks/isaaclab_tasks/utils/motion_lib/motion_lib_base.py'
tree=ast.parse(source.read_text())
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MotionLibBase')
klass.body=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in ['get_motion_state','_calc_frame_blend']]
ns={'torch':torch,'Tensor':torch.Tensor,'slerp':base.Reference('Jump').slerp}
exec(compile(ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[])),str(source),'exec'),ns)
rows=[]
for task in TASKS:
    ref=base.Reference(task);lib=ns['MotionLibBase']()
    for name,v in ref.a.items():setattr(lib,name,torch.as_tensor(v))
    errors={k:0. for k in ['pos','quat','joints','keypts','goal']};mismatch=0
    for i in range(len(ref.a['_motion_lengths'])):
        for t in [-1.,-.7,0.,.02,.1,.2,.4,.5,1.,1.23,2.,3.49,5.1,7.,9.98,10.98]:
            e=lib.get_motion_state(torch.tensor([i]),torch.tensor([t],dtype=torch.float32));a=ref.state(i,t)
            for name,official in [('pos','root_pos'),('quat','root_rot'),('joints','dof_pos'),('keypts','global_keypts')]:
                errors[name]=max(errors[name],float(abs(a[name]-e[official][0].numpy()).max()))
            errors['goal']=max(errors['goal'],float(abs(a['goal']-(e['grab_pos']+e['offsets'])[0].numpy()).max()))
            mismatch+=bool(a['closed'])!=bool(e['is_closed'][0])
    rows.append({'task':task,'reference_count':len(ref.a['_motion_lengths']),'times_per_reference':16,
        'maximum_absolute_errors':errors,'event_phase_mismatches':mismatch,
        'passed':max(errors.values())<2e-4 and mismatch==0})
RESULT.mkdir(parents=True,exist_ok=True)
(RESULT/'reference_interpolation_audit.json').write_text(json.dumps({'rows':rows,'passed':all(x['passed'] for x in rows)},indent=2))
print('COMPLETED_REFERENCE_INTERPOLATION_AUDIT',json.dumps(rows),flush=True)
assert all(x['passed'] for x in rows)
