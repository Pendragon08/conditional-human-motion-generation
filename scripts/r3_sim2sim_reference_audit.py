"""Compare adapter interpolation to the unmodified official method AST.

Executes only named numerical functions from the local pinned source, not its
package imports or simulator. Every reference is checked at 12 phase values.
"""
import ast
import json
from pathlib import Path
import types
import numpy as np
import torch
from r3_sim2sim_mujoco import Reference, ROOT

dc = ROOT / 'third_party/DreamControl/Training'
source = dc / 'source/isaaclab_tasks/isaaclab_tasks/utils/motion_lib/motion_lib_base.py'
rotations = dc / 'isaac_utils/isaac_utils/rotations.py'
tree = ast.parse(source.read_text())
klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionLibBase')
klass.body = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in ['get_motion_state', '_calc_frame_blend']]
slerp = next(n for n in ast.parse(rotations.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'slerp')
slerp.decorator_list = []
module = ast.Module(body=[slerp, klass], type_ignores=[])
ns = {'torch': torch, 'Tensor': torch.Tensor}
exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), ns)
rows = []
for task in ['Kick', 'Punch']:
    ref = Reference(task)
    lib = ns['MotionLibBase']()
    for name, value in ref.a.items():
        setattr(lib, name, torch.as_tensor(value))
    errors = {k: 0. for k in ['pos', 'quat', 'joints', 'keypts', 'goal']}
    closed_mismatch = 0
    for i in range(len(ref.a['_motion_lengths'])):
        for t in [0., .02, .1, .2, .5, 1., 1.23, 2., 3.49, 5.1, 7., 9.98]:
            expected = lib.get_motion_state(torch.tensor([i]), torch.tensor([t], dtype=torch.float32))
            actual = ref.state(i, t)
            for name, official in [('pos', 'root_pos'), ('quat', 'root_rot'), ('joints', 'dof_pos'), ('keypts', 'global_keypts')]:
                errors[name] = max(errors[name], float(np.max(np.abs(actual[name]-expected[official][0].numpy()))))
            errors['goal'] = max(errors['goal'], float(np.max(np.abs(actual['goal']-(expected['grab_pos']+expected['offsets'])[0].numpy()))))
            closed_mismatch += bool(actual['closed']) != bool(expected['is_closed'][0])
    row = {'task': task, 'reference_count': len(ref.a['_motion_lengths']), 'times_per_reference': 12,
           'maximum_absolute_errors': errors, 'event_phase_mismatches': closed_mismatch,
           'passed': max(errors.values()) < 2e-4 and closed_mismatch == 0}
    rows.append(row)
out = ROOT / 'results/sim2sim_mujoco/reference_interpolation_audit.json'
out.write_text(json.dumps({'rows': rows, 'passed': all(r['passed'] for r in rows)}, indent=2))
print('REFERENCE_INTERPOLATION_AUDIT', json.dumps(rows))
if not all(r['passed'] for r in rows):
    raise SystemExit(1)
