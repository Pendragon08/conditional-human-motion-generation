"""Adapt archived 29-body + binary-hand inputs for the upstream Viser viewer.

No fitting, diffusion, policy or simulation is performed here.
"""
import ast
import copy
import hashlib
import json
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import jaxlie
import numpy as np
import yourdfpy

R = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
U = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
OUT = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
BODY = U/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf'
HAND = U/'Sim2Real/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
MANIFEST = R/'stage3_supported_references_20261104/mixed2000_final_v1/native_motion_manifest.json'
assert sha(BODY) == '572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95'
assert sha(MANIFEST) == '6d35b617780968b45ef9d05c2014df17708d6b45bb7158438a13ec6c49df36cf'
OUT.mkdir(exist_ok=True)
assert not (OUT/'prepared.json').exists()
body = ET.parse(BODY).getroot()
hand = ET.parse(HAND).getroot()
for tree, base in [(body, BODY.parent), (hand, HAND.parent)]:
    for mesh in tree.findall('.//mesh'):
        path = (base/mesh.get('filename')).resolve()
        assert path.is_file(), path
        mesh.set('filename', str(path))
# Keep every body transform from the validated training URDF. Replace only its
# fixed rubber hands with the original repository's articulated hand subtrees.
for element in list(body):
    if element.get('name') in ('left_rubber_hand', 'right_rubber_hand',
                               'left_hand_palm_joint', 'right_hand_palm_joint'):
        body.remove(element)
for element in hand:
    if element.tag in ('link', 'joint') and element.get('name', '').startswith(('left_hand_', 'right_hand_')):
        body.append(copy.deepcopy(element))
adapted = OUT/'g1_29body_14hand_viewer.urdf'
ET.ElementTree(body).write(adapted, encoding='utf-8', xml_declaration=True)
u = yourdfpy.URDF.load(str(adapted), load_meshes=False)
reference_u = yourdfpy.URDF.load(str(BODY), load_meshes=False)
names = list(u.actuated_joint_names)
assert len(names) == 43 and len(set(names)) == 43
# Extract the existing evaluation hand commands without importing Isaac Sim.
env_source = R/'stage3_pick29_20261103/src/stage3_pick29_env.py'
tree = ast.parse(env_source.read_text())
commands = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ('left_hand', 'right_hand'):
                commands[target.id] = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords
                                      if kw.arg in ('open_command_expr', 'close_command_expr')}
assert set(commands) == {'left_hand', 'right_hand'}
selected = {'pick':157, 'pour':11000101, 'bimanual_box_lift':12000063, 'box_push_pull':13000388}
manifest = json.loads(MANIFEST.read_text())
clips, roots, quats, full_q, errors = [], [], [], [], []
cursor = 0
for task, cid in selected.items():
    row = next(x for x in manifest['records'] if x['task'] == task and x['candidate_id'] == cid)
    p = Path(row['source']); assert sha(p) == row['source_sha256']
    z = np.load(p, allow_pickle=False)
    q = z['q']; n = len(q); hands = z['hand_closed']; joint_order = row['body_joint_order']
    assert q.shape == (n,29) and hands.shape == (n,2) and np.all(np.isin(hands,[0,1]))
    expanded = np.zeros((n,43), dtype=np.float64)
    expanded[:,[names.index(name) for name in joint_order]] = q
    for side, col in [('left',0),('right',1)]:
        for name in [x for x in names if x.startswith(side+'_hand_')]:
            for field,closed in [('open_command_expr',False),('close_command_expr',True)]:
                matches = [v for pattern,v in commands[side+'_hand'][field].items() if re.fullmatch(pattern,name)]
                assert len(matches) == 1, (name,field,matches)
                expanded[hands[:,col] == int(closed),names.index(name)] = matches[0]
    # Check all body frames at every exported frame, independently of Viser.
    maximum = 0.
    for f in range(n):
        reference_u.update_cfg(dict(zip(joint_order,q[f])))
        u.update_cfg(dict(zip(names,expanded[f])))
        for name in ['left_wrist_yaw_link','right_wrist_yaw_link','left_ankle_roll_link','right_ankle_roll_link']:
            maximum = max(maximum,float(np.max(np.abs(u.get_transform(name)-reference_u.get_transform(name)))))
    assert maximum < 1e-10
    # Upstream adds 0.035 m only for viewing. Compensate in the converted copy.
    roots.append(z['root_pos'].astype(np.float64) - np.array([0,0,.035]))
    quats.append(z['root_quat']); full_q.append(expanded)
    slug = f'{len(clips)+1:02d}_{task}'
    clips.append({'task':task,'candidate_id':cid,'source':str(p),'source_sha256':sha(p),
                  'slug':slug,'start':cursor,'frames':n,'hand_closed':hands.astype(int).tolist(),
                  'body_fk_matrix_max_error':maximum,
                  'finger_joint_ranges':{name:[float(expanded[:,i].min()),float(expanded[:,i].max())]
                                         for i,name in enumerate(names) if '_hand_' in name}})
    cursor += n
data = {'global_pose':jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(np.concatenate(quats)), np.concatenate(roots)),
        'joints':np.concatenate(full_q)}
with (OUT/'all_four.pkl').open('xb') as f:pickle.dump(data,f)
provenance = {'status':'prepared','upstream_commit':'ec28b508313d6837de52c68247fbc8cf81fe4935',
              'original_viewer_sha256':sha(OUT/'upstream_visualize_trajectories.py'),
              'body_urdf_sha256':sha(BODY),'hand_urdf_sha256':sha(HAND),
              'hand_commands_source_sha256':sha(env_source),'adapted_urdf_sha256':sha(adapted),
              'converted_input_sha256':sha(OUT/'all_four.pkl'),'joint_names':names,'clips':clips,
              'scope':'Archived post-optimized reference with planned hands. Kinematic playback in original DreamControl Viser viewer; no object/contact simulation.'}
(OUT/'prepared.json').write_text(json.dumps(provenance,indent=2))
print(json.dumps({'status':'prepared','frames':cursor,'clips':len(clips),'body_fk_error':max(c['body_fk_matrix_max_error'] for c in clips)}))
