"""Compare G1 keypoints from publisher URDFs and existing DreamControl URDF.

Checks a small fixed set of real OmniRetarget frames, in root-local space. It
does not compare meshes, dynamics or every model and joint configuration.
"""
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import torch
import pytorch_kinematics as pk

from motion_bridge import SOURCE_NAMES
from waist_lock_ik import FIT_LINKS, _chain, _positions


def publisher_chain(path):
    root = ET.parse(path).getroot()
    for joint in list(root.findall('joint')):
        if joint.get('type') == 'floating':
            root.remove(joint)
    for link in list(root.findall('link')):
        if link.get('name') == 'world':
            root.remove(link)
    return pk.build_chain_from_urdf(ET.tostring(root, encoding='unicode')).to(dtype=torch.float64, device='cpu')


def main():
    torch.set_num_threads(1)
    root = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    official = root/'omni_g1_urdf_20261030/models/g1'
    dc = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
    reference = _chain(dc)
    results = []
    for archive_name, urdf_name in [('robot-object.zip', 'g1_29dof.urdf'),
                                     ('robot-terrain.zip', 'g1_29dof_spherehand.urdf'),
                                     ('robot-object-terrain.zip', 'g1_29dof_spherehand.urdf')]:
        candidate = publisher_chain(official/urdf_name)
        rnames = reference.get_joint_parameter_names()
        cnames = candidate.get_joint_parameter_names()
        if set(rnames) != set(SOURCE_NAMES) or set(cnames) != set(SOURCE_NAMES):
            raise ValueError('joint name mismatch')
        indices = {n: i for i, n in enumerate(SOURCE_NAMES)}
        with zipfile.ZipFile(root/'data/OmniRetarget_Dataset'/archive_name) as archive:
            members = sorted(n for n in archive.namelist() if n.endswith('.npz'))[:4]
            sample = []
            for member in members:
                with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as x:
                    q = x['qpos'][:, 7:36]
                sample.append(q[[0, len(q)//2, len(q)-1]])
        q = np.concatenate(sample, axis=0)
        with torch.no_grad():
            expected = _positions(reference, torch.tensor(q[:, [indices[n] for n in rnames]], dtype=torch.float64))
            actual = _positions(candidate, torch.tensor(q[:, [indices[n] for n in cnames]], dtype=torch.float64))
            errors = torch.linalg.vector_norm(expected - actual, dim=-1)
        results.append({'archive': archive_name, 'urdf': urdf_name, 'members': members,
                        'sampled_frames': len(q), 'mean_keypoint_difference_m': float(errors.mean()),
                        'max_keypoint_difference_m': float(errors.max()), 'keypoints': list(FIT_LINKS)})
    print(json.dumps({'results': results, 'note': 'Root-local kinematic comparison only; sample-limited.'}))


if __name__ == '__main__':
    main()
