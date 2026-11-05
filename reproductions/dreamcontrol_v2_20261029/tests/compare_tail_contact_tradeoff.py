"""Tune contact-proxy weights on the original three adversarial clips only.

Contact is estimated from source-ankle speed; terrain and real contact are
unknown. Frame-to-frame displacement is penalized, not velocity, so weights
are not comparable to a loss expressed in m/s. Do not evaluate on held-out data
until a configuration has been selected and documented.
"""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_ik import FIT_LINKS
from waist_lock_root_ik import fit_root_and_waist


def main():
    torch.set_num_threads(1)
    root = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    previous = json.loads((root / 'tail_fit_20261030/report.json').read_text())
    if len(previous['samples']) != 3:
        raise ValueError('expected three adversarial tuning clips')
    assets = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
    u29, u27 = assets/'g1_29dof.urdf', assets/'g1_27dof.urdf'
    fk29, fk27 = URDFFK(u29), URDFFK(u27)
    configs = [(0., 2., 10.), (1., 2., 10.), (1., 2., 100.),
               (1., 2., 400.), (1., 2., 1000.), (1., 10., 400.)]
    samples = []
    for previous_sample in previous['samples']:
        member = previous_sample['member']
        with zipfile.ZipFile(root/'data/OmniRetarget_Dataset'/previous_sample['archive']) as archive:
            with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as data:
                q = resample(data['qpos'], float(data['fps']))
        motion, _ = convert_layout(q)
        truth = fk29.forward(motion[:,:3], motion[:,3:7], motion[:,7:], SOURCE_NAMES, links=FIT_LINKS)
        velocity = np.diff(truth[:,4:6], axis=0)*20
        contact = np.linalg.norm(velocity, axis=-1) < .03
        outputs = []
        for tail, foot, contact_weight in configs:
            fit = fit_root_and_waist(motion, u29, u27, steps=120, tail_weight=tail,
                                     foot_position_weight=foot, contact_velocity_weight=contact_weight)
            pred = fk27.forward(fit['root_pos'], fit['root_quat_xyzw'], fit['joint_pos'], fit['joint_names'], links=FIT_LINKS)
            err = np.linalg.norm(pred-truth, axis=-1)
            velerr = np.linalg.norm(np.diff(pred[:,4:6], axis=0)*20-velocity, axis=-1)
            outputs.append({'tail':tail, 'foot':foot, 'contact':contact_weight,
                            'mean_error_m':float(err.mean()), 'max_error_m':float(err.max()),
                            'mean_foot_error_m':float(err[:,4:6].mean()),
                            'contact_velocity_error_mps':float(velerr[contact].mean()) if contact.any() else None})
        samples.append({'archive':previous_sample['archive'], 'member':member, 'frames':len(motion), 'configs':outputs})
    print(json.dumps({'samples':samples, 'steps':120, 'fps':20,
                      'warning':'Tuning clips only, no real contact, collision or dynamics validation.'}, allow_nan=False))


if __name__ == '__main__':
    main()
