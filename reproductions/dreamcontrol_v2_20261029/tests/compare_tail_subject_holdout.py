"""Exploratory subject-separated tail-loss comparison; not physics validation.

The tail parameter was tuned on subjects 8, 10 and 16. Choose three original
clips each from 3, 7 and 12 by deterministic name hash, excluding exact prior
audit paths. The foot loss was previously selected on broader data, so this is
not a fully untouched evaluation and independence of source recordings is not
proven. At most the first 200 20 Hz frames of a clip are evaluated.
"""
import hashlib
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
    archive_path = root / 'data/OmniRetarget_Dataset/robot-object.zip'
    prior = set()
    for folder in ('fit_quality_audit_50_20261030', 'fit_quality_audit_211_20261030'):
        manifest = json.loads((root / folder / 'manifest.json').read_text())
        prior.update(x['member'] for x in manifest['samples'] if x['archive'] == archive_path.name)
    assets = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
    u29, u27 = assets / 'g1_29dof.urdf', assets / 'g1_27dof.urdf'
    fk29, fk27 = URDFFK(u29), URDFFK(u27)
    samples = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for subject in (3, 7, 12):
            prefix = f'robot-object/sub{subject}_'
            pool = [n for n in names if n.startswith(prefix) and n.endswith('_original.npz') and n not in prior]
            pool.sort(key=lambda n: hashlib.sha256(n.encode()).hexdigest())
            if len(pool) < 3:
                raise ValueError(f'insufficient clips for subject {subject}: {len(pool)}')
            for member in pool[:3]:
                with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as data:
                    q = resample(data['qpos'], float(data['fps']))[:200]
                motion, _ = convert_layout(q)
                truth = fk29.forward(motion[:, :3], motion[:, 3:7], motion[:, 7:], SOURCE_NAMES, links=FIT_LINKS)
                velocity = np.diff(truth[:, 4:6], axis=0) * 20
                contact = np.linalg.norm(velocity, axis=-1) < .03
                results = {}
                for weight in (0., 1.):
                    fit = fit_root_and_waist(motion, u29, u27, steps=120, foot_position_weight=2.,
                                             contact_velocity_weight=10., tail_weight=weight)
                    pred = fk27.forward(fit['root_pos'], fit['root_quat_xyzw'], fit['joint_pos'],
                                        fit['joint_names'], links=FIT_LINKS)
                    err = np.linalg.norm(pred - truth, axis=-1)
                    velerr = np.linalg.norm(np.diff(pred[:, 4:6], axis=0) * 20 - velocity, axis=-1)
                    results[str(weight)] = {'mean_error_m': float(err.mean()), 'max_error_m': float(err.max()),
                                            'mean_foot_error_m': float(err[:, 4:6].mean()),
                                            'contact_velocity_error_mps': float(velerr[contact].mean()) if contact.any() else None}
                samples.append({'subject': subject, 'member': member, 'frames': len(motion), 'weights': results})
    print(json.dumps({'samples': samples, 'tail_fraction': .05, 'steps': 120,
                      'selection': 'three sha256-name-ranked unseen original paths per subject 3,7,12',
                      'warning': 'Only tail tuning is subject-separated; previous foot tuning and true source independence are unverified. No contact, collision or dynamics validation.'}, allow_nan=False))


if __name__ == '__main__':
    main()
