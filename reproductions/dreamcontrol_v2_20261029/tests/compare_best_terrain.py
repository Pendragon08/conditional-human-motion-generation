"""Cross-task diagnostic on four terrain clips; sources were previously audited.

Filename-derived climb groups have appeared in broader audits. This is not an
independent held-out evaluation. It only checks kinematic fit and an ankle-speed
proxy; ground height, collisions and dynamics are not modeled here.
"""
import hashlib
import io
import json
from pathlib import Path
import re
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
    archive_path = root/'data/OmniRetarget_Dataset/robot-terrain.zip'
    prior = set()
    for folder in ('fit_quality_audit_50_20261030','fit_quality_audit_211_20261030'):
        manifest = json.loads((root/folder/'manifest.json').read_text())
        prior.update(x['member'] for x in manifest['samples'] if x['archive'] == archive_path.name)
    assets = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
    u29, u27 = assets/'g1_29dof.urdf', assets/'g1_27dof.urdf'
    fk29, fk27 = URDFFK(u29), URDFFK(u27)
    samples = []
    with zipfile.ZipFile(archive_path) as archive:
        pool = [n for n in archive.namelist() if re.fullmatch(r'robot-terrain/climb_\d+_z_scale_[0-9.]+\.npz', n) and n not in prior]
        pool.sort(key=lambda n: hashlib.sha256(n.encode()).hexdigest())
        groups = set()
        selected = []
        for member in pool:
            group = re.match(r'robot-terrain/(climb_\d+)_', member).group(1)
            if group not in groups:
                selected.append(member)
                groups.add(group)
            if len(selected) == 4:
                break
        if len(selected) < 4:
            raise ValueError('fewer than four terrain paths absent from earlier audit')
        for member in selected:
            with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as data:
                q = resample(data['qpos'], float(data['fps']))[:200]
            motion, _ = convert_layout(q)
            truth = fk29.forward(motion[:,:3], motion[:,3:7], motion[:,7:], SOURCE_NAMES, links=FIT_LINKS)
            source_velocity = np.diff(truth[:,4:6], axis=0)*20
            mask = np.linalg.norm(source_velocity, axis=-1) < .03
            configs = []
            for method in ('previous', 'best_experimental'):
                if method == 'previous':
                    fit = fit_root_and_waist(motion, u29, u27, steps=120, learning_rate=.015,
                                             tail_weight=0., foot_position_weight=2., contact_velocity_weight=10.)
                else:
                    fit = fit_root_and_waist(motion, u29, u27, steps=720, learning_rate=.005,
                                             tail_weight=1., foot_position_weight=2., contact_velocity_weight=100.,
                                             return_best=True)
                pred = fk27.forward(fit['root_pos'], fit['root_quat_xyzw'], fit['joint_pos'], fit['joint_names'], links=FIT_LINKS)
                err = np.linalg.norm(pred-truth, axis=-1)
                verr = np.linalg.norm(np.diff(pred[:,4:6],axis=0)*20-source_velocity,axis=-1)
                configs.append({'method':method,'best_step':fit['best_step'],
                                'mean_error_m':float(err.mean()),'max_error_m':float(err.max()),
                                'mean_foot_error_m':float(err[:,4:6].mean()),
                                'contact_velocity_error_mps':float(verr[mask].mean()) if mask.any() else None})
            samples.append({'member':member,'frames':len(motion),'configs':configs})
    print(json.dumps({'samples':samples,'fps':20,'max_frames':200,
                      'warning':'Paths are new, filename-derived motion groups were seen in prior audit. No terrain contact or dynamics validation; configurations use different iteration budgets.'},allow_nan=False))


if __name__ == '__main__':
    main()
