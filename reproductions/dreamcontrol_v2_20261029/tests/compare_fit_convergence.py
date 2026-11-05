"""Explore optimizer convergence on the same three adversarial tuning clips.

Neither these clips nor the ankle-speed proxy establish real contact, physical
feasibility, or independent generalization. Configurations are exploratory.
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
    previous = json.loads((root/'tail_fit_20261030/report.json').read_text())
    if len(previous['samples']) != 3:
        raise ValueError('expected three adversarial tuning clips')
    assets = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
    u29, u27 = assets/'g1_29dof.urdf', assets/'g1_27dof.urdf'
    fk29, fk27 = URDFFK(u29), URDFFK(u27)
    configs = [(120,.015,10.), (120,.015,100.), (360,.015,100.),
               (360,.005,100.), (720,.005,100.)]
    samples = []
    for item in previous['samples']:
        with zipfile.ZipFile(root/'data/OmniRetarget_Dataset'/item['archive']) as archive:
            with np.load(io.BytesIO(archive.read(item['member'])), allow_pickle=False) as data:
                q = resample(data['qpos'], float(data['fps']))
        motion, _ = convert_layout(q)
        truth = fk29.forward(motion[:,:3], motion[:,3:7], motion[:,7:], SOURCE_NAMES, links=FIT_LINKS)
        source_velocity = np.diff(truth[:,4:6], axis=0)*20
        mask = np.linalg.norm(source_velocity, axis=-1) < .03
        results = []
        for steps, lr, weight in configs:
            fit = fit_root_and_waist(motion, u29, u27, steps=steps, learning_rate=lr,
                                     tail_weight=1., foot_position_weight=2., contact_velocity_weight=weight)
            pred = fk27.forward(fit['root_pos'], fit['root_quat_xyzw'], fit['joint_pos'], fit['joint_names'], links=FIT_LINKS)
            err = np.linalg.norm(pred-truth, axis=-1)
            verr = np.linalg.norm(np.diff(pred[:,4:6], axis=0)*20-source_velocity, axis=-1)
            results.append({'steps':steps,'learning_rate':lr,'contact_weight':weight,
                            'mean_error_m':float(err.mean()),'max_error_m':float(err.max()),
                            'mean_foot_error_m':float(err[:,4:6].mean()),
                            'contact_velocity_error_mps':float(verr[mask].mean()) if mask.any() else None})
        samples.append({'archive':item['archive'],'member':item['member'],'frames':len(motion),'configs':results})
    print(json.dumps({'samples':samples,'tail_weight':1.,'foot_position_weight':2.,'fps':20,
                      'warning':'Adversarial tuning clips only; no real contact, collision or dynamics validation.'}, allow_nan=False))


if __name__ == '__main__':
    main()
