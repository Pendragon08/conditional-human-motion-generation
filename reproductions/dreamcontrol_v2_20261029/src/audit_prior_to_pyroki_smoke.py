"""Independent structural/limit audit of the exploratory CPU chain output.

Does not independently rerun either the diffusion model or the solver.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN=ROOT/'prior_to_pyroki_real_smoke_v3_20261101';HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    report=json.loads((RUN/'report.json').read_text());prior=RUN/'synthetic_prior.npz';robot=RUN/'synthetic_robot.npz'
    if (sha(prior)!=report['prior_sha256'] or sha(robot)!=report['robot_output_sha256'] or
        sha(RUN/'prior_to_pyroki_real_smoke.py')!=report['script_sha256']):raise ValueError('chain report hash mismatch')
    with np.load(prior,allow_pickle=False) as f:
        points=np.asarray(f['canonical_points']);features=np.asarray(f['normalized_features']);meta=json.loads(f['metadata_utf8'].tobytes())
    with np.load(robot,allow_pickle=False) as f:
        q=np.asarray(f['joint_pos'],dtype=float);root=np.asarray(f['root_wxyz_xyz'],dtype=float)
        names=[str(x) for x in f['joint_names']];idx=np.asarray(f['input_frame_indices']);outmeta=json.loads(f['metadata_utf8'].tobytes())
    if (points.shape!=(1,8,22,3) or features.shape!=(1,263,1,8) or q.shape!=(7,27) or root.shape!=(7,7) or
        not all(np.isfinite(a).all() for a in (points,features,q,root)) or len(set(names))!=27 or
        not np.array_equal(idx,np.arange(1,8)) or idx.tolist()!=report['sample_indices'] or
        outmeta['input_sha256']!=sha(prior) or meta['human_pretrained_checkpoint_sha256']!=report['checkpoint_sha256'] or
        outmeta['training_or_physical_acceptance'] is not False or meta['training_or_physical_acceptance'] is not False):
        raise ValueError('chain structure, alignment or metadata mismatch')
    parsed={j.attrib['name']:(float(j.find('limit').attrib['lower']),float(j.find('limit').attrib['upper']))
            for j in ET.parse(URDF).getroot().findall('joint') if j.find('limit') is not None and 'lower' in j.find('limit').attrib}
    low=np.array([parsed[n][0] for n in names]);high=np.array([parsed[n][1] for n in names])
    excess=np.maximum(np.maximum(low[None,:]-q,q-high[None,:]),0)
    worst=np.unravel_index(np.argmax(excess),excess.shape)
    max_excess=float(excess[worst]);quat_error=float(np.max(np.abs(np.linalg.norm(root[:,:4],axis=1)-1)))
    if abs(max_excess-report['postopt_metadata']['max_joint_limit_violation_rad'])>1e-12 or quat_error>1e-4:
        raise ValueError('joint excess or root quaternion mismatch')
    result={'run_report_sha256':sha(RUN/'report.json'),'prior_sha256':sha(prior),'robot_output_sha256':sha(robot),
            'urdf_sha256':sha(URDF),'script_sha256':sha(__file__),'root_shape':list(root.shape),'joint_shape':list(q.shape),
            'input_frame_indices':idx.tolist(),'max_joint_excess_rad':max_excess,'worst_frame':int(worst[0]),
            'worst_joint':names[worst[1]],'joint_values_above_limits':int(np.count_nonzero(excess)),
            'max_root_quaternion_norm_error':quat_error,'training_ready':False,
            'limitations':['Same saved outputs, independently parsed URDF limits; no solver or model rerun.',
                'Human checkpoint, synthetic stats/hint and dummy ground; no real robot training, semantics or physics acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
