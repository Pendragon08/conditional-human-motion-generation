"""Explore bounded root-z corrections on four candidates without acceptance."""
import hashlib
import inspect
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from feature263 import canonicalize_z_up, encode
from g1_fk import URDFFK
from project_vertical_support_bound import project_vertical_support

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT=ROOT/'four_semantic_subset_20261101'
DATA=ROOT/'six_family_disjoint_preview_v2_20261101'
BOUND=ROOT/'vertical_support_bound_20261101'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    bound=json.loads((BOUND/'report.json').read_text());audit=json.loads((BOUND/'audit.json').read_text())
    assert sha(BOUND/'report.json')==audit['report_sha256'] and sha(BOUND/'trajectories.npz')==audit['trajectories_sha256']
    assert sha(INPUT/'manifest.json')==bound['manifest_sha256']
    records=json.loads((INPUT/'manifest.json').read_text())['records']
    states={r['sample_id']:r for r in json.loads((INPUT/'state_manifest.json').read_text())['records']}
    cases=[];fk=URDFFK(URDF)
    with np.load(BOUND/'trajectories.npz',allow_pickle=False) as old, (HERE/'failures.jsonl').open('x') as failures:
        for rec in records:
            sid=rec['sample_id'];state=states[sid];state_path=DATA/state['state_path'];feature_path=DATA/rec['feature_path']
            try:
                assert sha(state_path)==state['state_sha256'] and sha(feature_path)==rec['feature_sha256']
                with np.load(state_path,allow_pickle=False) as f:
                    root=f['root_wxyz_xyz'].astype(np.float64);q=f['joint_pos'].copy();t=f['sample_times'].copy();names=f['joint_names'].copy()
                with np.load(feature_path,allow_pickle=False) as f:original=f['features'].copy()
                assert np.array_equal(t,old[sid+'_times'])
                result=project_vertical_support(t,old[sid+'_com'],bound['gravity_m_s2'],max_shift_m=.02,smoothness_weight=10.)
                shift=result['height_shift_m'];adjusted=root.copy();adjusted[:,6]+=shift
                pts=fk.forward(adjusted[:,4:],adjusted[:,[1,2,3,0]],q,names.tolist())
                new=encode(pts,foot_velocity_squared_threshold=state['foot_velocity_squared_threshold'])
                if new.shape!=original.shape or not np.isfinite(new).all():raise ValueError('invalid re-encoding')
                with torch.no_grad():decoded=recover_from_ric(torch.from_numpy(new),22).numpy()
                decode_error=float(np.linalg.norm(decoded-canonicalize_z_up(pts)[:-1],axis=-1).max())
                if decode_error>1e-4:raise ValueError('decode self-check failed')
                out=HERE/(sid+'.npz')
                with out.open('xb') as f:
                    np.savez_compressed(f,features=new,root_wxyz_xyz=adjusted,joint_pos=q,sample_times=t,joint_names=names,
                        height_shift_m=shift,com_xyz=old[sid+'_com']+np.column_stack([np.zeros(len(t)),np.zeros(len(t)),shift]),
                        support_before=result['before_n_per_kg'],support_after=result['after_n_per_kg'])
                cases.append({'sample_id':sid,'status':'projected','output_sha256':sha(out),
                    'source_state_sha256':sha(state_path),'source_feature_sha256':sha(feature_path),
                    'max_abs_shift_m':float(np.abs(shift).max()),'solver_iterations':result['solver_iterations'],
                    'support_before_min_n_per_kg':float(result['before_n_per_kg'].min()),
                    'support_after_min_n_per_kg':float(result['after_n_per_kg'].min()),
                    'max_feature_component_difference':float(np.abs(new.astype(float)-original.astype(float)).max()),
                    'changed_foot_proxy_values':int(np.count_nonzero(new[:,-4:]!=original[:,-4:])),
                    'decode_max_error_m':decode_error})
            except Exception as exc:
                failures.write(json.dumps({'sample_id':sid,'error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})+'\n')
                failures.flush();cases.append({'sample_id':sid,'status':'failed','error_type':type(exc).__name__,'message':str(exc)})
    report={'input_manifest_sha256':sha(INPUT/'manifest.json'),'bound_report_sha256':sha(BOUND/'report.json'),
        'bound_audit_sha256':sha(BOUND/'audit.json'),'urdf_sha256':sha(URDF),'gravity_m_s2':bound['gravity_m_s2'],
        'max_root_z_shift_m':.02,'smoothness_weight':10.,'support_numerical_tolerance_n_per_kg':1e-6,
        'script_sha256':sha(Path(__file__)),'projection_module_sha256':sha(HERE/'project_vertical_support_bound.py'),
        'fk_module_sha256':sha(Path(inspect.getfile(URDFFK))),'encoder_module_sha256':sha(Path(inspect.getfile(encode))),
        'failures_sha256':sha(HERE/'failures.jsonl'),'cases':cases,'training_ready':False,
        'limitations':['Experimental root-only correction, not author v2 or paper filter.',
                       'The 20 mm limit and smoothness weight are project exploration choices.',
                       'A nonnegative vertical-force bound is necessary only under stated assumptions, never sufficient for dynamics.',
                       'No measured ground, actual forces, contact, friction, torque limits or caption revalidation.',
                       'Filtering, unknown real FPS and different model masses can change results; no training approval.']}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'cases':len(cases),'projected':sum(x['status']=='projected' for x in cases),'training_ready':False}))


if __name__=='__main__':main()
