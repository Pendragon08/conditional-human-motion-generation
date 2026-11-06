"""Read-only comparison of an old admitted Pick bank against new geometric tests."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'repair_prior29_v2_20261105'
O=R/'stage3_pick29_20261103/reference_scale500_20261104'
OUT=S/'pick_legacy_reference_comparison_v1.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    if OUT.is_file():
        print(OUT.read_text())
        return
    bank=O/'final_library_v1/reference_bank.json'
    assert sha(bank)=='01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572'
    rows=read(bank)['references']
    assert len(rows)==500
    new=S/'pick_wrist_anchor_diagnostic_v3/candidate_10000007/report.json'
    assert sha(new)=='216cf8875c734b41251354885287922a0d134d9ad7ca0a9cd92e9a330ac9be33'
    recent=read(new)
    raw=Path(recent['source_sample'])
    assert sha(raw)==recent['source_sample_sha256']
    with np.load(raw,allow_pickle=False) as z:
        m=json.loads(z['metadata_utf8'].tobytes())
    object_new=np.array(m['task_object_position'])
    nearest=sorted(rows,key=lambda row:np.linalg.norm(np.asarray(row['task_object_position'])-object_new))[:12]
    comparison=[]
    for row in nearest:
        old=Path(row['path'])
        assert sha(old)==row['sha256']
        proof=Path(row['postopt_report'])
        assert sha(proof)==row['postopt_report_sha256']
        kinematics=read(Path(row['kinematic_report']))
        assert kinematics['output_sha256']==row['sha256']
        sample=Path(kinematics['sample'])
        assert sha(sample)==kinematics['sample_sha256']
        with np.load(sample,allow_pickle=False) as z:
            points=z['canonical_points'][0]
            goal=z['target_points']
            meta=json.loads(z['metadata_utf8'].tobytes())
        rotation=np.asarray(meta['canonical_to_world_rotation'])
        translation=np.asarray(meta['canonical_to_world_translation'])
        raw=points@rotation.T+translation
        goal=goal@rotation.T+translation
        with np.load(old,allow_pickle=False) as z:
            fitted=z['keypoints']
            quat=Rotation.from_quat(z['right_wrist_quat'][:,[1,2,3,0]])
        errors=np.linalg.norm(fitted-raw,axis=-1)
        max_pos=float(np.linalg.norm(fitted[:,21]-goal[:,21],axis=-1).max())
        assert abs(max_pos-kinematics['max_hand_error_m']) < 1e-5
        rmse=float(np.sqrt(np.mean(errors**2)))
        peak=float(errors.max())
        angle=float(kinematics['max_wrist_angle_rad'])
        wrist_step=float((quat[:-1].inv()*quat[1:]).magnitude().max())
        comparison.append({'candidate_id':kinematics['candidate_id'],'source_sha256':kinematics['sample_sha256'],
                           'object_distance_m':float(np.linalg.norm(np.asarray(row['task_object_position'])-object_new)),
                           'wrist_position_m':max_pos,'wrist_angle_rad':angle,'wrist_step_rad':wrist_step,
                           'raw_reconstruction_rmse_m':rmse,'raw_reconstruction_peak_m':peak,
                           'passes_new_absolute_subset':max_pos<.03 and angle<.20 and wrist_step<=.25 and rmse<=.05 and peak<=.15})
    result={'status':'old_admitted_reference_comparison_complete_not_new_admission','old_bank_sha256':sha(bank),
            'candidate_id_new':10000007,'new_report_sha256':sha(new),
            'old_rows':comparison,'new_subset_pass_count':sum(row['passes_new_absolute_subset'] for row in comparison),
            'scope':'12 old admitted rows nearest by object center, each compared to its own raw and task targets; no re-optimization, contact or new admission'}
    content=json.dumps(result,indent=2,allow_nan=False)
    with OUT.open('x') as stream:stream.write(content)
    print(content)


if __name__=='__main__':
    main()
