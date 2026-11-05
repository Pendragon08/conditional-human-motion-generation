"""Build exactly500 hash-bound Pick references after native/source audits.

CPU-only integrity/kinematic recomputation. The actual500-bank loader and8912
scene witness remain a separate prerequisite; this script never starts PPO.
"""
import argparse,hashlib,json,shutil,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
sys.path[:0]=[str(S/'src'),str(R/'training_disk_guard_20261101')]
from stage3_pick29_core import BODY_JOINTS,SCENE_SHA,sha
from g1_fk import URDFFK
ASSET=S/'assets/g1_29dof_with_hands_urdf_aligned.usd'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--candidate-audit',type=Path,action='append',required=True)
    a=p.parse_args();assert not a.output_dir.exists()
    assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2+10*1024**2
    assert sha(URDF)=='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95'
    started=time.monotonic();pilot_path=O/'native_pilot_v1/reference_bank.json';pilot=json.loads(pilot_path.read_text())
    items=list(pilot['references']);batch_proofs=[];known={0,1,2,3}
    for directory in sorted((O/'native_batches').glob('batch_[0-9]*')):
        summary=json.loads((directory/'summary.json').read_text());assert summary['status']=='native_batch_completed'
        assert not known.intersection(summary['candidate_ids']);known.update(summary['candidate_ids'])
        native=directory/'native/report.json';assert sha(native)==summary['native_report_sha256']
        bank=directory/'native/reference_bank.json'
        if bank.exists():
            assert sha(bank)==summary['native_manifest_sha256'];items.extend(json.loads(bank.read_text())['references'])
        batch_proofs.append({'summary':str(directory/'summary.json'),'sha256':sha(directory/'summary.json')})
    assert len(items)>=500,'native-admitted reference count has not reached500'
    noise_audits={};audit_proofs=[]
    for path in a.candidate_audit:
        audit=json.loads(path.read_text());assert audit['passed'] and audit['candidates']==len(audit['details'])
        for detail in audit['details']:
            assert detail['noise_recomputed'] and detail['target_object_binding'] and detail['20Hz_120states']
            key=detail['candidate_id'];assert key not in noise_audits,'duplicate IDs across candidate source audits'
            noise_audits[key]={'detail':detail,'audit':str(path),'audit_sha256':sha(path)}
        audit_proofs.append({'path':str(path),'sha256':sha(path),'candidates':audit['candidates']})
    # Selection depends only on immutable candidate ID, not task success or error.
    def identifier(item):return int(json.loads(Path(item['kinematic_report']).read_text())['candidate_id'])
    items=sorted(items,key=identifier);assert len({identifier(i) for i in items})==len(items)
    selected=items[:500];fk=URDFFK(URDF)
    limits={j.get('name'):j.find('limit') for j in ET.parse(URDF).getroot().findall('joint') if j.get('type')=='revolute'}
    low=np.array([float(limits[n].get('lower')) for n in BODY_JOINTS]);high=np.array([float(limits[n].get('upper')) for n in BODY_JOINTS])
    ids=set();noises=set();features=set();states=set();details=[];asset_sha=sha(ASSET)
    for item in selected:
        state_path=Path(item['path']).resolve();assert state_path.is_relative_to(O.resolve())
        assert sha(state_path)==item['sha256']
        native_path=Path(item['postopt_report']);assert sha(native_path)==item['postopt_report_sha256']
        native=json.loads(native_path.read_text());kinpath=Path(item['kinematic_report']);kin=json.loads(kinpath.read_text())
        assert native['eligible_for_calibrated_scene_reference'] and all(native['native_checks'].values())
        assert native['asset_sha256']==asset_sha and native['scene_sha256']==SCENE_SHA and native['native_replay_frames']==120
        assert sha(kinpath)==native['kinematic_report_sha256'] and all(kin['checks'].values()) and kin['output_sha256']==item['sha256']
        sample=Path(kin['sample']);assert sha(sample)==kin['sample_sha256']
        with np.load(state_path,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        with np.load(sample,allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
        meta=json.loads(s['metadata_utf8'].tobytes());original=json.loads(source['metadata_utf8'].tobytes());assert meta==original
        cid=meta['candidate_id'];proof=noise_audits[cid];assert proof['detail']['file_sha256']==kin['sample_sha256']
        assert cid==kin['candidate_id'] and meta['initial_noise_sha256']==kin['source_initial_noise_sha256']
        assert hashlib.sha256(source['features'].tobytes()).hexdigest()==meta['feature_content_sha256']==kin['source_feature_content_sha256']
        assert s['joint_names'].tolist()==BODY_JOINTS and float(s['fps'])==20
        assert np.array_equal(s['sample_times'],np.arange(120)/20.)
        for key,shape in {'q':(120,29),'root_pos':(120,3),'root_quat':(120,4),'keypoints':(120,22,3),
                          'right_wrist_quat':(120,4),'hand_closed':(120,2)}.items():
            assert s[key].shape==shape and np.isfinite(s[key]).all(),key
        assert np.max(low-s['q'])<=1e-7 and np.max(s['q']-high)<=1e-7
        for key in ['root_quat','right_wrist_quat']:assert np.max(np.abs(np.linalg.norm(s[key],axis=-1)-1))<1e-4
        assert np.all(s['hand_closed'][:,0]==0) and np.all(s['hand_closed'][:40,1]==0) and np.all(s['hand_closed'][40:,1]==1)
        obj=np.asarray(item['task_object_position']);assert np.all(np.abs(obj[:2]-[.35,-.2])<=.0600001) and obj[2]==.9
        assert np.allclose(obj,s['task_object_position'],atol=1e-12) and np.allclose(obj,kin['task_object_position'],atol=1e-12)
        assert np.allclose(obj,native['task_object_position'],atol=1e-12)
        assert np.array_equal(s['task_object_quaternion_wxyz'],np.array([1.,0.,0.,0.]))
        target=source['target_points']@np.array(meta['canonical_to_world_rotation']).T+np.array(meta['canonical_to_world_translation'])
        assert np.allclose(target[40,21],obj+[0,0,.04],atol=2e-7) and np.allclose(target[80,21],obj+[0,0,.20],atol=2e-7)
        points=fk.forward(s['root_pos'],s['root_quat'][:,[1,2,3,0]],s['q'],BODY_JOINTS)
        fk_error=float(np.abs(points-s['keypoints']).max());assert fk_error<1e-5
        error=np.linalg.norm(points[:,21]-target[:,21],axis=-1);assert float(error.max())<.15
        assert abs(float(error[40])-kin['hand_error_at_grasp_m'])<1e-6
        content=hashlib.sha256(b''.join(np.ascontiguousarray(s[k]).tobytes() for k in ['q','root_pos','root_quat','hand_closed'])).hexdigest()
        assert cid not in ids and meta['initial_noise_sha256'] not in noises and meta['feature_content_sha256'] not in features and content not in states
        ids.add(cid);noises.add(meta['initial_noise_sha256']);features.add(meta['feature_content_sha256']);states.add(content)
        details.append({'candidate_id':cid,'state_sha256':item['sha256'],'native_report_sha256':item['postopt_report_sha256'],
            'sample_sha256':kin['sample_sha256'],'noise_recomputation_audit_sha256':proof['audit_sha256'],
            'robot_state_content_sha256':content,'fk_recompute_max_error_m':fk_error,'wrist_error_at2s_m':float(error[40]),
            'maximum_wrist_target_error_m':float(error.max()),'task_object_position':obj.tolist()})
    calibration=Path(pilot['calibration_report']);assert sha(calibration)==pilot['calibration_report_sha256']
    record={k:v for k,v in pilot.items() if k!='references'}
    record.update(references=selected,task='pick',object_pose_policy='per_reference_fixed',minimum_formal_reference_count=500,
        selection_policy='first500 native-admitted unique candidate IDs; all surplus files preserved',candidate_source_audits=audit_proofs,
        native_batch_summaries=batch_proofs,accepted_native_references_available=len(items))
    a.output_dir.mkdir(exist_ok=False);manifest=a.output_dir/'reference_bank.json'
    manifest.open('x').write(json.dumps(record,indent=2))
    audit={'status':'cpu_integrity_passed_native_loader_pending','reference_count':500,'manifest':str(manifest),
        'manifest_sha256':sha(manifest),'available_native_references':len(items),'candidate_source_audits':audit_proofs,
        'unique_ID_noise_feature_robot_state_contents':True,'details':details,'source_code_sha256':sha(__file__),
        'asset_sha256':asset_sha,'scene_sha256':SCENE_SHA,'urdf_sha256':sha(URDF),'elapsed_seconds':time.monotonic()-started,
        'scope':'full generated reference integrity/kinematic/source/native proof recomputation; not PD tracking or task success',
        'required_next_step':'actual defaultReferenceBank500/8912-env128step capacity witness before final admission receipt or PPO'}
    (a.output_dir/'cpu_audit.json').open('x').write(json.dumps(audit,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in audit.items() if k!='details'}),flush=True)

if __name__=='__main__':main()
