"""Independently check batch provenance and FK; do not infer training readiness."""
import hashlib,io,json,traceback,zipfile
from pathlib import Path
import numpy as np
from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from audit_robot_states import audit_robot_states

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'data/OmniRetarget_Dataset/robot-object.zip'
ANNOT=ROOT/'omomo_text_pairing_20261101/omomo_text_anno.zip'
ROBOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
LINKS=('left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
       'left_ankle_roll_link','right_ankle_roll_link','left_shoulder_roll_link','right_shoulder_roll_link',
       'left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    report=json.loads((HERE/'report.json').read_text()); records=[json.loads(x) for x in (HERE/'records.jsonl').read_text().splitlines()]
    assert sha(HERE/'records.jsonl')==report['records_sha256'] and sha(ARCHIVE)==report['source_archive_sha256']
    assert sha(ANNOT)=='2d11fb37fef294b7f6816c74740694d12f28d62e0f638e348c678dc51ee1ad7c'
    assert len(records)==report.get('attempted',report['selected']) and len({r['sample_id'] for r in records})==len(records)
    assert sha(ROBOT/'g1_29dof.urdf')==report['urdf29_sha256'] and sha(ROBOT/'g1_27dof.urdf')==report['urdf27_sha256']
    fk29=URDFFK(ROBOT/'g1_29dof.urdf'); fk27=URDFFK(ROBOT/'g1_27dof.urdf'); output=[]
    with zipfile.ZipFile(ARCHIVE) as z, zipfile.ZipFile(ANNOT) as a,(HERE/'audit_rows_v2.jsonl').open('x',buffering=1) as stream:
        for r in records:
            try:
                assert r['technical_processing']=='success'
                raw=z.read(r['source_member']); assert hashlib.sha256(raw).hexdigest()==r['source_sha256']
                text=a.read(r['annotation_member']); assert hashlib.sha256(text).hexdigest()==r['annotation_sha256']
                annotation=json.loads(text.decode('utf8'))
                assert list(annotation)==[r['sample_id']] and annotation[r['sample_id']]==r['caption']
                with np.load(io.BytesIO(raw),allow_pickle=False) as d:
                    motion,obj=convert_layout(d['qpos']); assert float(d['fps'])==30.
                full=HERE/r['state30']; assert sha(full)==r['state30_sha256']
                with np.load(full,allow_pickle=False) as d:
                    root=d['root_wxyz_xyz']; q=d['joint_pos']; names=list(d['joint_names']); times=d['sample_times']
                    assert np.array_equal(obj,d['object_wxyz_xyz']) and np.array_equal(times,np.arange(len(times))/30.)
                target=fk29.forward(motion[:,:3],motion[:,3:7],motion[:,7:],list(SOURCE_NAMES),links=LINKS)
                pred=fk27.forward(root[:,4:],root[:,[1,2,3,0]],q,names,links=LINKS)
                err=np.linalg.norm(target-pred,axis=-1)
                assert abs(float(err.mean())-r['fit_mean_error_m'])<1e-8
                assert abs(float(err.max())-r['fit_max_error_m'])<1e-8
                state=audit_robot_states([{'sample_id':r['sample_id'],'feature_path':r['feature'],
                    'feature_sha256':r['feature_sha256'],'state_path':r['state20'],
                    'state_sha256':r['state20_sha256'],'foot_velocity_squared_threshold':.002}],
                    feature_root=HERE,state_root=HERE,urdf_path=ROBOT/'g1_27dof.urdf',
                    tolerance_rad=1e-6,fk_decode_tolerance_m=1e-4,reencoding_tolerance=1e-5)
                assert state['state_technical_checks_passed']
                out={'sample_id':r['sample_id'],'status':'technical_checks_passed','source_caption_bytes_match':True,
                    'independent_fk_mean_error_m':float(err.mean()),'independent_fk_max_error_m':float(err.max()),
                    'state_check':state['cases'][0],'training_ready':False}
            except Exception as e:
                out={'sample_id':r['sample_id'],'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'training_ready':False}
            output.append(out);stream.write(json.dumps(out,allow_nan=False)+'\n');stream.flush()
    result={'report_sha256':sha(HERE/'report.json'),'audit_script_sha256':sha(Path(__file__)),
        'records':len(records),'selected':report['selected'],'stopped_reason':report.get('stopped_reason'),
        'technical_passed':sum(x['status']=='technical_checks_passed' for x in output),
        'failed':[x['sample_id'] for x in output if x['status']!='technical_checks_passed'],
        'annotation_zip_sha256':sha(ANNOT),'audit_rows_sha256':sha(HERE/'audit_rows_v2.jsonl'),
        'limitations':['Full source conversion helper shared; 20 Hz FIR not independently recomputed.',
        'Caption identity and bytes do not establish text semantics or original-recording equivalence.',
        'No source license, contact, collision or dynamic acceptance; not a paper filter.'],'training_ready':False}
    with (HERE/'audit_v2.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
