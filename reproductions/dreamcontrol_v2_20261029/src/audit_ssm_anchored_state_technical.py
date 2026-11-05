"""Re-run existing full state-to-feature checks for two unaccepted candidates."""
import hashlib
import json
from pathlib import Path
from audit_robot_states import audit_robot_states

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SRC=ROOT/'ssm_anchored_20hz_candidates_20261031'
HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    audit=json.loads((SRC/'independent_audit.json').read_text())
    records=[]
    for row in audit['cases']:
        sid=row['id']; name=sid+'.npz'; path=SRC/name
        assert sha(path)==row['output_sha256']
        records.append({'sample_id':sid,'feature_path':name,'state_path':name,
                        'feature_sha256':sha(path),'state_sha256':sha(path),
                        'foot_velocity_squared_threshold':.002})
    result=audit_robot_states(records,feature_root=SRC,state_root=SRC,urdf_path=URDF,
                              tolerance_rad=1e-6,fk_decode_tolerance_m=1e-4,reencoding_tolerance=1e-5)
    result.update({'input_audit_sha256':sha(SRC/'independent_audit.json'),'script_sha256':sha(Path(__file__)),
                   'thresholds_are_exploratory':True})
    with (HERE/'state_technical.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'technical_passed':result['state_technical_checks_passed'],'samples':len(records),'training_ready':False}))


if __name__=='__main__':main()
