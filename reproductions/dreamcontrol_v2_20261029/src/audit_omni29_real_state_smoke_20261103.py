"""Run project state/feature technical audit against one real 29DoF artifact."""
import hashlib,json,sys
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'omni29_direct_text_motion_pilot_20261103'
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
T=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
sys.path[:0]=[str(R/'gmr29_validation_work_20261103'),str(R),str(R/'ssm_new6_20hz_resume_20261101'),str(T)]
from audit_robot_states import audit_robot_states
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=D/'state_features.npz';h=sha(p);rep=json.loads((D/'report.json').read_text())
 if h!=rep['output_sha256']:raise ValueError('source hash')
 rows=[{'sample_id':'sub3_largebox_011_original','feature_path':p.name,'feature_sha256':h,'state_path':p.name,'state_sha256':h,'foot_velocity_squared_threshold':.002,'encoding_variant':'sign_continuous_v1'}]
 result=audit_robot_states(rows,feature_root=D,state_root=D,urdf_path=U,tolerance_rad=1e-6,fk_decode_tolerance_m=1e-4,reencoding_tolerance=1e-5)
 out=R/'gmr29_validation_work_20261103/omni29_real_state_smoke.json'
 with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps({'technical_pass':result['state_technical_checks_passed'],'training_ready':result['training_ready'],'limit_failed_samples':result['limit_failed_samples'],'fk_decode_max_m':result['cases'][0]['fk_decode_max_error_m'],'reencode_max':result['cases'][0]['feature_reencoding_max_abs_error']}))
if __name__=='__main__':main()
