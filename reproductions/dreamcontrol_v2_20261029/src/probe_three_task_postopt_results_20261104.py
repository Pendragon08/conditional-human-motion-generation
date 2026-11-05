import json,pathlib
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104'
for line in (B/'pilot_postopt_v2/records.jsonl').read_text().splitlines():
 j=json.loads(line);print(json.dumps({k:j[k] for k in ['candidate_id','task','status','rejection_reason','rejection_metrics','max_joint_projection_rad','max_hand_error_m','hand_error_at_grasp_m','max_wrist_angle_rad','checks','passed_requested_kinematic_checks','output'] if k in j}))
