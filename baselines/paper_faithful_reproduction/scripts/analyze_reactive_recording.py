"""Recompute explicit diagnostic tracking errors from synchronous session streams.

Paper thresholds are used, but the 30-body selection is an explicit choice:
authors have not released their benchmark evaluator. Not a paper benchmark.
"""
import argparse,ast,hashlib,json
from pathlib import Path
import numpy as np
import mujoco

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--inference-source',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 names=None
 for node in ast.parse(a.inference_source.read_text(encoding='utf-8')).body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='G1_29DOF_JOINT_NAMES' for t in node.targets):names=ast.literal_eval(node.value)
 assert len(names)==29
 model=mujoco.MjModel.from_xml_path(str(a.model));d=mujoco.MjData(model)
 joint_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
 addresses=model.jnt_qposadr[joint_ids]
 pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis');torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
 bodies=[pelvis]+[int(model.jnt_bodyid[i]) for i in joint_ids]
 assert len(set(bodies))==30
 raw=np.load(a.session,allow_pickle=False)
 assert np.array_equal(raw['real_robot_policy_step'],raw['active_plan_policy_step'])
 assert np.array_equal(raw['real_robot_sample_perf_s'],raw['active_plan_sample_perf_s'])
 positions={}
 for key in ['real_robot_motion36','active_plan_motion36']:
  rows=[]
  for pose in raw[key]:
   d.qpos[:]=model.qpos0;d.qpos[:3]=pose[:3];d.qpos[3:7]=pose[[6,3,4,5]];d.qpos[addresses]=pose[7:]
   mujoco.mj_forward(model,d);rows.append(d.xpos[bodies].copy())
  positions[key]=np.asarray(rows)
 actual=positions['real_robot_motion36'];reference=positions['active_plan_motion36']
 absolute=np.linalg.norm(actual-reference,axis=-1).mean(1)
 relative=np.linalg.norm((actual-actual[:,:1])-(reference-reference[:,:1]),axis=-1).mean(1)
 xy=np.linalg.norm(actual[:,0,:2]-reference[:,0,:2],axis=-1)
 minimum_height=np.minimum(actual[:,0,2],actual[:,bodies.index(torso),2])
 failure=(minimum_height<=.30)|(relative>=.1)|(xy>=3.)
 receipt=json.loads(a.receipt.read_text())
 latencies=[x['elapsed_s'] for x in receipt['planner_calls']][1:]
 out={'scope':'single prompt, single seed, early checkpoint, 30-body diagnostic; NOT paper benchmark',
      'frames':len(actual),'synchronous_streams_verified':True,'fps_simulation':50,'simulation_seconds':len(actual)/50,
      'body_names':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) for b in bodies],
      'aMPJPE_mean_mm':float(absolute.mean()*1000),'rMPJPE_mean_mm':float(relative.mean()*1000),
      'rMPJPE_max_frame_mm':float(relative.max()*1000),'max_xy_drift_m':float(xy.max()),
      'min_pelvis_or_torso_height_m':float(minimum_height.min()),
      'failure_frames':int(failure.sum()),'diagnostic_full_episode_success':not bool(failure.any()),
      'first_failure_frame':int(np.where(failure)[0][0]) if failure.any() else None,
      'threshold_source':'ReactiveBFM PDF Appendix A1; body selection explicit reconstruction',
      'first_planner_call_including_compilation_s':receipt['planner_calls'][0]['elapsed_s'],
      'warm_planner_mean_s':float(np.mean(latencies)),'warm_planner_max_s':float(np.max(latencies)),
      'effective_replans':receipt['effective_replans'],'input_sha256':sha(a.session),'model_sha256':sha(a.model),
      'source_sha256':sha(__file__),'checkpoint_sha256':receipt['checkpoint_sha256']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out))
if __name__=='__main__':main()
