"""Positive/negative contact detector fixtures; no policy score or model edits."""
import json
import numpy as np
import mujoco
from r3_sim2sim_completed_mujoco import CompletedTransfer, RESULT

rows=[]
for task in ['Sit','Pick','Bimanual-Pick']:
    t=CompletedTransfer(task);t.reset(0)
    if t.object_id>=0:
        obj=t.meta['object'];mass=float(t.model.body_mass[t.object_id])
        assert abs(mass-obj['mass'])<1e-6
        j=mujoco.mj_name2id(t.model,mujoco.mjtObj.mjOBJ_JOINT,'task_object_free')
        assert t.model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE
        t.data.qpos[t.object_qadr:t.object_qadr+3]=t.data.xpos[t.body_ids['right_wrist_yaw_link']]
        mujoco.mj_forward(t.model,t.data)
        hand_positive='right' in t.contacts()[1]
        t.data.qpos[t.object_qadr:t.object_qadr+3]=[100,100,100]
        mujoco.mj_forward(t.model,t.data)
        hand_negative=not t.contacts()[1]
        t.data.qpos[t.object_qadr:t.object_qadr+3]=[100,100,-.01]
        mujoco.mj_forward(t.model,t.data)
        floor_positive=False
        for c in t.data.contact[:t.data.ncon]:
            names={mujoco.mj_id2name(t.model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in [c.geom1,c.geom2]}
            if {'floor','task_object_geom'}.issubset(names):floor_positive=True
        row={'task':task,'free_joint':True,'mass_matches_actual_isaac_sample':True,
            'right_hand_overlap_detected':hand_positive,'distant_object_no_hand_contact':hand_negative,
            'object_floor_collision_detected':floor_positive,'score_episodes_modified':False}
        row['passed']=hand_positive and hand_negative and floor_positive
    else:
        seat=next(x for x in t.meta['statics'] if x['name']=='seat')
        t.data.qpos[:3]=np.asarray(seat['position'])+[0,0,seat['size'][2]/2]
        mujoco.mj_forward(t.model,t.data);positive=t.contacts()[2]
        t.data.qpos[:3]=[100,100,100];mujoco.mj_forward(t.model,t.data);negative=not t.contacts()[2]
        row={'task':task,'seat_overlap_detected':positive,'distant_pelvis_no_seat_contact':negative,
             'score_episodes_modified':False,'passed':positive and negative}
    rows.append(row)
(RESULT/'contact_detector_fixtures.json').write_text(json.dumps({'rows':rows,'passed':all(x['passed'] for x in rows)},indent=2))
print('COMPLETED_CONTACT_FIXTURES',json.dumps(rows),flush=True)
assert all(x['passed'] for x in rows)
