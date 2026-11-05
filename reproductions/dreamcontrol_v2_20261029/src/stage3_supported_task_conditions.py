"""Explicit position, orientation and hand schedules for three supported tasks.

These are planned project scenes, not measured object trajectories. Only the
position channels enter DDPM; orientations and hands must be fitted and verified
later. A generated candidate is never reference admission or interaction success.
"""
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

TASKS=('pour','bimanual_box_lift','box_push_pull')
PROMPTS={
 'pour':'A person picks up a water bottle and pours out its contents.',
 'bimanual_box_lift':'Lift the largebox, move the largebox, and put down the largebox.',
 'box_push_pull':'Push the largebox, release the hands, then pull the largebox, and set it back down.'}
FPS=20;FRAMES=180

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def path(times,knots,values):
 values=np.asarray(values,dtype=float)
 return np.stack([np.interp(times,knots,values[:,i]) for i in range(values.shape[1])],axis=-1)

def build(task,ids,seed,initial,initial_rotations):
 assert task in TASKS
 times=np.arange(FRAMES)/FPS;world=np.broadcast_to(initial,(len(ids),FRAMES,22,3)).copy()
 table={'center':[.45,0.,.4],'size':[.6,.6,.8]}
 scene={'task':task,'table':table,'ground_z':0.,'coordinate_frame':'simulator XYZ Z-up',
  'scenario_scope':'bounded project tabletop pilot, not a published full task distribution',
  'object_motion_semantics':'desired object path for planning; not actual simulated motion',
  'hand_contact_calibration':'pending native43-joint asset verification; no grasp success claim'}
 if task=='pour':
  center=np.array([.35,-.20,.90]);size=[.05,.05,.20]
  scene.update(object_type='waterbottle_proxy',object_size=size,receiver_center=[.35,-.08,.84],receiver_size=[.10,.10,.08])
 else:
  center=np.array([.32,0.,.90]);size=[.26,.38,.20]
  scene.update(object_type='largebox_proxy',object_size=size)
 scene['nominal_object_center']=center.tolist();scene_sha=digest(scene)
 object_initial=[];desired_objects=[];orientations=[];hands=[];plans=[]
 for b,i in enumerate(ids):
  delta=np.r_[np.random.default_rng(seed+i).uniform(-.025,.025,2),0.]
  c=center+delta;object_initial.append(c)
  if task=='pour':
   obj=path(times,[0,2,3.5,4.5,5.5,7,8,8.95],[c,c,c+[0,.08,.16],c+[0,.08,.16],c+[0,.08,.16],c+[0,0,.16],c,c])
   wrist=obj+np.array([0.,0.,.04]);world[b,:,21]=path(times,[0,2,3.5,4.5,5.5,7,8,8.95],
    [initial[21],wrist[40],wrist[70],wrist[90],wrist[110],wrist[140],wrist[160],wrist[-1]])
   angle=np.interp(times,[0,2,3.5,4.5,5.5,6.5,8.95],[0,0,0,-75,-75,0,0])
   right=Rotation.from_euler('x',angle,degrees=True)*Rotation.from_euler('y',90,degrees=True)
   left=Rotation.from_euler('y',90,degrees=True)
   contact=((times>=2)&(times<8));hand=np.column_stack([np.zeros(FRAMES,dtype=int),contact.astype(int)])
   phases={'contact_s':2.,'lift_s':3.5,'tilt_s':4.5,'tilt_hold_end_s':5.5,'upright_s':6.5,'place_s':8.,'tilt_world_x_degrees':-75.}
  elif task=='bimanual_box_lift':
   obj=path(times,[0,2,3.5,5.5,7.5,8.95],[c,c,c+[0,0,.14],c+[.10,0,.14],c+[.10,0,0],c+[.10,0,0]])
   for index,side in [(20,1),(21,-1)]:
    offset=np.array([0,side*(size[1]/2+.025),0])
    wrist=obj+offset
    world[b,:,index]=path(times,[0,2,3.5,5.5,7.5,8.3,8.95],[initial[index],wrist[40],wrist[70],wrist[110],wrist[150],wrist[150]+[0,side*.05,0],wrist[150]+[0,side*.05,0]])
   right=Rotation.from_euler('zy',[-90,90],degrees=True);left=Rotation.from_euler('zy',[90,90],degrees=True)
   contact=((times>=2)&(times<7.7));hand=np.column_stack([contact,contact]).astype(int)
   phases={'two_hand_contact_s':2.,'lift_s':3.5,'carry_end_s':5.5,'place_s':7.5,'release_s':7.7,'lift_height_m':.14,'carry_displacement_m':[.10,0,0]}
  else:
   obj=path(times,[0,2,3.5,4.5,5.2,7.2,8.95],[c,c,c+[.10,0,0],c+[.10,0,0],c+[.10,0,0],c,c])
   for index,side in [(20,1),(21,-1)]:
    offset=np.array([-size[0]/2-.025,side*.105,0])
    contact=obj+offset
    world[b,:,index]=path(times,[0,2,3.5,4.3,4.8,5.2,7.2,8,8.95],
     [initial[index],contact[40],contact[70],contact[70]+[-.05,0,.03],contact[70]+[-.05,0,.03],contact[104],contact[144],contact[144]+[-.05,0,.03],contact[144]+[-.05,0,.03]])
   right=Rotation.from_euler('y',90,degrees=True);left=Rotation.from_euler('y',90,degrees=True)
   contact=((times>=5.2)&(times<7.4));hand=np.column_stack([contact,contact]).astype(int)
   phases={'push_contact_s':2.,'push_end_s':3.5,'release_start_s':4.3,'pull_contact_s':5.2,'return_s':7.2,'push_displacement_m':[.10,0,0]}
  matrices=[]
  for rotation,index in [(left,20),(right,21)]:
   q=rotation.as_quat();q=np.broadcast_to(q,(FRAMES,4)).copy() if q.ndim==1 else q
   approach=Slerp([0,2],Rotation.from_quat(np.stack([Rotation.from_matrix(initial_rotations[index]).as_quat(),q[40]])))(np.minimum(times,2)).as_quat()
   q=np.where((times<=2)[:,None],approach,q)
   matrices.append(q[:,[3,0,1,2]])
  orientations.append(np.stack(matrices,axis=1));desired_objects.append(obj);hands.append(hand)
  plans.append({'task':task,'candidate_id':int(i),'scene_sha256':scene_sha,'phases':phases,
   'object_initial_position':c.tolist(),'position_targets_enter_diffusion':True,
   'orientation_targets':'postoptimization_only; DDPM hint has position channels only',
   'hand_schedule':'planned binary commands, not verified grasp/contact',
   'source_caption':PROMPTS[task],'final_reference_admission':False})
 mask=np.zeros((len(ids),FRAMES,22),dtype=bool);mask[:,:,0]=True;mask[:,:,21]=True
 if task!='pour':mask[:,:,20]=True
 mask[:,::5,10:12]=True  # actual left/right ankle-roll points; not knee indices4/5
 return {'world':world,'mask':mask,'object_initial':np.asarray(object_initial),
  'desired_object_positions':np.asarray(desired_objects),'desired_wrist_quat_wxyz':np.asarray(orientations),
  'planned_hand_closed':np.asarray(hands,dtype=np.int8),'plans':plans,'scene':scene,'scene_sha256':scene_sha,'prompt':PROMPTS[task]}
