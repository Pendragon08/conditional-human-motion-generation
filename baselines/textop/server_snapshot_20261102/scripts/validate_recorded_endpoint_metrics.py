"""Small CPU fixtures for terminal off-by-one, joint order and early failure."""
import copy,json
import numpy as np
from recorded_endpoint_metrics import endpoint_metrics
def main():
    names=['joint_'+str(i) for i in range(29)];bodies=['pelvis']+['link_'+str(i) for i in range(29)]
    ref={'joint_pos':np.zeros((3,29)),'body_pos_w':np.zeros((3,30,3)),'body_quat_w':np.tile([1.,0,0,0],(3,30,1))}
    ref['body_pos_w'][:,0,0]=[0,9,10];ref['joint_pos'][-1]=np.arange(29)*.001
    trace={'time_seconds':np.arange(3)/50,'root_pos':np.array([[0.,0,0],[9,0,0],[10,0,0]]),'root_quat_wxyz':np.tile([-1.,0,0,0],(3,1)),
       'joint_pos':ref['joint_pos'][:,::-1].copy(),'joint_names':np.asarray(names[::-1]),'reference_frame':np.array([[0],[0],[1]])}
    trial={'reference_frames':3,'control_steps':2,'complete':True,'failed':False}
    j=endpoint_metrics(trace,ref,trial,names,bodies);assert j['terminal_world_root_position_error_m']==j['terminal_joint_rmse_rad']==j['terminal_root_orientation_geodesic_rad']==0
    checks=['independent_Tminus1_not_consumed_Tminus2','joint_order_by_name','quaternion_sign_invariance']
    short={k:(v[:2] if k!='joint_names' else v) for k,v in trace.items()};early=dict(trial,control_steps=1,complete=False,failed=True)
    j=endpoint_metrics(short,ref,early,names,bodies);assert not j['reference_endpoint_independently_compared'] and 'terminal_joint_rmse_rad' not in j;checks.append('early_failure_not_extrapolated')
    for key,value in [('time_seconds',np.array([0,.02,.05])),('reference_frame',np.array([[0],[1],[2]])),('joint_names',np.asarray(['bad']*29))]:
        bad=copy.deepcopy(trace);bad[key]=value
        try:endpoint_metrics(bad,ref,trial,names,bodies)
        except AssertionError:checks.append('reject_'+key)
        else:raise AssertionError('Invalid fixture accepted: '+key)
    print(json.dumps({'status':'TERMINAL_METRIC_CPU_FIXTURES_PASSED','checks':checks,'count':len(checks),'synthetic_only':True,'new_physical_trials':0,'models_loaded':0}))
if __name__=='__main__':main()
