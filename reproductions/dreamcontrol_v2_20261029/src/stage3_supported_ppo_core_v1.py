"""Private task references and ordered, measured object-interaction milestones."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from stage3_pick29_core import BODY_JOINTS,interpolate_quat,sha

TASKS=('pour','bimanual_box_lift','box_push_pull')
PRIOR='109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9'

class TaskReferenceBank:
    def __init__(self,path,task,asset_sha,device,minimum=500):
        manifest=json.loads(Path(path).read_text());items=manifest['references']
        assert manifest['task']==task and task in TASKS and len(items)>=minimum
        assert minimum>=2 and (minimum<500 or len(items)==500)
        assert manifest['asset_sha256']==asset_sha
        self.scene=manifest['scene'];self.scene_sha256=manifest['scene_sha256']
        assert hashlib.sha256(json.dumps(self.scene,sort_keys=True,separators=(',',':')).encode()).hexdigest()==self.scene_sha256
        rows=[];positions=[];quats=[];seen={k:set() for k in ['id','noise','feature','state']}
        keys=['q','root_pos','root_quat','keypoints','hand_closed','left_wrist_quat','right_wrist_quat','planned_object_positions']
        for r in items:
            assert r['task']==task and sha(r['path'])==r['sha256']
            proof=json.loads(Path(r['native_proof']).read_text());kin=json.loads(Path(r['kinematic_report']).read_text())
            assert sha(r['native_proof'])==r['native_proof_sha256']
            assert proof['eligible_for_calibrated_scene_reference'] and all(proof['native_checks'].values())
            assert proof['asset_sha256']==asset_sha and proof['scene_sha256']==self.scene_sha256
            assert proof['source_state_sha256']==r['sha256'] and proof['kinematic_report_sha256']==sha(r['kinematic_report'])
            assert kin['output_sha256']==r['sha256'] and kin['passed_requested_kinematic_checks']
            assert sha(kin['sample'])==kin['sample_sha256']
            with np.load(r['path'],allow_pickle=False) as z:
                meta=json.loads(z['metadata_utf8'].tobytes())
                assert meta['task']==task and meta['checkpoint_sha256']==PRIOR
                assert meta['scene_sha256']==self.scene_sha256 and float(z['fps'])==20
                assert meta['initial_noise_sha256']==kin['source_initial_noise_sha256']
                assert meta['feature_content_sha256']==kin['source_feature_content_sha256']
                assert z['joint_names'].tolist()==BODY_JOINTS and z['q'].shape==(180,29)
                assert np.array_equal(z['sample_times'],np.arange(180)/20)
                row={k:z[k].astype(np.float32) for k in keys}
                assert all(row[k].shape==shape for k,shape in {
                    'q':(180,29),'root_pos':(180,3),'root_quat':(180,4),'keypoints':(180,22,3),
                    'hand_closed':(180,2),'left_wrist_quat':(180,4),'right_wrist_quat':(180,4),
                    'planned_object_positions':(180,3)}.items())
                assert all(np.isfinite(v).all() for v in row.values())
                assert np.isin(row['hand_closed'],[0,1]).all()
                for key in ['root_quat','left_wrist_quat','right_wrist_quat']:
                    assert np.max(np.abs(np.linalg.norm(row[key],axis=-1)-1))<1e-4
                state=hashlib.sha256(b''.join(np.ascontiguousarray(z[k]).tobytes() for k in ['q','root_pos','root_quat','hand_closed'])).hexdigest()
                values={'id':meta['candidate_id'],'noise':meta['initial_noise_sha256'],
                        'feature':meta['feature_content_sha256'],'state':state}
                for key,value in values.items():assert value not in seen[key];seen[key].add(value)
                positions.append(z['task_object_position'].copy());quats.append(z['task_object_quaternion_wxyz'].copy());rows.append(row)
        self.data={k:torch.tensor(np.stack([r[k] for r in rows]),device=device) for k in keys}
        self.object_positions=torch.tensor(np.array(positions),device=device,dtype=torch.float32)
        self.object_quaternions=torch.tensor(np.array(quats),device=device,dtype=torch.float32)
        self.count=len(rows);self.lengths=torch.full((self.count,),180,device=device,dtype=torch.long)
        self.manifest_sha256=sha(path)
    def query(self,ids,seconds):
        clock=(seconds.clamp_min(0)*20).clamp_max(179);lo=clock.floor().long();hi=(lo+1).clamp_max(179)
        result={}
        for key,v in self.data.items():
            a,b=v[ids,lo],v[ids,hi];t=(clock-lo).reshape((-1,)+(1,)*(a.ndim-1))
            result[key]=(a if key=='hand_closed' else interpolate_quat(a,b,t)
                         if key.endswith('quat') else torch.lerp(a,b,t))
        result['after_grasp']=seconds>=2
        return result

def empty_milestones(n,device):
    return {k:torch.zeros(n,device=device,dtype=torch.bool) for k in
            ['contact','lifted','tilted','upright','carried','pushed','released','recontacted','success']}

def update_milestones(task,flags,seconds,position,initial,tilt_radians,speed,
                      hand_contacts,hand_closed,wrist_gaps):
    """Inputs are measured physical states; planned object paths are not inputs."""
    delta=position-initial;both=hand_contacts.all(-1);closed=hand_closed.all(-1)
    flat=(delta[:,2].abs()<.035)&(tilt_radians<.3)
    if task=='pour':
        touch=hand_contacts[:,1]&hand_closed[:,1]
        flags['contact']|=touch&(seconds>=1.5)
        flags['lifted']|=flags['contact']&touch&(delta[:,2]>.10)
        flags['tilted']|=flags['lifted']&touch&(tilt_radians>torch.pi/3)&(delta[:,2]>.08)
        flags['upright']|=flags['tilted']&(tilt_radians<.26)&(seconds>5.5)
        flags['success']|=(flags['upright']&(delta.norm(dim=-1)<.04)&(speed<.15)
                           &~hand_closed[:,1]&(seconds>=8))
    elif task=='bimanual_box_lift':
        flags['contact']|=both&closed&(seconds>=1.5)
        flags['lifted']|=flags['contact']&both&closed&(delta[:,2]>.10)
        flags['carried']|=flags['lifted']&both&closed&(delta[:,0]>.08)&(delta[:,2]>.08)
        goal=delta-torch.tensor([.10,0.,0.],device=position.device)
        flags['success']|=(flags['carried']&(goal.norm(dim=-1)<.04)&flat&(speed<.15)
                           &~hand_closed.any(-1)&(seconds>=7.7))
    elif task=='box_push_pull':
        flags['contact']|=both&(seconds>=1.5)
        flags['pushed']|=flags['contact']&flat&(delta[:,0]>.08)&(delta[:,1].abs()<.05)
        flags['released']|=(flags['pushed']&~hand_contacts.any(-1)&~hand_closed.any(-1)
                            &(wrist_gaps.amin(-1)>.045)&(seconds>=4.3))
        flags['recontacted']|=flags['released']&both&closed&(seconds>=5.2)
        flags['success']|=flags['recontacted']&flat&(delta[:,:2].norm(dim=-1)<.025)&(speed<.15)&(seconds>=7.2)
    else:raise ValueError(task)
    return flags
