"""Bounded private qpos journal from real pre-autoreset states.

Keep every executed pose (36 float32 values) and streaming current-target
errors. Avoid duplicating whole-library arrays into NPZ files. This does not
change physics, termination, targets, policy or acceptance thresholds.
"""
import hashlib,json,pathlib
import numpy as np
from strict_tracking_coverage import verify_trial

FIELDS=[('reference_frame',(1,)),('root_pos',(3,)),('root_quat_wxyz',(4,)),
        ('joint_pos',(29,)),('reference_joint_pos',(29,)),('actions',(29,)),
        ('body_pos',(14,3)),('reference_body_pos',(14,3))]
class ExecutionTrace:
    def __init__(self,folder,motion_id,reference_frames,joint_names,runtime_kind='isaaclab_physics',budget=None):
        self.folder=pathlib.Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        assert len(joint_names)==len(set(joint_names))==29 and reference_frames>=2
        assert runtime_kind in ['isaaclab_physics','synthetic_cpu_fixture']
        assert len(motion_id)==64 and all(c in '0123456789abcdef' for c in motion_id)
        self.budget=budget;assert budget is not None
        self.path=self.folder/'qpos.private.bin';self.stream=self.path.open('xb')
        self.samples=0;self.body_sum=0.;self.body_max=0.;self.joint_sq_sum=0.
        self.meta=dict(schema='textop.compact_physics_trace.v1',status='INCOMPLETE',motion_id=motion_id,
            reference_frames=reference_frames,joint_names=list(joint_names),control_hz=50,
            runtime_kind=runtime_kind,phase='INITIAL_THEN_PRE_AUTORESET_TERMINATION_COMPUTE',
            qpos_layout='root_xyz,root_quaternion_wxyz,joints29',row_width=36,dtype='little_endian_float32',
            samples=0,final_reference_endpoint_comparison_performed=False)
        self._save()
    def _save(self):
        p=self.folder/'trace.private.json';t=p.with_name(p.name+'.partial')
        t.write_text(json.dumps(self.meta,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
    def append(self,state):
        assert not self.stream.closed and self.samples<self.meta['reference_frames']
        values={}
        for k,shape in FIELDS:
            v=np.asarray(state[k],dtype='<f4');assert v.shape==shape and np.isfinite(v).all(),k
            values[k]=v
        assert float(values['reference_frame'][0])==max(0,self.samples-1)
        assert .99<float(np.linalg.norm(values['root_quat_wxyz']))<1.01
        row=np.concatenate([values[k].reshape(-1) for k in ['root_pos','root_quat_wxyz','joint_pos']]).astype('<f4',copy=False)
        assert row.shape==(36,) and self.budget['used']+row.nbytes<=self.budget['maximum']
        self.stream.write(row.tobytes());self.stream.flush();self.budget['used']+=row.nbytes
        if self.samples:
            e=np.linalg.norm(values['body_pos']-values['reference_body_pos'],axis=-1)
            self.body_sum+=float(np.sum(e,dtype=np.float64));self.body_max=max(self.body_max,float(e.max()))
            d=values['joint_pos'].astype(np.float64)-values['reference_joint_pos'].astype(np.float64)
            self.joint_sq_sum+=float(np.sum(d*d))
        self.samples+=1;self.meta['samples']=self.samples
        # The binary is flushed every frame; bounded metadata every50 frames.
        if self.samples==1 or self.samples%50==0:self._save()
    def finish(self,trial):
        verify_trial(trial,self.meta['reference_frames'],0)
        assert self.samples==trial['control_steps']+1 and trial['control_steps']>0
        self.stream.close();assert self.path.stat().st_size==self.samples*36*4
        metrics=dict(aligned_14_body_position_error_mean_m=self.body_sum/(14*trial['control_steps']),
            aligned_14_body_position_error_max_m=self.body_max,
            joint_rmse_rad=(self.joint_sq_sum/(29*trial['control_steps']))**.5,
            definition='Streaming current-control-target errors before autoreset; not final endpoint or paper semantic scores')
        digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.meta.update(status='FINISHED_PHYSICAL_TRIAL' if self.meta['runtime_kind']=='isaaclab_physics' else 'FINISHED_SYNTHETIC_FIXTURE',
            trial=trial,metrics=metrics,binary_sha256=digest,binary_bytes=self.path.stat().st_size,
            consumed_reference_frame_min=0,consumed_reference_frame_max=trial['control_steps']-1,
            reference_final_frame=self.meta['reference_frames']-1)
        self._save()
        return dict(folder=str(self.folder),samples=self.samples,binary_sha256=digest,binary_bytes=self.path.stat().st_size,
            metrics=metrics,runtime_kind=self.meta['runtime_kind'],physical_failure=trial['failed'])
    def close_incomplete(self):
        if not self.stream.closed:self.stream.close()
        if self.meta['status']=='INCOMPLETE':
            self.meta['samples']=self.samples;self.meta['binary_sha256']=hashlib.sha256(self.path.read_bytes()).hexdigest();self._save()
