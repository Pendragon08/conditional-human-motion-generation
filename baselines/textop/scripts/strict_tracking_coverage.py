"""Contiguous whole-motion accounting for the future strict rollout executor.

This bookkeeping performs no simulation. An episode timeout is incomplete
evidence, never a successful physical trial. A simultaneous failure wins over
natural motion completion. Start at frame0; Tframes require T-1transitions.
"""
from dataclasses import dataclass

@dataclass
class StrictCoverage:
 reference_frames:int
 repeat:int=0
 steps:int=0
 end_reason:str='running'
 def __post_init__(self):
  assert type(self.reference_frames) is int and self.reference_frames>=2
  assert type(self.repeat) is int and self.repeat>=0
 def observe(self,*,terminated=False,motion_end=False,external_timeout=False):
  assert self.end_reason=='running','A finished trial must not collect a new episode'
  assert all(type(x) is bool for x in [terminated,motion_end,external_timeout])
  self.steps+=1;assert self.steps<=self.reference_frames-1
  if terminated:self.end_reason='termination'
  elif external_timeout:self.end_reason='external_timeout'
  elif motion_end:
   assert self.steps==self.reference_frames-1,'Natural end reported before entire motion coverage'
   self.end_reason='motion_end'
  elif self.steps==self.reference_frames-1:raise AssertionError('Executor failed to identify natural motion completion')
 def record(self):
  assert self.end_reason!='running','No incomplete running trial may become a physical decision'
  return {'repeat':self.repeat,'start_frame':0,'reference_frames':self.reference_frames,
          'control_steps':self.steps,'end_reason':self.end_reason,'failed':self.end_reason=='termination',
          'complete':self.end_reason=='motion_end'}

def verify_trial(trial,reference_frames,repeat):
 assert trial['repeat']==repeat and trial['start_frame']==0
 assert type(reference_frames) is int and reference_frames>=2 and trial['reference_frames']==reference_frames
 steps=trial['control_steps'];assert type(steps) is int and 1<=steps<=reference_frames-1
 assert type(trial['failed']) is bool and type(trial['complete']) is bool
 if trial['end_reason']=='motion_end':
  assert steps==reference_frames-1 and trial['complete'] and not trial['failed']
 elif trial['end_reason']=='termination':
  assert trial['failed'] and not trial['complete']
 else:raise AssertionError('External timeout or unknown end is incomplete strict evidence')
 return trial['failed']
