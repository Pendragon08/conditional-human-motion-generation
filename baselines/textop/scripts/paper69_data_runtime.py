"""Process-local fixes for the pinned official motion sampler's index bounds.

Use the existing official primitive extraction/encoding/text-selection math.
Sample only valid sources, include the last legal segment start, and allow a
35pose-frame source to supply its sole history2/future8/four-primitive window.
No corpus, weights or shared source changes occur when importing this module.
"""
import random
import numpy as np
import torch

def install_data_sampling():
 from robotmdar.dataloader.data import SkeletonPrimitiveDataset
 def sample(self,generator=None):
  valid=list(self.valid_indices);assert valid,'No valid motion sources'
  if self.weighted_sample:
   weights=np.asarray(self.seq_weights,dtype=np.float64)[valid]
   assert np.isfinite(weights).all() and (weights>=0).all() and weights.sum()>0
   positions=np.random.choice(len(valid),size=self.batch_size,replace=True,p=weights/weights.sum()).tolist()
  else:positions=torch.randint(0,len(valid),(self.batch_size,),generator=generator).tolist()
  result=[]
  for position in positions:
   source=self.raw_data[valid[position]];last=source['length']-self.segment_len;assert last>=0
   if self.weighted_sample and self.frame_weight:
    weights=source['frame_weights'];assert len(weights)==last+1
    start=random.choices(range(last+1),weights=weights,k=1)[0]
   else:start=int(torch.randint(0,last+1,(1,),generator=generator).item())
   result.append(self._generate_motion_primitives(source,start))
  return result
 SkeletonPrimitiveDataset._sample_motion_batch=sample
 return {'valid_index_mapping_fixed':True,'last_legal_start_included':True,'exact35_pose_frames_supported':True,
         'official_primitive_extraction_preserved':True,'upstream_files_modified':False}
