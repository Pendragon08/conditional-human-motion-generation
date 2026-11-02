"""Candidate index agreement using horizontal root-displacement magnitudes.

The SnapMoGen feature example applies only yaw rotations to root velocity, so
the horizontal norm is invariant. Inputs are caller-authenticated numeric data;
this module neither reads files nor imports NumPy, models or simulator SDKs.
"""
import math
def compare(root_positions,feature_horizontal_velocity):
 if len(feature_horizontal_velocity)<32 or len(root_positions)<len(feature_horizontal_velocity)+2:raise ValueError('insufficient equal coverage')
 ys=[]
 for pair in feature_horizontal_velocity:
  if len(pair)!=2 or not all(math.isfinite(float(v)) for v in pair):raise ValueError('invalid feature velocity')
  ys.append(math.hypot(float(pair[0]),float(pair[1])))
 if not all(len(x)==3 and all(math.isfinite(float(v)) for v in x) for x in root_positions):raise ValueError('invalid root positions')
 peak=max(ys)
 if peak<=1e-10:return dict(status='INCONCLUSIVE',reason='no horizontal motion',candidates=[])
 use=[i for i,y in enumerate(ys) if y>peak*1e-5]
 if len(use)<32:return dict(status='INCONCLUSIVE',reason='too few moving frames',candidates=[])
 candidates=[]
 for offset in (0,1):
  xs=[];selected=[]
  for i in use:
   a=root_positions[i+offset];b=root_positions[i+offset+1]
   xs.append(math.hypot(float(b[0])-float(a[0]),float(b[2])-float(a[2])));selected.append(ys[i])
  xx=sum(x*x for x in xs);yy=sum(y*y for y in selected)
  if xx<=1e-20 or yy<=1e-20:return dict(status='INCONCLUSIVE',reason='insufficient candidate motion',candidates=[])
  scale=sum(x*y for x,y in zip(xs,selected))/xx
  err=math.sqrt(sum((y-scale*x)**2 for x,y in zip(xs,selected))/yy)
  candidates.append(dict(offset=offset,positive_scale=scale,normalized_rmse=err))
 order=sorted(candidates,key=lambda x:x['normalized_rmse']);best,other=order
 if best['positive_scale']<=0 or best['normalized_rmse']>0.001 or other['normalized_rmse']<max(0.01,best['normalized_rmse']*10):
  return dict(status='INCONCLUSIVE',reason='candidate errors not sufficiently separated',moving_frames=len(use),candidates=candidates)
 return dict(status='OFFSET_IDENTIFIED_FOR_TESTED_SOURCE',raw_frame_offset=best['offset'],moving_frames=len(use),candidates=candidates)
