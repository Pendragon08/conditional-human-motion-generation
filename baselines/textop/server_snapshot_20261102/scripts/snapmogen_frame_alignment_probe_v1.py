"""Draft: compare official feature/BVH timelines only under the common lock."""
import argparse,hashlib,json,os,pathlib,sys
if sys.flags.optimize or not sys.flags.isolated:raise RuntimeError('isolated interpreter required')
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');RUN='user_snapmogen_frame_alignment_probe_v1'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True);a=p.parse_args();request=pathlib.Path(a.request)
if os.getuid()!=1013 or request.resolve()!=request or request!=ROOT/'receipts'/(RUN+'_request.json') or sha(request)!=a.trusted_request_sha256:raise ValueError('request identity')
j=json.loads(request.read_text());out=ROOT/'step34_20261029'/RUN
if j['run']!=RUN or j['authorized'] is not True or out.resolve()!=out or out.stat().st_uid!=1013:raise ValueError('scope')
sys.path.insert(0,str(ROOT/'scripts'))
from augmentation5368_audit_binding_v1 import lock_owned_by_supervisor
st=(ROOT/'receipts/full_library_capacity.lock').stat()
if not lock_owned_by_supervisor(pathlib.Path('/proc/locks').read_text().splitlines(),os.getppid(),os.major(st.st_dev),os.minor(st.st_dev),st.st_ino):raise ValueError('supervisor lock')
for name,digest in j['bound_sources'].items():
 q=ROOT/'scripts'/name
 if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=digest:raise ValueError('source changed '+name)
from snapmogen_index_alignment_contract_v1 import compare
site=ROOT/'step34_20261029/user_snapmogen_python311_runtime_v1/env_v1/site';sys.path.insert(0,str(site))
import numpy as np
if site not in pathlib.Path(np.__file__).resolve().parents or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU environment')
expected=('m_st3_00000','st3_00000','st3_00002','m_st3_00002','st3_00004')
if tuple(x['name'] for x in j['samples'])!=expected:raise ValueError('sample order')
def read_root(path,expected_frames):
 positions=[];channels=False;motion=False;declared=None;time=None
 with path.open('r',encoding='utf-8') as f:
  for line in f:
   s=line.strip()
   if not motion:
    if s=='CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation' and not channels:channels=True
    if s=='MOTION':motion=True
   elif declared is None and s.startswith('Frames:'):declared=int(s.split(':',1)[1])
   elif time is None and s.startswith('Frame Time:'):time=float(s.split(':',1)[1])
   elif time is not None and s:
    fields=s.split()
    if len(fields)<6:raise ValueError('BVH motion line')
    positions.append(tuple(float(v) for v in fields[:3]))
 if not(channels and motion and declared==expected_frames and len(positions)==declared and time and time>0):raise ValueError('BVH header or frame count')
 return positions,time
results=[]
for item in j['samples']:
 name=item['name'];bvh=ROOT/'step34_20261029/user_snapmogen_full_sources_v1'/(name+'.bvh');feature=ROOT/'step34_20261029/user_snapmogen_feature_alignment_v1'/(name+'.npy')
 if bvh.resolve()!=bvh or feature.resolve()!=feature or bvh.stat().st_uid!=1013 or feature.stat().st_uid!=1013 or sha(bvh)!=item['bvh_sha256'] or sha(feature)!=item['feature_sha256']:raise ValueError('sample identity')
 root,period=read_root(bvh,item['bvh_frames']);a=np.load(feature,allow_pickle=False)
 if a.ndim!=2 or a.shape!=(item['bvh_frames']-2,296) or a.dtype!=np.float64 or not np.isfinite(a[:,:3]).all():raise ValueError('published feature format')
 result=compare(root,a[:,1:3]);result.update(name=name,bvh_sha256=item['bvh_sha256'],feature_sha256=item['feature_sha256'],bvh_frames=len(root),feature_frames=len(a),frame_time=period)
 if sha(bvh)!=item['bvh_sha256'] or sha(feature)!=item['feature_sha256']:raise ValueError('source changed')
 results.append(result)
independent=[x for x in results if not x['name'].startswith('m_')];offsets=[x.get('raw_frame_offset') for x in results if x['status']=='OFFSET_IDENTIFIED_FOR_TESTED_SOURCE']
consistent=(len(independent)>=3 and len(offsets)==len(results) and len(set(offsets))==1)
value=dict(status='SNAPMOGEN_FEATURE_ALIGNMENT_MEASURED_AWAITING_AUDIT',samples=results,independent_source_count=len(independent),raw_frame_offset=offsets[0] if consistent else None,decision='CONSISTENT_FOR_TESTED_SOURCES' if consistent else 'INCONCLUSIVE',source_scope='five files, three unmirrored motion sequences; not proof for every published feature file',motion_arrays_persisted=False,policy_loaded=False,simulator_loaded=False)
with (out/'receipt.json').open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
print(json.dumps(value))
