"""Read-only input magnitude diagnosis after anomalous full-library PPO loss.

Original poses only; no physics, filtering, deletion or training. Per-motion
identifiers/measurements stay private. This cannot diagnose executed contacts.
"""
import argparse,datetime,hashlib,io,json,os,pathlib,tarfile,time,traceback
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np
from session import REPO
from audit_preprocessing_backups import sha

B=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
def stats(rows):
 result={'sources':len(rows),'pose_frames':sum(x['frames'] for x in rows),
         'sources_with_any_joint_speed_over_20_rad_s':sum(x['max_joint_speed_rad_s']>20 for x in rows),
         'finite_difference_frames_with_any_joint_speed_over_20_rad_s':sum(x['speed_violation_frames'] for x in rows),
         'sources_with_root_below_zero':sum(x['min_root_height_m']<0 for x in rows),
         'minimum_root_height_m':min(x['min_root_height_m'] for x in rows)}
 for field in ['max_joint_speed_rad_s','max_root_linear_speed_m_s','max_root_angular_speed_rad_s','max_abs_joint_position_rad','max_abs_root_position_m']:
  values=np.array([x[field] for x in rows]);result[field]={'max':float(values.max()),'median_of_source_maxima':float(np.median(values)),
       'p95_of_source_maxima':float(np.quantile(values,.95)),'p99_of_source_maxima':float(np.quantile(values,.99))}
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert a.run.replace('_','').isalnum()
 out=B/'results/reference_input_magnitudes_20261026'/a.run;out.mkdir(parents=True,exist_ok=False)
 private=LOCAL/'datasets/input_magnitude_diagnostics_private_20261026'/a.run;private.mkdir(parents=True,exist_ok=False)
 base=B/'results/preprocessing_backup_audit_20261025/after_kit_v1/receipt.json';accepted=json.loads(base.read_text(encoding='utf-8'))
 cfg=json.loads((B/'configs/preprocessing-completion-audit.json').read_text(encoding='utf-8'))
 assert accepted['preprocessing_complete'] and accepted['verified_sources']==8278
 by_id={x['id']:x for x in accepted['cohorts']};rows=[];cohorts=[];start=time.monotonic()
 receipt={'status':'RUNNING','read_only':True,'gpu_used':False,'physics_steps':0,'optimizer_updates':0,
          'physical_filter_executed':False,'dataset_accepted':False,'full_generator_training_started':False,
          'preprocessing_receipt_sha256':sha(base),'source_sha256':sha(pathlib.Path(__file__))}
 try:
  for spec in cfg['cohorts']:
   archive=LOCAL/'datasets'/spec['archive'];assert sha(archive)==by_id[spec['id']]['archive_sha256']
   local=[];manifest=None
   with tarfile.open(archive,mode='r|gz') as stream:
    for member in stream:
     if not member.isfile():continue
     name=pathlib.PurePosixPath(member.name).name
     if name=='prepared_manifest.private.jsonl':
      manifest={pathlib.PurePosixPath(x['output_file']).name:x for x in map(json.loads,stream.extractfile(member).read().decode().splitlines())}
     elif name.endswith('.private.npz'):
      blob=stream.extractfile(member).read()
      with np.load(io.BytesIO(blob),allow_pickle=False) as data:q=data['qpos_29'].astype(np.float64)
      assert q.shape[1]==36 and len(q)>=2 and np.isfinite(q).all()
      jv=np.abs(np.diff(q[:,7:],axis=0))*50;rv=np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1)*50
      quat=q[:,3:7];quat=quat/np.linalg.norm(quat,axis=1,keepdims=True)
      av=2*np.arccos(np.clip(np.abs(np.sum(quat[:-1]*quat[1:],axis=1)),0,1))*50
      local.append({'payload':name,'sha256':hashlib.sha256(blob).hexdigest(),'frames':len(q),
         'max_joint_speed_rad_s':float(jv.max()),'max_root_linear_speed_m_s':float(rv.max()),
         'max_root_angular_speed_rad_s':float(av.max()),'min_root_height_m':float(q[:,2].min()),
         'max_abs_root_position_m':float(np.abs(q[:,:3]).max()),'max_abs_joint_position_rad':float(np.abs(q[:,7:]).max()),
         'speed_violation_frames':int((jv>20).any(axis=1).sum())})
   assert manifest is not None and len(local)==len(manifest)==spec['target']
   for row in local:
    prior=manifest[row['payload']];assert row['sha256']==prior['output_sha256'] and row['frames']==prior['frames50']
    row.update(canonical_source=prior['canonical_source'],split=prior['split'])
   (private/(spec['id']+'.private.json')).write_text(json.dumps(local,indent=2),encoding='utf-8')
   rows+=local;cohorts.append({'id':spec['id'],**stats(local)})
  assert len(rows)==8278
  receipt.update(status='REFERENCE_INPUT_MAGNITUDE_AUDIT_FINISHED',cohorts=cohorts,aggregate=stats(rows),
      output_writers_closed=True,
      limitation='Original preprocessed poses and 50Hz finite differences only. Threshold20rad/s is an upstream reward reference, not an acceptance rule. No simulator contact/reward/state evidence; cannot establish the cause of anomalous PPO loss.')
 except BaseException:
  (private/'failure.private.txt').write_text(traceback.format_exc(),encoding='utf-8')
  receipt.update(status='FAILED',output_writers_closed=True);raise
 finally:
  receipt.update(elapsed_seconds=time.monotonic()-start,time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
  (out/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
