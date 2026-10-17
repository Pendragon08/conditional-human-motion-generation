"""Round-trip/kinematic consistency, not motion generation or tracking quality."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reimplementation'))
from codec262 import G1Codec,quat_matrix
from features import BlockNormalizer
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
codec=G1Codec(ROOT/'configs/g1_codec_skeleton.json')
rows=[];training=[]
for split in ('train','test'):
 files=sorted((ROOT/'data'/split).glob('*.npz'))
 if not files:raise ValueError('Missing '+split+' data')
 for path in files:
  with np.load(path,allow_pickle=False) as d:motion={k:d[k] for k in d.files}
  fps=float(motion['fps'][0]);native=codec.native_from_motion(motion)
  features,meta=codec.encode(native,fps)
  recovered,decoded,projection=codec.decode(features,meta)
  pos_error=float(np.abs(recovered[:,:3]-native[:,:3]).max())
  rot_error=float(np.abs(quat_matrix(recovered[:,3:7])-quat_matrix(native[:,3:7])).max())
  q_error=float(np.abs(recovered[:,13:42]-native[:,13:42]).max())
  fk_error=float(np.linalg.norm(decoded['body_pos_w']-motion['body_pos_w'],axis=-1).max())
  if max(pos_error,rot_error,q_error)>2e-4 or fk_error>5e-4:
   raise AssertionError((path,pos_error,rot_error,q_error,fk_error))
  # Velocity channels are discrete reconstruction; do not pretend exact physical
  # solver velocities can be recovered from the lossy descriptor.
  velocity_error=float(np.abs(recovered[:,7:13]-native[:,7:13]).mean())
  out=a.output/split/path.stem;out.parent.mkdir(exist_ok=True)
  np.savez_compressed(str(out)+'_262.npz',features=features)
  Path(str(out)+'_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
  rows.append({'split':split,'motion_file':str(path),'motion_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'frames':len(features),'position_max_abs_m':pos_error,'rotation_matrix_max_abs':rot_error,
    'joint_max_abs_rad':q_error,'fk_body_position_max_m':fk_error,
    'velocity_reconstruction_mean_abs':velocity_error,'projection':projection})
  if split=='train':training.append(features)
normalizer=BlockNormalizer.fit(training)
np.savez(a.output/'normalization.npz',mean=normalizer.mean,std=normalizer.std)
report={'status':'COMPLETED','evaluation_type':'self_supervised_proxy',
 'purpose':'descriptor round-trip and URDF-vs-stored-FK consistency only',
 'schema':meta['schema'],'author_exact_codec_reproduced':False,'normalization_fit_split':'train_only',
 'normalization_training_sequences':len(training),'sequence_count':len(rows),'rows':rows,
 'max_position_error_m':max(r['position_max_abs_m'] for r in rows),
 'max_joint_error_rad':max(r['joint_max_abs_rad'] for r in rows),
 'max_fk_error_m':max(r['fk_body_position_max_m'] for r in rows),
 'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
  ('reimplementation/codec262.py','configs/g1_codec_skeleton.json','scripts/validate_codec262.py')}}
(a.output/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('PHYGILE_CODEC262_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('rows','source_sha256')}))
