"""Reconstruct hash-bound source members from bounded stdin frames; not training data."""
import contextlib, hashlib, io, json, os, shutil, sys
from pathlib import Path
import numpy as np
import smplx
import torch
from bmlmovi_stream_protocol_20261102 import read_frame

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def guard():
    if shutil.disk_usage(ROOT).free < 8*1024**3+256*1024**2: raise RuntimeError('disk safety threshold')
def main():
    queue=Path(sys.argv[1]); out=Path(sys.argv[2]); qhash=sha(queue)
    data=json.loads(queue.read_text(encoding='utf8')); rows=data['members']
    if data['training_ready'] is not False or len({r['member'] for r in rows})!=len(rows): raise ValueError('invalid queue')
    expected={r['member']:r for r in rows}; seen=set(); models={}; failures=0
    guard();out.mkdir(exist_ok=False);torch.set_num_threads(1)
    with (out/'records.jsonl').open('x',encoding='utf8',buffering=1) as log:
      while True:
        guard(); frame=read_frame(sys.stdin.buffer)
        if frame is None: break
        head,payload=frame; member=head.get('member'); row=expected.get(member)
        if row is None or member in seen: raise ValueError('unexpected or duplicate member')
        seen.add(member); digest=hashlib.sha256(payload).hexdigest()
        if digest!=row['source_sha256'] or head.get('sha256')!=digest or len(payload)!=row['member_size']: raise ValueError('payload identity mismatch')
        result={'member':member,'source_sha256':digest,'training_ready':False,'caption_accepted':False}
        try:
          with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            fps=float(z['mocap_frame_rate']); gender=str(z['gender'].item()); trans=z['trans'].copy(); orient=z['root_orient'].copy(); body=z['pose_body'].copy(); betas=np.asarray(z['betas'][:16],np.float32)
          n=len(trans)
          if (n!=row['source_frames'] or fps!=row['declared_fps'] or gender not in ('male','female','neutral') or n<2 or n>100000 or trans.shape!=(n,3) or orient.shape!=(n,3) or body.shape!=(n,63) or betas.shape!=(16,) or not all(np.isfinite(x).all() for x in (trans,orient,body,betas))): raise ValueError('invalid source parameters')
          mp=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz'); mh=sha(mp)
          if gender not in models:
            with contextlib.redirect_stdout(sys.stderr): models[gender]=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
          model=models[gender]; chunks=[]
          for start in range(0,n,60):
            guard(); end=min(start+60,n); size=end-start
            t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32); zero=lambda w:torch.zeros((size,w))
            with torch.no_grad():
              p=model(betas=t(np.broadcast_to(betas,(size,16)).copy()),global_orient=t(orient[start:end]),body_pose=t(body[start:end]),transl=t(trans[start:end]),left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),leye_pose=zero(3),reye_pose=zero(3),expression=zero(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
            chunks.append(p)
          human=np.concatenate(chunks); witness=np.array([0,n//2,n-1],dtype=np.int64)
          if human.shape!=(n,22,3) or not np.isfinite(human).all() or sha(mp)!=mh: raise ValueError('invalid output or model changed')
          dest=out/(f'source_{len(seen)-1:04d}.npz')
          with dest.open('xb') as f:
            np.savez_compressed(f,human_joints=human,source_frames=np.arange(n,dtype=np.int64),declared_fps=np.array(fps),witness_frames=witness,witness_trans=trans[witness],witness_orient=orient[witness],witness_body=body[witness],witness_betas=betas,witness_gender=np.array(gender))
          result.update(status='reconstructed_unreviewed',artifact=dest.name,artifact_sha256=sha(dest),frames=n,declared_fps=fps,model_sha256=mh,nominal_rows=row['nominal_rows']);failures=0
        except Exception as e:
          result.update(status='failed',error=f'{type(e).__name__}: {e}');failures+=1
        log.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
        print(json.dumps({'member':member,'status':result['status']}),file=sys.stderr,flush=True)
        if failures>=3: raise RuntimeError('three consecutive failures')
    if seen!=set(expected) or sha(queue)!=qhash: raise ValueError('incomplete or modified queue')
    with (out/'report.json').open('x',encoding='utf8') as f: json.dump({'queue_sha256':qhash,'records_sha256':sha(out/'records.jsonl'),'requested':len(rows),'training_ready':False,'limitations':['Declared clock, locked-head model and zero hand/face poses.','No caption, robot or physics acceptance.']},f,indent=2)
if __name__=='__main__':main()
