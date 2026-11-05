"""Verify GRAB ObjectModel row-vector transform on real S3 object samples."""
import hashlib,json,zipfile,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rodrigues
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_human_source_20261030'))
from grab_restricted_reader import read_grab_field

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    archive=ROOT/'data/GRAB/grab__s3.zip';receipt=json.loads((ROOT/'grab_s3_human_source_20261030/report.json').read_text())
    if sha(archive)!=receipt['archive_sha256']:raise ValueError('archive hash mismatch')
    with zipfile.ZipFile(archive) as z: obj=read_grab_field(z,receipt['source_member'],'object')
    frames=np.array([264,300,340,381]);rv=np.asarray(obj['params']['global_orient'])[frames];tr=np.asarray(obj['params']['transl'])[frames]
    v=np.array([[.02,.01,-.01],[-.05,.03,.02],[.01,-.07,.05]])
    with torch.no_grad():matrix=batch_rodrigues(torch.as_tensor(rv,dtype=torch.float64)).numpy()
    r=Rotation.from_rotvec(rv)
    matrix_err=float(np.abs(matrix-r.as_matrix()).max())
    official=np.einsum('vj,tjk->tvk',v,matrix)+tr[:,None,:]
    candidate=np.stack([r[i].inv().apply(v)+tr[i] for i in range(len(frames))])
    inverse=np.stack([r[i].apply(official[i]-tr[i]) for i in range(len(frames))])
    wrong=np.stack([r[i].apply(v)+tr[i] for i in range(len(frames))])
    err=float(np.abs(official-candidate).max());inverse_err=float(np.abs(inverse-v[None]).max())
    if err>1e-7 or inverse_err>1e-7 or matrix_err>1e-7:raise ValueError('transform mismatch')
    doc={'archive_sha256':sha(archive),'frames':frames.tolist(),'script_sha256':sha(__file__),
         'reference':'https://github.com/otaheri/GRAB/blob/master/tools/objectmodel.py',
         'official_convention':'row vertex @ Rodrigues rotation + translation',
         'matrix_max_difference':matrix_err,'forward_max_difference_m':err,
         'inverse_max_difference_m':inverse_err,'old_forward_max_difference_m':float(np.abs(official-wrong).max()),
         'warning':'Checks mathematical transform against GRAB official code, not mesh order, contact labels or physical ground truth.'}
    with Path(__file__).with_name('rotation_test.json').open('x',encoding='utf8') as f:json.dump(doc,f,indent=2,allow_nan=False)
    print(json.dumps(doc),flush=True)
if __name__=='__main__':main()
