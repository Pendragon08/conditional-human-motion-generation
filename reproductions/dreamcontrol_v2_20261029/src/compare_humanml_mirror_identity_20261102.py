"""Compare one official sample with a third-party HumanML3D mirror witness."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'humanml_mirror_samples_20261102'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=BASE/'official_012314.npy';b=BASE/'012314.npy'
    x=np.load(a,allow_pickle=False);y=np.load(b,allow_pickle=False)
    same_shape=x.shape==y.shape
    result={'official_sha256':sha(a),'mirror_sha256':sha(b),'official_shape':list(x.shape),'mirror_shape':list(y.shape),
            'official_dtype':str(x.dtype),'mirror_dtype':str(y.dtype),'shapes_equal':same_shape,
            'exact_array_equal':bool(np.array_equal(x,y)) if same_shape else False,
            'max_abs_diff':float(np.max(np.abs(x.astype(float)-y.astype(float)))) if same_shape else None,
            'training_ready':False,'limitations':['One example comparison cannot authenticate other samples.',
                   'Even identical joints do not establish original marker time, source license, or model identities.']}
    with (BASE/'identity_comparison.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
