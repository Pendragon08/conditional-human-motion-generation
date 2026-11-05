"""Restricted read of one verified GRAB sequence; report schema only."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from grab_restricted_reader import read_grab_field


def describe(value):
    if isinstance(value,dict): return {k:describe(v) for k,v in sorted(value.items())}
    if isinstance(value,np.ndarray): return {'type':'array','shape':list(value.shape),'dtype':str(value.dtype)}
    if isinstance(value,(list,tuple)): return {'type':type(value).__name__,'length':len(value)}
    return {'type':type(value).__name__}


def main():
    path=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/data/GRAB/grab__s2.zip')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    expected='d77eb7d13cfb3e3d0b81f08ccf9d8d79320fa6541352e6e65173c56d4b01b9b6'
    if h.hexdigest()!=expected: raise ValueError('GRAB archive hash mismatch')
    with zipfile.ZipFile(path) as archive:
        member=sorted(n for n in archive.namelist() if n.endswith('.npz'))[0]
        fields={name:describe(read_grab_field(archive,member,name)) for name in ('body','object','contact')}
    print(json.dumps({'archive_sha256':h.hexdigest(),'member':member,'fields':fields,
                      'warning':'One archive member with restricted reader, schema only; no SMPL-X reconstruction, contact verification or retargeting.'},allow_nan=False))


if __name__=='__main__': main()
