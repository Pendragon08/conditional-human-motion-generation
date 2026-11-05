"""Inspect GRAB object-array NPY headers and pickle opcodes without unpickling.

Never execute pickle instructions or import classes named by the data. This is
metadata only and is not validation or retargeting of human motion.
"""
import hashlib
import io
import json
from pathlib import Path
import pickletools
import zipfile
import numpy as np


def main():
    path=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/data/GRAB/grab__s2.zip')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    with zipfile.ZipFile(path) as outer:
        names=sorted(n for n in outer.namelist() if n.endswith('.npz'))
        if not names: raise ValueError('no sequences')
        member=names[0]
        with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
            rows=[]
            for key in ('body.npy','object.npy','contact.npy'):
                raw=inner.read(key); f=io.BytesIO(raw)
                version=np.lib.format.read_magic(f)
                if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
                elif version==(2,0): shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
                else: raise ValueError(f'unsupported NPY version {version}')
                if not dtype.hasobject: raise ValueError(f'{key} is not an object array')
                ops=set(); globals_=[]
                for op,arg,pos in pickletools.genops(raw[f.tell():]):
                    ops.add(op.name)
                    if op.name=='GLOBAL': globals_.append(arg)
                rows.append({'name':key,'npy_version':list(version),'shape':list(shape),'dtype':str(dtype),
                             'pickle_globals':sorted(set(globals_)),'pickle_opcodes':sorted(ops)})
    print(json.dumps({'archive_sha256':h.hexdigest(),'member':member,'fields':rows,
                      'warning':'Static pickle opcode inspection only; no object deserialization or motion validation.'},allow_nan=False))


if __name__=='__main__': main()
