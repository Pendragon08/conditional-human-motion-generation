"""Restricted GRAB NPY-object reader for locally obtained dataset archives.

Only a small allowlist of NumPy globals and a restricted opcode set is
accepted; arrays must contain primitive numeric data and nested dictionaries
must use string keys. This reduces, but does not eliminate, the risk of parsing
hostile pickle data; use only on the user's verified GRAB archives.
"""
from __future__ import annotations
import io
import pickle
import pickletools
import zipfile
import numpy as np


_GLOBALS={('numpy','dtype'):np.dtype,('numpy','ndarray'):np.ndarray,
          ('numpy.core.multiarray','_reconstruct'):np.core.multiarray._reconstruct}
_OPS={'APPEND','APPENDS','BINBYTES','BINFLOAT','BINGET','BININT','BININT1','BININT2','BINPUT',
      'BINUNICODE','BUILD','EMPTY_DICT','EMPTY_LIST','EMPTY_TUPLE','GLOBAL','LONG1','LONG_BINPUT',
      'LONG_BINGET','MARK','NEWFALSE','NEWTRUE','NONE','PROTO','REDUCE','SETITEM','SETITEMS',
      'SHORT_BINBYTES','STOP','TUPLE','TUPLE1','TUPLE2','TUPLE3'}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        try: return _GLOBALS[(module,name)]
        except KeyError as exc: raise pickle.UnpicklingError(f'forbidden global {module}.{name}') from exc
    def persistent_load(self,pid): raise pickle.UnpicklingError('persistent references are forbidden')


def _validate(value,depth=0):
    if depth>20: raise ValueError('nesting is too deep')
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in 'biuf': raise ValueError('nested arrays must be real numeric or boolean')
        if value.dtype.kind in 'f' and not np.isfinite(value).all(): raise ValueError('non-finite array')
        return
    if isinstance(value,dict):
        if any(not isinstance(k,str) for k in value): raise ValueError('dictionary keys must be strings')
        for item in value.values(): _validate(item,depth+1)
        return
    if isinstance(value,(list,tuple)):
        for item in value: _validate(item,depth+1)
        return
    if isinstance(value,(str,bytes,bool,int,float,np.integer,np.floating)) or value is None:
        if isinstance(value,(float,np.floating)) and not np.isfinite(value): raise ValueError('non-finite scalar')
        return
    raise ValueError(f'unsupported object type: {type(value).__name__}')


def read_object_npy(raw: bytes):
    if not isinstance(raw,bytes) or len(raw)>192*1024**2: raise ValueError('invalid or oversized NPY')
    f=io.BytesIO(raw); version=np.lib.format.read_magic(f)
    if version==(1,0): shape,_,dtype=np.lib.format.read_array_header_1_0(f)
    elif version==(2,0): shape,_,dtype=np.lib.format.read_array_header_2_0(f)
    else: raise ValueError('unsupported NPY version')
    if shape!=() or not dtype.hasobject: raise ValueError('expected scalar object array')
    payload=raw[f.tell():]
    for op,arg,_ in pickletools.genops(payload):
        if op.name not in _OPS: raise pickle.UnpicklingError(f'forbidden pickle opcode: {op.name}')
        if op.name=='GLOBAL' and tuple(arg.split(' ',1)) not in _GLOBALS:
            raise pickle.UnpicklingError(f'forbidden pickle global: {arg}')
    result=RestrictedUnpickler(io.BytesIO(payload)).load()
    if not isinstance(result,np.ndarray) or result.shape!=() or result.dtype!=np.dtype('O'):
        raise ValueError('unexpected object-array container')
    value=result.item()
    if not isinstance(value,dict): raise ValueError('expected dictionary')
    _validate(value)
    return value


def read_grab_field(archive: zipfile.ZipFile, member: str, field: str):
    if field not in ('body','object','contact'): raise ValueError('unsupported GRAB field')
    info=archive.getinfo(member)
    if info.file_size>192*1024**2: raise ValueError('oversized motion member')
    with zipfile.ZipFile(io.BytesIO(archive.read(info))) as nested:
        info=nested.getinfo(field+'.npy')
        if info.file_size>192*1024**2: raise ValueError('oversized field')
        return read_object_npy(nested.read(info))
