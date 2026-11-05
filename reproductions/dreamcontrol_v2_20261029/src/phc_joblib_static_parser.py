"""Bounded symbolic parser for a narrow joblib pickle format; executes no globals.

Supports only plain dictionaries, primitive scalars/tuples, defaultdict(dict),
and joblib's numeric ndarray wrapper with explicit padding. Every other opcode,
class, callable, dtype or oversized payload is rejected. This is not a general
pickle or joblib reader, and a parsed issue list is not training acceptance.
"""
import hashlib
import io
import json
import math
import pickletools
import struct
from pathlib import Path

MARK = object()
GLOBALS = {'collections defaultdict','builtins dict','joblib.numpy_pickle NumpyArrayWrapper',
           'numpy ndarray','numpy dtype'}
DTYPES = {'i8':'q','i4':'i','i2':'h','i1':'b','u8':'Q','u4':'I','u2':'H','u1':'B',
          'f8':'d','f4':'f','b1':'?'}


class DType:
    def __init__(self, name):
        if name not in DTYPES: raise ValueError('unsupported dtype: '+repr(name))
        self.name = name
        self.order = '<'


class Wrapper:
    def __init__(self): self.state = None


def parse(raw):
    if not isinstance(raw,bytes) or len(raw) > 4_000_000:
        raise ValueError('unsupported source size')
    f = io.BytesIO(raw); stack = []; memo = {}; wrappers = 0; operations = 0
    def popmark():
        try: index = len(stack) - 1 - stack[::-1].index(MARK)
        except ValueError: raise ValueError('missing mark')
        values = stack[index+1:]; del stack[index:]; return values
    for op,arg,pos in pickletools.genops(f):
        operations += 1
        if operations > 2_000_000: raise ValueError('too many opcodes')
        name = op.name
        if name in ('PROTO','FRAME'): continue
        if name == 'STOP':
            if len(stack) != 1 or f.tell() != len(raw): raise ValueError('invalid end state')
            return stack[0], {'opcodes':operations,'numeric_arrays':wrappers}
        if name == 'MARK': stack.append(MARK)
        elif name in ('EMPTY_DICT',): stack.append({})
        elif name in ('EMPTY_LIST',): stack.append([])
        elif name == 'EMPTY_TUPLE': stack.append(())
        elif name == 'NONE': stack.append(None)
        elif name == 'NEWTRUE': stack.append(True)
        elif name == 'NEWFALSE': stack.append(False)
        elif name in ('BINUNICODE','SHORT_BINUNICODE','UNICODE','BININT','BININT1','BININT2','LONG1','LONG4','BINFLOAT'):
            stack.append(arg)
        elif name == 'GLOBAL':
            if arg not in GLOBALS: raise ValueError('unsupported global: '+str(arg))
            stack.append(('GLOBAL',arg))
        elif name in ('BINPUT','LONG_BINPUT','PUT'):
            if not stack: raise ValueError('empty memo value')
            memo[int(arg)] = stack[-1]
        elif name in ('BINGET','LONG_BINGET','GET'):
            if int(arg) not in memo: raise ValueError('unknown memo')
            stack.append(memo[int(arg)])
        elif name == 'MEMOIZE': memo[len(memo)] = stack[-1]
        elif name == 'TUPLE': stack.append(tuple(popmark()))
        elif name in ('TUPLE1','TUPLE2','TUPLE3'):
            n = int(name[-1]); values = stack[-n:]; del stack[-n:]; stack.append(tuple(values))
        elif name == 'SETITEM':
            value = stack.pop(); key = stack.pop(); target = stack[-1]
            if not isinstance(target,dict): raise ValueError('SETITEM target')
            target[key] = value
        elif name == 'SETITEMS':
            values = popmark(); target = stack[-1]
            if not isinstance(target,dict) or len(values)%2: raise ValueError('SETITEMS target')
            for i in range(0,len(values),2): target[values[i]] = values[i+1]
        elif name == 'APPEND':
            item = stack.pop(); target = stack[-1]
            if not isinstance(target,list): raise ValueError('APPEND target')
            target.append(item)
        elif name == 'APPENDS':
            items = popmark(); target = stack[-1]
            if not isinstance(target,list): raise ValueError('APPENDS target')
            target.extend(items)
        elif name == 'REDUCE':
            args = stack.pop(); cls = stack.pop()
            if not isinstance(args,tuple): raise ValueError('REDUCE args')
            if cls == ('GLOBAL','collections defaultdict') and args == (('GLOBAL','builtins dict'),):
                stack.append({})
            elif cls == ('GLOBAL','numpy dtype') and args and isinstance(args[0],str):
                stack.append(DType(args[0]))
            else: raise ValueError('unsupported REDUCE')
        elif name == 'NEWOBJ':
            args = stack.pop(); cls = stack.pop()
            if cls != ('GLOBAL','joblib.numpy_pickle NumpyArrayWrapper') or args != ():
                raise ValueError('unsupported NEWOBJ')
            stack.append(Wrapper())
        elif name == 'BUILD':
            state = stack.pop(); obj = stack[-1]
            if isinstance(obj,DType):
                if not isinstance(state,tuple) or len(state)<2 or state[1] not in ('<','>','|','='):
                    raise ValueError('unsupported dtype state')
                obj.order = '<' if state[1] in ('<','|','=') else '>'
            elif isinstance(obj,Wrapper):
                if not isinstance(state,dict): raise ValueError('invalid wrapper state')
                shape = state.get('shape'); dtype = state.get('dtype')
                if state.get('subclass') != ('GLOBAL','numpy ndarray') or not isinstance(dtype,DType):
                    raise ValueError('unsupported ndarray wrapper')
                if not isinstance(shape,tuple) or len(shape)>3 or any(not isinstance(n,int) or n<0 for n in shape):
                    raise ValueError('invalid ndarray shape')
                count = math.prod(shape); fmt = dtype.order + DTYPES[dtype.name]
                width = struct.calcsize(fmt); size = count * width
                if count > 500_000 or size > 4_000_000 or state.get('order') not in ('C','F'):
                    raise ValueError('array exceeds bounds')
                alignment = state.get('numpy_array_alignment_bytes')
                if alignment != 16: raise ValueError('unsupported joblib alignment')
                padding = f.read(1)
                if len(padding)!=1 or padding[0]>16: raise ValueError('invalid padding length')
                if f.read(padding[0]) != b'\xff'*padding[0]: raise ValueError('invalid padding')
                payload = f.read(size)
                if len(payload)!=size: raise ValueError('truncated array')
                values = [v[0] for v in struct.iter_unpack(fmt,payload)]
                stack[-1] = {'_numeric_array':values,'shape':shape,'dtype':dtype.name}
                wrappers += 1
            else: raise ValueError('unsupported BUILD')
        else: raise ValueError('unsupported opcode '+name+' at '+str(pos))
    raise ValueError('missing STOP')


def main():
    here = Path(__file__).resolve().parent
    raw = (here/'source.pkl').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != '2c336550d452e5769bc12d199c285bb3717af651f3691ee18ac6b82f2de0a3d8':
        raise ValueError('source changed')
    index, stats = parse(raw)
    if not isinstance(index,dict) or any(not isinstance(k,str) or not isinstance(v,dict) for k,v in index.items()):
        raise ValueError('unexpected index structure')
    byte_report = json.loads((here/'byte_scan.json').read_text())
    candidate_keys = byte_report['candidate_key_counts']
    sample = {name:{'key':item['literal'],'present':item['literal'] in index,
                    'value':index.get(item['literal'])} for name,item in candidate_keys.items()}
    issues = {}
    for value in index.values():
        issue = value.get('issue')
        if not isinstance(issue,str): raise ValueError('unexpected issue type')
        issues[issue] = issues.get(issue,0) + 1
    result = {'source_sha256':digest, 'entry_count':len(index), 'issue_counts':issues,
              'parser_stats':stats, 'candidate_keys':sample, 'training_ready':False,
              'limitations':['symbolic non-executing bounded parse, not independently verified complete',
                  'PHC downloadable list and key convention may differ from v2 authors',
                  'absence does not prove physical feasibility, source identity or permission',
                  'PHC conversion classifies SSM_synced as test dataset']}
    with (here/'parsed_summary.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':main()
