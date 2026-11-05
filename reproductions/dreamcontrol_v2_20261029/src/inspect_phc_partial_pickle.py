"""Inspect a bounded pickle prefix without loading data or executing globals."""
import hashlib
import json
import pickletools
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    raw = (HERE/'source.pkl').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != '2c336550d452e5769bc12d199c285bb3717af651f3691ee18ac6b82f2de0a3d8':
        raise ValueError('source changed')
    recent = deque(maxlen=35); names = set(); error = None
    try:
        for op,arg,pos in pickletools.genops(raw):
            if op.name == 'GLOBAL': names.add(arg)
            shown = repr(arg)
            recent.append({'offset':pos,'opcode':op.name,'argument':shown[:200]})
    except ValueError as exc:
        error = str(exc)
    result = {'source_sha256':digest,'prefix_last_operations':list(recent),
              'prefix_globals':sorted(names),'stopped':error,
              'next_64_bytes_hex':raw[1235:1299].hex(),
              'limitations':'Static inspection stops at first unsupported joblib array payload; no unpickling or eligibility conclusion.'}
    with (HERE/'partial_pickle_inspection.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':main()
