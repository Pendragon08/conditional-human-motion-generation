"""Static pickle opcode inventory; does not deserialize or execute globals."""
import collections
import hashlib
import json
import pickletools
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source.pkl'


def main():
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != '2c336550d452e5769bc12d199c285bb3717af651f3691ee18ac6b82f2de0a3d8':
        raise ValueError('downloaded file changed')
    counts = collections.Counter()
    globals_ = collections.Counter()
    markers = []
    for op, arg, pos in pickletools.genops(raw):
        counts[op.name] += 1
        if op.name == 'GLOBAL': globals_[arg] += 1
        if isinstance(arg,str) and any(s in arg.lower() for s in ('atusquat','running_on_the_spot','jumping_jacks','punch_kick')):
            markers.append({'offset':pos,'value':arg})
    result = {'source_sha256':digest,'bytes':len(raw),'opcode_counts':dict(sorted(counts.items())),
              'globals':dict(globals_),'candidate_text_matches':markers,
              'warning':'Static opcode scan only; does not deserialize, map issue records, certify source authenticity or establish filtering eligibility.'}
    with (HERE/'static_inventory.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':main()
