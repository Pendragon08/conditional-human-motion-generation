"""Read bounded SSM reference marker names for potential ground provenance.

Matching names alone cannot establish ground calibration or actual coordinates.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    config=json.loads(Path(sys.argv[1]).read_text())
    evidence=json.loads(Path(config['ssm_source_evidence_manifest']).read_text())
    index=Path(config['ssm_alignment_path'])
    if sha(index)!=evidence['alignment_sha256']:raise ValueError('alignment hash mismatch')
    by_id={x['sample_id']:x for x in evidence['receipts']}
    alignment={row['motion_id']:row for line in index.read_text().splitlines()
               if (row:=json.loads(line)).get('mirrored') is False and row.get('motion_id') in by_id}
    if set(by_id)!=set(alignment):raise ValueError('missing alignment')
    out=[]
    for sid in sorted(by_id):
        receipt=Path(alignment[sid]['source_receipt'])
        if sha(receipt)!=by_id[sid]['receipt_sha256']:raise ValueError('receipt hash mismatch')
        meta=json.loads(receipt.read_text());ref=Path(meta['reference_file'])
        if sha(ref)!=meta['reference_sha256']:raise ValueError('reference hash mismatch')
        with np.load(ref,allow_pickle=False) as f:
            raw=np.asarray(f['marker_labels'])
        if raw.ndim!=1 or len(raw)>10000 or raw.dtype.kind not in 'US':raise ValueError('unexpected labels')
        names=[v.decode('utf8') if isinstance(v,bytes) else str(v) for v in raw]
        matches=[n for n in names if re.search(r'ground|floor|\bfloor\b|calib|origin',n,re.I)]
        out.append({'sample_id':sid,'reference_sha256':sha(ref),'marker_names':names,
                    'potential_ground_names':matches})
    print(json.dumps({'cases':out,'training_ready':False,
        'limitations':['Only exact label text inspected; absent matching names do not prove ground markers unavailable.',
                       'Names alone do not prove point type, floor contact, calibration or coordinates.']},indent=2))
if __name__=='__main__':main()
