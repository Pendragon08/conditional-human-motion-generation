"""Independently recount all matched marker coordinates and the closed preflight."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def labels(data):return [x.decode('utf8') if isinstance(x,bytes) else str(x) for x in data]


def main():
    pre=json.loads(next(x for x in (HERE/'preflight.log').read_text().splitlines() if x.startswith('{')))
    source=pre['ssm_source_evidence_audit'];assert source['marker_evidence_cases']==2 and source['training_ready'] is False
    assert set(source['samples_without_marker_evidence'])=={'ACCAD_004347','MOSH_000798_crop'}
    assert pre['caption_evidence_audit']['verified_semantic_receipts']==4 and pre['state_audit']['state_technical_checks_passed']
    cfg=json.loads((HERE/'config.json').read_text());align=Path(cfg['ssm_alignment_path']);bundle=Path(cfg['ssm_source_evidence_manifest'])
    assert sha(align)==source['alignment_sha256'] and sha(bundle)==source['evidence_manifest_sha256']
    rows={x['motion_id']:x for x in map(json.loads,align.read_text().splitlines()) if x['mirrored'] is False}
    cases=[]
    for case in source['cases']:
        entry=rows[case['sample_id']];receipt_path=Path(entry['source_receipt']);assert sha(receipt_path)==case['receipt_sha256']
        receipt=json.loads(receipt_path.read_text());a_path=Path(receipt['smplx_file']);b_path=Path(receipt['reference_file'])
        assert sha(a_path)==case['source_sha256'] and sha(b_path)==case['reference_sha256']
        with np.load(a_path,allow_pickle=False) as f:a=f['markers'].copy();la=labels(f['labels']);arate=float(f['mocap_frame_rate'])
        with np.load(b_path,allow_pickle=False) as f:b=f['marker_data'].copy();lb=labels(f['marker_labels']);brate=float(f['mocap_framerate'])
        common=sorted(set(la)&set(lb));assert len(common)==case['common_marker_labels'] and len(a)==len(b)==case['frames_compared']
        count=0;minimum=len(common);max_diff=0.
        for i in range(len(a)):
            frame_count=0
            for name in common:
                x=a[i,la.index(name)];y=b[i,lb.index(name)]
                vx=all(math.isfinite(float(v)) for v in x) and any(float(v)!=0 for v in x)
                vy=all(math.isfinite(float(v)) for v in y) and any(float(v)!=0 for v in y)
                assert vx==vy
                if not vx:continue
                frame_count+=1
                for j in range(3):max_diff=max(max_diff,abs(float(x[j])-float(y[j])));count+=1
            minimum=min(minimum,frame_count)
        assert count==case['valid_coordinate_count'] and minimum==case['minimum_valid_markers_per_frame']
        assert max_diff==case['marker_max_abs_difference_m']==0 and arate==case['source_header_rate_hz'] and brate==case['reference_header_rate_hz']
        cases.append({'sample_id':case['sample_id'],'frames':len(a),'coordinates_checked':count,'max_difference_m':max_diff})
    assert pre['feature_audit']['training_ready'] is False
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    assert 'Ran 9 tests' in (HERE/'tests.log').read_text() and 'OK' in (HERE/'tests.log').read_text()
    assert not list(HERE.glob('*.pt'))
    out={'preflight_log_sha256':sha(HERE/'preflight.log'),'source_evidence_manifest_sha256':sha(bundle),
         'cases':cases,'training_ready':False,'entry_rejected_training':True,'script_sha256':sha(Path(__file__)),
         'limitations':'Same numeric source/reference markers; no official HumanML intermediate, independent clock or physical acceptance.'}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
