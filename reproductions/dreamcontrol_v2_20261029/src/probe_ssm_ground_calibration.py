"""Check whether the available labelled 011120 body markers can calibrate ground."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ground_calibration import fit_ground_surface

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMAN=ROOT/'ssm_rawfps_human_batch_20261031/011120'
ALIGN=ROOT/'ssm_id_alignment_20261030/ids.jsonl'
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    row=next(r for r in map(json.loads,ALIGN.read_text().splitlines()) if r['motion_id']=='011120' and not r['mirrored'])
    receipt=Path(row['source_receipt']); prior=json.loads(receipt.read_text()); human=json.loads((HUMAN/'human_report.json').read_text())
    source=Path(prior['smplx_file'])
    assert sha(receipt)==human['receipt_sha256'] and sha(source)==prior['smplx_sha256']==human['source_sha256']
    with np.load(source,allow_pickle=False) as d:
        markers=d['markers'].copy(); labels=[str(x) for x in d['labels']]
    names=['LHEE','RHEE','LTOE','RTOE','LMT1','RMT1','LMT5','RMT5']
    assert len(labels)==len(set(labels)) and all(labels.count(name)==1 for name in names)
    points=markers[:,[labels.index(name) for name in names],:]
    rejected=None
    try:
        fit_ground_surface(points,names,observation_kind='body_surface_markers',source_to_robot=np.eye(4),source_up=[0,0,1],
                           stationary_tolerance_m=.001,plane_tolerance_m=.001,min_normal_alignment=.99)
    except ValueError as exc:rejected=str(exc)
    if rejected is None or 'body markers are insufficient' not in rejected:raise AssertionError('body marker calibration was not rejected')
    result={'sample_id':'011120','source_sha256':sha(source),'source_receipt_sha256':sha(receipt),
            'source_marker_shape':list(markers.shape),'foot_body_labels':names,'calibration_status':'unavailable_body_markers_only',
            'rejection_reason':rejected,'training_ready':False,'script_sha256':sha(Path(__file__)),
            'limitations':['Check only identifies the eight listed body markers; it does not inventory other files or labels.',
                'The module does not establish provenance or calibration even if supplied with differently declared points.',
                'No physical quality or real ground has been accepted.']}
    with (HERE/'probe.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
