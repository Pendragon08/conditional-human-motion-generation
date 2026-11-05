"""Trace two already marker-verified SSM clips through official stride/slicing."""
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PUBLIC=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public')
OUT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    alignment=ROOT/'ssm_id_alignment_20261030/ids.jsonl'
    if sha(alignment)!='575c9c008298dbf30d21317f258346a3dd11cc7efc6858d242a26c11c6333b46':raise ValueError('alignment changed')
    if sha(PUBLIC/'index.csv')!='80226910d7259655ab5f411fd807b7651037341b4806fde025946a74d62b1065':raise ValueError('index changed')
    if sha(OUT/'report.json')!='d1cb58eb2c5d2614b103354c5e75d35566d736d011d5b55003d63d0de8990c5d':raise ValueError('official semantics report changed')
    rows={r['motion_id']:r for r in map(json.loads,alignment.read_text().splitlines())}
    with (PUBLIC/'index.csv').open(newline='',encoding='utf8') as f:
        index={r['new_name'].removesuffix('.npy'):r for r in csv.DictReader(f)}
    result=[]
    for name in ('000008','002511'):
        row=rows[name]; src=index[name];receipt_path=Path(row['source_receipt']);receipt=json.loads(receipt_path.read_text())
        reference=Path(receipt['reference_file']); target=Path(receipt['smplx_file'])
        if sha(reference)!=receipt['reference_sha256'] or sha(target)!=receipt['smplx_sha256']:
            raise ValueError('source hash mismatch')
        if receipt['timeline']['status']!='FRAME_IDENTITY_VERIFIED_BY_MEASURED_MARKERS':raise ValueError('missing measured marker witness')
        if row['source_path']!=src['source_path'] or row['source_path']!=receipt['source_group'] or row['mirrored']:
            raise ValueError('wrong source')
        with np.load(reference,allow_pickle=False) as f:
            fps=float(f['mocap_framerate']); frames=int(f['trans'].shape[0])
        if fps!=row['reference_fps'] or frames!=row['reference_raw_frame_count'] or fps!=receipt['timeline']['reference_fps']:
            raise ValueError('reference metadata mismatch')
        stride=int(fps/20); start=int(src['start_frame']); stop=int(src['end_frame'])
        if start!=row['start'] or stop!=row['stop'] or stride!=row['recorded_stride']:
            raise ValueError('index or stride mismatch')
        sampled=list(range(0,frames,stride)); chosen=sampled[start:stop]
        if len(chosen)!=stop-start or chosen[-1]>=frames:raise ValueError('truncated slice')
        result.append({'id':name,'source_path':src['source_path'],'reference_sha256':sha(reference),'smplx_sha256':sha(target),
                       'receipt_sha256':sha(receipt_path),'reference_fps':fps,'reference_frames':frames,
                       'reference_stride':stride,'sampled_fps':fps/stride,'slice':[start,stop],
                       'first_raw_frame':chosen[0],'last_raw_frame':chosen[-1],'samples':len(chosen),
                       'first_to_last_seconds':(chosen[-1]-chosen[0])/fps,
                       'nominal_20hz_first_to_last_seconds':(len(chosen)-1)/20,
                       'reference_time_origin_s':chosen[0]/fps})
    out={'script_sha256':sha(Path(__file__)),'alignment_sha256':sha(alignment),'witnesses':result,
         'limitations':'Uses prior recorded measured-marker identity; marker arrays are not recomputed here. Official stride and slice interpretation only; no interpolation, mirror, model equivalence, contact, dynamics or training acceptance. SSM has no cross-official-split source group in the earlier audit.'}
    with (OUT/'ssm_time_witness.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,sort_keys=True)
    print(json.dumps(out,sort_keys=True),flush=True)

if __name__=='__main__':main()
