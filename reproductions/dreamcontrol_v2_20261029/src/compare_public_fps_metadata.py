"""Compare a third-party AMASS CSV with local evidence, without accepting timing."""
import csv
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
CSV=PROJECT/'data/derived/realdream_fps_metadata_20261101/fps.csv'
REPORTS=PROJECT/'reports'
URL='https://huggingface.co/datasets/realdream-ai/AMASS/resolve/d38490c0e60dcefae81461fc5f4765fe66727831/fps.csv'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    expected={
        '000008':'AMASS_amass_202203_SSM_synced_20161014_50033_ATUSquat_sync_poses.npz',
        '011120':'AMASS_amass_202203_SSM_synced_20160330_03333_ATU_001_poses.npz',
        'ACCAD_004347':'AMASS_amass_202203_ACCAD_Male1Walking_c3d_Male1 Cal_poses.npz',
        'MOSH_000798_crop':'AMASS_amass_202203_MPI_mosh_00008_misc_poses.npz'}
    with CSV.open(encoding='utf8',newline='') as f:
        rows=list(csv.DictReader(f,skipinitialspace=True))
    matches={}
    for sid,name in expected.items():
        found=[r for r in rows if r['fname_new'].strip()==name]
        if len(found)!=1:raise ValueError(f'{sid} expected exactly one CSV row; got {len(found)}')
        row=found[0]
        matches[sid]={'csv_name':name,'source_frames_declared':int(row['len_ori']),
                      'source_fps_declared':float(row['fps']),'new_frames_declared':int(row['len_new'])}
    refpath=REPORTS/'ssm_source_preflight_20261101.json'
    ref={x['sample_id']:x for x in json.loads(refpath.read_text())['cases']}
    for sid in ('000008','011120'):
        r=ref[sid];m=matches[sid]
        m.update({'local_reference_fps':r['reference_header_rate_hz'],'local_smplx_fps':r['source_header_rate_hz'],
                  'local_matched_marker_frames':r['frames_compared'],
                  'reference_rate_equal':m['source_fps_declared']==r['reference_header_rate_hz'],
                  'frame_count_equal':m['source_frames_declared']==r['frames_compared']})
    for sid,filename in [('ACCAD_004347','accad_004347_fullrate_human_20261031.json'),
                         ('MOSH_000798_crop','mosh_000798_fullrate_human_20261031.json')]:
        path=REPORTS/filename;local=json.loads(path.read_text())
        matches[sid].update({'local_candidate_rate':local['fps_declared'],
                             'local_candidate_excerpt_frames':local['frames'],
                             'local_candidate_source_sha256':local['source_sha256'],
                             'local_human_report_sha256':sha(path),
                             'rate_equal':matches[sid]['source_fps_declared']==local['fps_declared']})
    output={'csv_url_fixed_commit':URL,'csv_sha256':sha(CSV),'csv_rows':len(rows),
            'local_ssm_report_sha256':sha(refpath),'cases':matches,'training_ready':False,
            'limitations':['Third-party metadata with no dataset card or provenance for the CSV; not an independent clock measurement.',
                           'Similar paths or frame counts do not prove the same source bytes or HumanML intermediate.',
                           'Exact FPS agreement with local reference headers could arise from copying the same metadata.',
                           'Different ACCAD filename spacing and path mapping remain unresolved.',
                           'No motion, caption, ground, license, or training acceptance follows from this comparison.']}
    with (REPORTS/'public_amass_fps_metadata_20261101.json').open('x',encoding='utf8') as f:
        json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(output,ensure_ascii=False))

if __name__=='__main__':main()
