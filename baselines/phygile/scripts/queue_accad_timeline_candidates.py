"""Wait only for the targeted ACCAD reference, then audit indexed fit timelines.

No motion arrays are deleted or written to the server, and no candidate is
silently promoted to an approved motion/text training pair.
"""
import argparse
import datetime
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
import numpy as np
from summarize_amass_acquisition import alive


def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    status=a.output/'status.json'
    job=a.base/'results/humanml_g1_pipeline_20261021/accad_reference_download'
    receipt=job/'ACCAD_reference_smplh.json';pid=json.loads((job/'launch.json').read_text())['pid']
    while not receipt.exists():
        save(status,{'status':'WAITING_FOR_TARGETED_REFERENCE','download_pid':pid,
             'updated_at':datetime.datetime.now().isoformat(),'full_amass_download':False})
        if not alive(pid):
            save(status,{'status':'REFERENCE_DOWNLOAD_NEEDS_ATTENTION','download_pid':pid,
                         'training_pairs_created':0,'no_data_deleted':True});return
        time.sleep(30)
    record=json.loads(receipt.read_text());archive=Path(record['local_archive'])
    import hashlib
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):h.update(chunk)
    if h.hexdigest()!=record['sha256']:raise ValueError('Reference archive SHA256 mismatch')
    import csv
    rows=[x for x in csv.DictReader((a.base/'results/humanml_g1_pipeline_20261021/indexed_inventory/indexed_asset_coverage.csv').open()) if x['amass_download_name']=='ACCAD']
    norm=lambda s:re.sub('[^a-z0-9/]','',s.lower())
    lookup={norm(x['expected_original_smplh_file']):x for x in rows}
    if len(lookup)!=len(rows):raise ValueError('Ambiguous normalized reference names')
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-i',str(a.identity),a.host,
        'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 '+a.remote_python+' '+a.remote_worker]
    results=[];seen=set()
    with (a.output/'worker.stderr.txt').open('w') as log:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8')
        try:
            with tarfile.open(archive,'r|bz2') as tar:
                for member in tar:
                    if not member.isfile() or norm(member.name) not in lookup:continue
                    row=lookup[norm(member.name)];key=row['humanml_source_path']
                    if key in seen:raise ValueError('Duplicate reference recording')
                    seen.add(key)
                    payload=tar.extractfile(member).read()
                    with np.load(io.BytesIO(payload),allow_pickle=False) as d:
                        n=len(d['trans']);idx=np.unique(np.linspace(0,n-1,min(n,256),dtype=int))
                        request={'source_group':key,'smplx_file':row['server_smplx_path'],
                            'reference_frames':n,'reference_fps':float(d['mocap_framerate']),
                            'sample_indices':idx.tolist(),'reference_trans':d['trans'][idx].tolist(),
                            'reference_poses66':d['poses'][idx,:66].tolist()}
                        has_markers='marker_data' in d.files
                    child.stdin.write(json.dumps(request)+'\n');child.stdin.flush()
                    line=child.stdout.readline()
                    if not line:raise RuntimeError('Reference diagnostic worker ended unexpectedly')
                    result=json.loads(line);result.update({'reference_member':member.name,
                        'reference_file_sha256':hashlib.sha256(payload).hexdigest(),
                        'reference_contains_measured_markers':has_markers})
                    results.append(result)
                    save(a.output/'candidates.json',results)
                    save(status,{'status':'RUNNING_CANDIDATE_ALIGNMENT','processed_sources':len(results),
                         'required_indexed_sources':len(rows),'training_pairs_created':0,
                         'updated_at':datetime.datetime.now().isoformat(),'no_data_deleted':True})
                    print(json.dumps({'processed_sources':len(results),'last_status':result['status']}),flush=True)
        finally:
            child.stdin.close();child.stdout.close();child.wait(timeout=120)
    save(status,{'status':'CANDIDATE_DIAGNOSTICS_COMPLETED','processed_sources':len(results),
        'required_indexed_sources':len(rows),'unmatched_sources':len(rows)-len(seen),
        'training_pairs_created':0,'approved_for_training':False,'no_data_deleted':True,
        'reference_archive_sha256':record['sha256'],'data_written_on_server':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['base','output','identity']:p.add_argument('--'+key,type=Path,required=True)
    for key in ['host','remote-python','remote-worker']:p.add_argument('--'+key,required=True)
    main(p.parse_args())
