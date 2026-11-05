"""Fetch two bounded public DART fps dictionaries for research comparison.

Independent project metadata is not original capture-time ground truth and
does not authorize changing any source header or training data.
"""
import hashlib,json,shutil
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data/dart_fps_metadata_20261102'
FILES={'fps_dict.json':'1L-0nhlEuyBXwszEu5oTRkzvtJ6ExlaEy','fps_dict_all.json':'1L4p9Le8J1uw53J226OD_QELXIRZshU0T'}
def main():
    if shutil.disk_usage(ROOT).free<512*1024**2:raise RuntimeError('local disk budget')
    OUT.mkdir(exist_ok=False);found={}
    for name,id in FILES.items():
      url='https://drive.usercontent.google.com/download'
      with requests.get(url,params={'id':id,'export':'download','confirm':'t'},stream=True,timeout=(20,30)) as r:
        r.raise_for_status();blocks=[];total=0
        for b in r.iter_content(65536):
            total+=len(b)
            if total>20*1024*1024:raise ValueError('metadata file exceeds limit')
            blocks.append(b)
        payload=b''.join(blocks);obj=json.loads(payload)
        if not isinstance(obj,dict) or not obj:raise ValueError('expected nonempty fps dictionary')
        with (OUT/name).open('xb') as f:f.write(payload)
        found[name]={'google_drive_file_id':id,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),
                     'entries':len(obj),'content_type':r.headers.get('Content-Type')}
    result={'source_readme':'https://github.com/zkf1997/DART/blob/bb67ae1ed6ce051080468bf15bc6e54a6c3f8417/README.md',
        'source_processing':'https://github.com/zkf1997/DART/blob/bb67ae1ed6ce051080468bf15bc6e54a6c3f8417/data_scripts/extract_dataset.py',
        'linked_public_folder_id':'1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps','files':found,'training_ready':False,
        'limitations':['DART metadata, not AMASS original per-file clock or source identity.',
          'Public folder does not independently establish dataset redistribution permissions.',
          'The fps dictionary derivation, exact correspondence and physical time still require examination.']}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
