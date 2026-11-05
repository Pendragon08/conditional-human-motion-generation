"""Compare BABEL files against their currently stored ZIP, with CRC and hashes."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024/data_preparation_20261025/babel_v1_0')
OUT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/textop_babel_audit_20261030')

def hash_stream(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    archive=ROOT/'babel_v1-0_release.zip'
    with archive.open('rb') as f: archive_hash=hash_stream(f)
    rows=[]
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        for n in ('train','val','test','extra_train','extra_val'):
            member='babel_v1.0_release/'+n+'.json'
            with z.open(member) as f: expected=hash_stream(f)
            file=ROOT/(n+'.json')
            with file.open('rb') as f: observed=hash_stream(f)
            obj=json.loads(file.read_text(encoding='utf8'))
            rows.append({'name':n,'sha256':observed,'match_zip':observed==expected,
                         'records':len(obj),'type':type(obj).__name__})
    OUT.mkdir(mode=0o700)
    summary={'archive_sha256':archive_hash,'zip_crc_bad':bad,'json_files':rows,
             'limitations':['ZIP-to-file consistency, not independent official provenance or license verification.',
                            'No BABEL-to-AMASS temporal alignment validation or training acceptance.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
