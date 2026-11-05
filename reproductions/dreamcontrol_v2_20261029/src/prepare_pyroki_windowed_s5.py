"""Adapt the previously audited S1 window code to the S5 source path only."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    previous=ROOT/'pyroki_grab_s1_pour_windowed_20261030';manifest=json.loads((previous/'manifest.json').read_text())
    source=ROOT/'grab_s5_cup_human_source_20261030/report.json';doc=json.loads(source.read_text())
    if sha(source)!='9f16fb32e02d7d6f07d73bee3bd0b3fbd7583ebb3fd2d21c679c489a8e758a33' or sha(source.parent/'human_source.npz')!=doc['artifact_sha256']:
        raise ValueError('S5 human source changed')
    old="ROOT/'grab_s1_pour_human_source_20261030/human_source.npz'";new="ROOT/'grab_s5_cup_human_source_20261030/human_source.npz'"
    outputs={}
    for filename,key in (('pyroki_grab_windowed_pilot.py','s1_driver_sha256'),('audit_pyroki_windowed_pilot.py','s1_audit_sha256')):
        src=previous/filename
        if sha(src)!=manifest[key]:raise ValueError('previous code changed')
        text=src.read_text()
        if text.count(old)!=1:raise ValueError('source placeholder mismatch')
        dest=HERE/filename
        with dest.open('x',encoding='utf8',newline='\n') as f:f.write(text.replace(old,new))
        outputs[filename]=sha(dest)
    result={'source_report_sha256':sha(source),'source_artifact_sha256':doc['artifact_sha256'],
            'base_manifest_sha256':sha(previous/'manifest.json'),'adapted_files':outputs,'script_sha256':sha(__file__),
            'limitations':'Same S1 legacy PyRoki and experimental weights, only source path changes. Assumed ground and low-speed proxy, not author v2 code, real contact or train-ready data.'}
    with (HERE/'manifest.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
