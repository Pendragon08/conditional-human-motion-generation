"""Read a bounded subset from a third-party uncompressed HumanML3D mirror.

The user authorized noncommercial research use. The mirror is not an official
release. Individual files are diagnostics only, never training acceptance.
"""
import argparse, hashlib, io, json, shutil, tarfile
from pathlib import Path
import numpy as np
import requests

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
URL=('https://huggingface.co/datasets/zyyy12138/motion-smd-data/resolve/'
     '113b64cb43c0e0c25ec09f93542ec92f575ad148/humanml3d/new_joints.tar')
SIZE=1111336960
OFFICIAL_EXAMPLE='739df5e2e025d21ae5ce01ea7e0a8d3038448a46e88865d95a638e6242ee8b6b'
WANTED={'004271.npy','007555.npy','012314.npy'}

class Counted:
    def __init__(self,raw):self.raw=raw;self.total=0;self.last=0
    def read(self,n=-1):
        value=self.raw.read(n);self.total+=len(value)
        if self.total-self.last>=64*1024**2:
            print('READ_BYTES',self.total,flush=True);self.last=self.total
            if shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:raise RuntimeError('disk budget')
        if self.total>SIZE:raise RuntimeError('exceeded declared content length')
        return value

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:raise RuntimeError('disk budget')
    a.out.mkdir(exist_ok=False);found={};members=0
    with requests.get(URL,stream=True,timeout=(20,45),headers={'Accept-Encoding':'identity'}) as response:
      response.raise_for_status()
      if int(response.headers.get('Content-Length','-1'))!=SIZE:raise ValueError('content length mismatch')
      stream=Counted(response.raw)
      with tarfile.open(fileobj=stream,mode='r|',bufsize=1024*1024) as archive:
        for item in archive:
            members+=1;name=Path(item.name).name
            if name not in WANTED or not item.isfile():continue
            if name in found or item.size<=0 or item.size>1024*1024:raise ValueError('ambiguous or oversized member')
            payload=archive.extractfile(item).read()
            if len(payload)!=item.size:raise ValueError('truncated member')
            array=np.load(io.BytesIO(payload),allow_pickle=False)
            if array.ndim!=3 or array.shape[1:]!=(22,3) or not np.isfinite(array).all():raise ValueError('invalid joints')
            target=a.out/name
            with target.open('xb') as f:f.write(payload)
            digest=hashlib.sha256(payload).hexdigest()
            found[name]={'member':item.name,'bytes':len(payload),'sha256':digest,'shape':list(array.shape)}
            print('FOUND',name,digest,flush=True)
            if len(found)==len(WANTED):break
    if set(found)!=WANTED:raise ValueError('target members not all found')
    result={'url':URL,'expected_full_tar_size':SIZE,
            'publisher_reported_full_tar_sha256':'a261219891b1003b88c66f4d94ce46b6e192bc857fbaff8bcfbb63a908803219',
            'full_tar_sha256_verified':False,'bytes_streamed':stream.total,'tar_members_seen':members,'members':found,
            'one_official_example_sha256':OFFICIAL_EXAMPLE,
            'one_example_bytes_match_official':found['012314.npy']['sha256']==OFFICIAL_EXAMPLE,
            'training_ready':False,
            'limitations':['Third-party mirror, not an official release; no full tar verification.',
             'Matching one public example does not prove all mirror members are authentic.',
             'SMPL-H-derived normalized joints are not raw markers or confirmed SMPL-X timestamps.',
             'AMASS and source license restrictions still apply; diagnostic use only.']}
    with (a.out/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
