"""Acquire the official link exposed by the authorized BABEL download page.

The independently accepted BABEL license permits personal research use, not
redistribution. Keep the original ZIP and all JSON records outside Git.
"""
import datetime,hashlib,json,pathlib,shutil,sys,urllib.request,zipfile
from session import REPO
sys.stdout.reconfigure(encoding='utf-8')
BASE=REPO/'baselines/textop';DEST=BASE/'results/data_preparation_20261025'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024');OUT=LOCAL/'datasets/BABEL'
URL='https://babel.is.tue.mpg.de/media/upload/data/babel_v1-0_release.zip'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);final=OUT/'babel_v1-0_release.zip'
    if not final.exists():
        part=final.with_suffix('.zip.partial');assert not part.exists(),'Previous partial retained; inspect before retry'
        client=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
        with client.open(URL,timeout=90) as response:
            kind=response.headers.get('Content-Type','');assert 'text/html' not in kind,'Download returned HTML, not ZIP'
            expected=int(response.headers.get('Content-Length','0'))
            assert expected<2*1024**3 and shutil.disk_usage(OUT).free>expected+2*1024**3
            prefix=response.read(4);assert prefix.startswith(b'PK'),'Download did not return a ZIP'
            with part.open('xb') as f:
                f.write(prefix)
                while True:
                    block=response.read(4*1024**2)
                    if not block:break
                    assert shutil.disk_usage(OUT).free>len(block)+2*1024**3
                    f.write(block)
            if expected:assert part.stat().st_size==expected
        with zipfile.ZipFile(part) as archive:assert archive.testzip() is None
        part.rename(final)
    files=[]
    with zipfile.ZipFile(final) as archive:
        assert archive.testzip() is None
        members=[x for x in archive.infolist() if not x.is_dir()]
        assert sum(x.file_size for x in members)<2*1024**3
        assert shutil.disk_usage(OUT).free>sum(x.file_size for x in members)+2*1024**3
        for member in members:
            pure=pathlib.PurePosixPath(member.filename)
            assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in member.filename
            if pure.name not in ['train.json','val.json','test.json','extra_train.json','extra_val.json']:continue
            target=OUT/pure.name;data=archive.read(member);digest=hashlib.sha256(data).hexdigest()
            if target.exists():assert sha(target)==digest
            else:
                with target.open('xb') as f:f.write(data)
            obj=json.loads(data)
            assert isinstance(obj,dict)
            def count_labels(singular,plural):
                return sum(len((ann or {}).get('labels') or []) for v in obj.values()
                           for ann in ((v.get(plural) or []) if plural in v else [v.get(singular)]))
            seq_labels=count_labels('seq_ann','seq_anns')
            frame_labels=count_labels('frame_ann','frame_anns')
            files.append({'file':pure.name,'bytes':len(data),'sha256':digest,'sequence_records':len(obj),
                          'sequence_labels':seq_labels,'frame_labels':frame_labels})
    assert {x['file'] for x in files}=={'train.json','val.json','test.json','extra_train.json','extra_val.json'}
    receipt={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
             'official_url':URL,'download_page_observed_authenticated':True,
             'independent_license_already_accepted':True,'license_url':'https://babel.is.tue.mpg.de/license.html',
             'status':'OFFICIAL_BABEL_V1_0_DOWNLOADED_ZIP_CRC_AND_JSON_VERIFIED',
             'archive_bytes':final.stat().st_size,'archive_sha256':sha(final),'zip_crc_verified':True,
             'files':files,'raw_annotations_published':False,'motion_alignment_validated':False}
    (DEST/'babel_acquisition.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (LOCAL/'BABEL原始标注下载核验_20261025.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
