"""Extract only indexed missing motions from already licensed local archives.

Writes durable private raw files on C:, never into the Git checkout. No network
download, data deletion, environment installation, or training is performed.
"""
import collections, hashlib, json, pathlib, shutil, sys, tarfile, time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from reconcile_raw_sources import source_key

HOME=pathlib.Path('C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/textop_reproduction_20261024/private_assets')
ARCHIVES=pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives')
EXPECTED=pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/下载与安装进度.json')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    rows=json.loads((HOME/'release_input_index.private.json').read_text(encoding='utf-8'))
    wanted={r['canonical_source']:r for r in rows if not r['raw_stageii_candidates'] and r['canonical_source'].split('/')[0] in ['BMLmovi','EyesJapanDataset']}
    assert len(wanted)==sum(not r['raw_stageii_candidates'] and r['canonical_source'].split('/')[0] in ['BMLmovi','EyesJapanDataset'] for r in rows), 'Ambiguous duplicate normalized key'
    out=HOME/'indexed_raw_expansion_v1';out.mkdir(parents=True,exist_ok=False)
    expected={x['subset']:x for x in json.loads(EXPECTED.read_text(encoding='utf-8'))['subsets']}
    receipt={'requested':len(wanted),'completed':0,'bytes':0,'groups':{},'status':'VERIFYING_LOCAL_ARCHIVES','new_downloads':0,'durable_storage':str(out),'physical_filter_executed':False}
    path=HOME/'raw_expansion_progress.json'
    def update():
        receipt['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());path.write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
    found={};record=out/'raw_index.private.jsonl'
    update()
    for group in ['BMLmovi','EyesJapanDataset']:
        archive=ARCHIVES/(group+'_smplx_g.tar.bz2');digest=sha(archive)
        assert digest==expected[group]['sha256'], 'Archive checksum mismatch: '+group
        receipt['groups'][group]={'archive_bytes':archive.stat().st_size,'archive_sha256':digest,'matched':0,'bytes':0};receipt['status']='EXTRACTING_INDEXED_MOTIONS';update()
        with tarfile.open(archive,'r|bz2') as source:
            for member in source:
                if not member.isfile() or not member.name.endswith('_stageii.npz'):continue
                key=source_key(member.name)
                if key not in wanted:continue
                if key in found:raise RuntimeError('Multiple archive members normalize to '+key)
                if shutil.disk_usage(out).free-member.size<6*1024**3:raise RuntimeError('C: reserve below6GiB; all original archives and completed files preserved')
                dest=out/(hashlib.sha256(key.encode()).hexdigest()[:20]+'_stageii.npz');partial=dest.with_suffix('.npz.partial');h=hashlib.sha256()
                with source.extractfile(member) as incoming,partial.open('xb') as output:
                    for block in iter(lambda:incoming.read(4*1024**2),b''):h.update(block);output.write(block)
                assert partial.stat().st_size==member.size
                partial.rename(dest)
                row={**wanted[key],'local_raw_path':str(dest),'raw_sha256':h.hexdigest(),'archive_member':member.name,'archive_sha256':digest,'raw_stageii_candidates':[]}
                found[key]=row
                with record.open('a',encoding='utf-8') as output:output.write(json.dumps(row)+'\n')
                receipt['completed']+=1;receipt['bytes']+=member.size;receipt['groups'][group]['matched']+=1;receipt['groups'][group]['bytes']+=member.size
                if receipt['completed']%10==0:update()
    receipt['unmatched_requested']=len(set(wanted)-set(found));receipt['status']='LOCAL_EXTRACTION_COMPLETE' if not receipt['unmatched_requested'] else 'LOCAL_EXTRACTION_PARTIAL_UNMATCHED';update()
    (out/'ready.private.json').write_text(json.dumps(list(found.values())),encoding='utf-8')

if __name__=='__main__':
    try:main()
    except Exception as error:
        p=HOME/'raw_expansion_progress.json';state=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {};state.update(status='STOPPED_WITH_ERROR',error=repr(error));p.write_text(json.dumps(state,indent=2),encoding='utf-8');raise
