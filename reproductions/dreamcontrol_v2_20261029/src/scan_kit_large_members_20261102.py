"""Read only KIT members skipped by the 64 MiB limit; no extraction to disk."""
import hashlib, io, json, math, os, shutil, tarfile, time, zipfile
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
prior = root/'data/kit_smplx_stream_inventory_20261102/records.jsonl'
out = root/'data/kit_smplx_large_members_20261102'
archive = Path(r'D:\具身智能\PhyGile复现\后续AMASS下载_20261021\archives\KIT_smplx_g.tar.bz2')
targets = {r['member']:r for r in (json.loads(x) for x in prior.read_text(encoding='utf-8').splitlines()) if r.get('error') == 'member exceeds 64 MiB per-record memory limit'}
if len(targets) != 33: raise RuntimeError('unexpected unresolved count')
out.mkdir(exist_ok=False)
st = archive.stat(); start = time.time(); found = set()
with (out/'records.jsonl').open('x',encoding='utf-8',buffering=1) as f, (out/'progress.jsonl').open('x',encoding='utf-8',buffering=1) as log, archive.open('rb') as raw:
    log.write(json.dumps({'event':'start','targets':len(targets),'archive_size':st.st_size,'archive_mtime_ns':st.st_mtime_ns})+'\n')
    with tarfile.open(fileobj=raw,mode='r|bz2') as tar:
        for index,m in enumerate(tar,1):
            if index%250==0:
                if shutil.disk_usage(out).free<512*1024*1024: raise RuntimeError('free disk below 512 MiB')
                log.write(json.dumps({'event':'progress','members_seen':index,'found':len(found),'bytes_approx':raw.tell(),'elapsed_s':round(time.time()-start,1)})+'\n')
            if m.name not in targets: continue
            rec={'member':m.name,'member_size':m.size,'nominal_rows':targets[m.name]['nominal_rows'],'training_accepted':False}
            try:
                if m.size>256*1024*1024: raise ValueError('member exceeds 256 MiB memory bound')
                blob=tar.extractfile(m).read()
                if len(blob)!=m.size: raise ValueError('truncated member')
                rec['member_sha256']=hashlib.sha256(blob).hexdigest()
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    with z.open('poses.npy') as p:
                        version=np.lib.format.read_magic(p)
                        if version==(1,0): shape,_,dtype=np.lib.format.read_array_header_1_0(p)
                        elif version==(2,0): shape,_,dtype=np.lib.format.read_array_header_2_0(p)
                        else: raise ValueError('unsupported numpy header')
                    with z.open('mocap_frame_rate.npy') as q: fps=float(np.lib.format.read_array(q,allow_pickle=False))
                if len(shape)!=2 or dtype.hasobject or not math.isfinite(fps) or fps<20: raise ValueError('invalid header')
                count=len(range(0,shape[0],int(fps/20)))
                rec.update({'poses_shape':list(shape),'declared_fps':fps,'nominal_20hz_count':count})
                for row in rec['nominal_rows']:
                    try:
                        a,b=int(row['start']),int(row['end'])
                        row['interval_within_nominal_count']= b<=count if 0<=a<b else None
                    except ValueError: row['interval_within_nominal_count']=None
                del blob
            except Exception as e: rec['error']=str(e)
            f.write(json.dumps(rec,ensure_ascii=False)+'\n');found.add(m.name)
            if found==set(targets):break
    end=archive.stat()
    summary={'event':'complete','found':len(found),'missing':sorted(set(targets)-found),'elapsed_s':round(time.time()-start,1),'size_unchanged':st.st_size==end.st_size,'mtime_unchanged':st.st_mtime_ns==end.st_mtime_ns,'training_ready':False,'limitations':['Name match is not identity. FPS is declared, not measured.','No physical or text acceptance.']}
    log.write(json.dumps(summary,ensure_ascii=False)+'\n')
    with (out/'summary.json').open('x',encoding='utf-8') as fsum:json.dump(summary,fsum,ensure_ascii=False,indent=2)
