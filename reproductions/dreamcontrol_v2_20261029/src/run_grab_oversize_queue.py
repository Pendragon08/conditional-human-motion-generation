"""Sequentially recheck 128MiB-capped GRAB fields in isolated directories.

Only already-inspected, trusted archive entries are handled, with a per-entry
192 or 256 MiB raw-field cap. No automatic resume or overwrite is implemented.
"""
import hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
AGG=ROOT/'grab_remaining_contact_aggregate_20261030/report.json'
QUEUE=ROOT/'grab_remaining_contact_queue_20261030/queue_manifest.json'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    if len(sys.argv)!=2 or sha(AGG)!=sys.argv[1]:raise ValueError('aggregate input hash mismatch')
    agg=json.loads(AGG.read_text());manifest=json.loads(QUEUE.read_text())
    jobs={j['subject']:j for j in manifest['subjects']};entries=agg['unresolved']
    if len(entries)!=30 or len({(e['subject'],e['member']) for e in entries})!=30:raise ValueError('unexpected unresolved list')
    files={c:(HERE/f'audit_{c}.py',HERE/f'reader_{c}.py') for c in (192,256)}
    p192,r192=files[192];p256,r256=files[256]
    def change_cap(data,old,new):
        return data.replace(f'{old}*1024**2'.encode(),f'{new}*1024**2'.encode()).replace(f'{old}MiB'.encode(),f'{new}MiB'.encode())
    if change_cap(p192.read_bytes(),192,256)!=p256.read_bytes() or change_cap(r192.read_bytes(),192,256)!=r256.read_bytes():raise ValueError('reader/program variants differ beyond cap')
    original=ROOT/'grab_remaining_contact_queue_20261030'
    if change_cap(p192.read_bytes(),192,128)!=(original/'audit_grab_subject_contacts.py').read_bytes():raise ValueError('program differs beyond cap')
    if change_cap(r192.read_bytes(),192,128)!=(original/'grab_restricted_reader.py').read_bytes():raise ValueError('reader differs beyond cap')
    source_hashes={str(c):{'program':sha(files[c][0]),'reader':sha(files[c][1])} for c in files}
    outcomes=[]
    with (HERE/'jobs.jsonl').open('x',encoding='utf8') as log:
        for number,entry in enumerate(entries):
            s=entry['subject'];member=entry['member'];job=jobs[s];size=entry['npy_bytes']['contact.npy']
            if entry['error']!='contact exceeds 128MiB cap' or not 128*1024**2<size<=256*1024**2:raise ValueError('unexpected field size/error')
            cap=192 if size<=192*1024**2 else 256
            archive=ROOT/'data/GRAB'/f'grab__{s}.zip'
            if sha(archive)!=job['sha256']:raise ValueError('source archive mismatch')
            with zipfile.ZipFile(archive) as z:
                members=sorted(n for n in z.namelist() if n.endswith('.npz'))
                if len(members)!=job['count'] or member not in members:raise ValueError('source member missing')
                index=members.index(member)
            target=HERE/f'{s}_{number:02d}';target.mkdir(mode=0o700)
            for src,dest in ((files[cap][0],'audit_grab_subject_contacts.py'),(files[cap][1],'grab_restricted_reader.py')):
                with (target/dest).open('xb') as f:f.write(src.read_bytes())
            cmd=[sys.executable,str(target/'audit_grab_subject_contacts.py'),'--subject',s,'--archive-sha256',job['sha256'],'--expected-count',str(job['count']),'--start-index',str(index),'--limit','1']
            with (target/'run.log').open('x',encoding='utf8') as output:
                result=subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,env=os.environ.copy(),check=False)
            if result.returncode or not (target/'report.json').is_file():raise RuntimeError('audit failed '+member)
            report=json.loads((target/'report.json').read_text());records=[json.loads(x) for x in (target/'rows.jsonl').read_text().splitlines()]
            if (len(records)!=1 or records[0]['member']!=member or report['archive_sha256']!=job['sha256'] or
                report['script_sha256']!=source_hashes[str(cap)]['program'] or report['reader_sha256']!=source_hashes[str(cap)]['reader'] or
                sha(target/'rows.jsonl')!=report['records_sha256']):raise ValueError('audit result mismatch '+member)
            row={'subject':s,'member':member,'cap_mib':cap,'source_contact_npy_bytes':size,'status':records[0]['status'],
                 'result':records[0],'report_sha256':sha(target/'report.json'),'directory':target.name}
            log.write(json.dumps(row)+'\n');log.flush();outcomes.append(row)
            print(json.dumps({'number':number+1,'subject':s,'member':member,'cap_mib':cap,'status':row['status']}),flush=True)
    if sha(AGG)!=sys.argv[1] or any(sha(files[c][0])!=source_hashes[str(c)]['program'] or sha(files[c][1])!=source_hashes[str(c)]['reader'] for c in files):raise RuntimeError('queue inputs changed')
    summary={'aggregate_sha256':sys.argv[1],'queue_manifest_sha256':sha(QUEUE),'script_sha256':sha(Path(__file__)),
             'source_hashes':source_hashes,'jobs_sha256':sha(HERE/'jobs.jsonl'),'outcomes':outcomes,
             'limitations':'Read-only trusted data, bounded deserialization and structural checks only. Unresolved results are not automatically corrupt. Physical contact, vertex order, temporal truth and train readiness not established.'}
    with (HERE/'COMPLETE.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps({'processed':len(outcomes),'ok':sum(x['status']=='ok' for x in outcomes),'unresolved':sum(x['status']!='ok' for x in outcomes)}),flush=True)
if __name__=='__main__':main()
