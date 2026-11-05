"""Single-process, sequential CPU queue for read-only GRAB structure audits.

Each subject receives an exclusive new directory; partial results are retained.
There is deliberately no automatic resume or replacement of existing outputs.
"""
import hashlib,json,os,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    if len(sys.argv)!=2 or len(sys.argv[1])!=64:raise ValueError('expected manifest SHA-256')
    manifest=HERE/'queue_manifest.json';expected=sys.argv[1]
    if sha(manifest)!=expected:raise ValueError('queue manifest mismatch')
    jobs=json.loads(manifest.read_text())['subjects'];subjects=[j['subject'] for j in jobs]
    if len(jobs)!=8 or len(set(subjects))!=8 or set(subjects)!={'s1','s4','s5','s6','s7','s8','s9','s10'}:raise ValueError('unexpected subject list')
    program=HERE/'audit_grab_subject_contacts.py';reader=HERE/'grab_restricted_reader.py'
    program_hash=sha(program);reader_hash=sha(reader);complete=[]
    with (HERE/'jobs.jsonl').open('x',encoding='utf8') as log:
        for job in jobs:
            subject=job['subject'];target=HERE/subject;target.mkdir(mode=0o700)
            for src in (program,reader):
                with (target/src.name).open('xb') as f:f.write(src.read_bytes())
            command=[sys.executable,str(target/program.name),'--subject',subject,'--archive-sha256',job['sha256'],'--expected-count',str(job['count'])]
            with (target/'run.log').open('x',encoding='utf8') as output:
                result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=os.environ.copy(),check=False)
            report_path=target/'report.json'
            if result.returncode or not report_path.is_file():
                failure={'subject':subject,'returncode':result.returncode,'status':'failed'}
                log.write(json.dumps(failure)+'\n');log.flush();raise RuntimeError('subject audit failed: '+subject)
            report=json.loads(report_path.read_text())
            if (report['subject']!=subject or report['archive_sha256']!=job['sha256'] or
                report['scanned_sequences']!=job['count'] or report['script_sha256']!=program_hash or
                report['reader_sha256']!=reader_hash or sha(target/'rows.jsonl')!=report['records_sha256']):raise ValueError('subject report mismatch')
            row={'subject':subject,'status':'complete','report_sha256':sha(report_path),'counts':report['counts'],'scanned':report['scanned_sequences']}
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True);complete.append(row)
    if sha(manifest)!=expected or sha(program)!=program_hash or sha(reader)!=reader_hash:raise RuntimeError('queue inputs changed')
    report={'manifest_sha256':expected,'program_sha256':program_hash,'reader_sha256':reader_hash,'jobs_sha256':sha(HERE/'jobs.jsonl'),
            'subjects':complete,'warning':'Structural audits only; unresolved cap-limited records are not corrupt. No contact truth, time alignment, physics, retargeting or training acceptance is established.'}
    with (HERE/'COMPLETE.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'complete':len(complete)}),flush=True)
if __name__=='__main__':main()
