"""Verify oversize rechecks and merge with eight-subject structural reports."""
import hashlib,json,zipfile
from collections import Counter
from pathlib import Path,PurePosixPath

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MAIN=ROOT/'grab_remaining_contact_queue_20261030';EXTRA=ROOT/'grab_oversize_contact_queue_v2_20261030'
AGG=ROOT/'grab_remaining_contact_aggregate_20261030/report.json';STAGE=ROOT/'grab_mesh_stage_20261030'
EXPECTED='82f8fb7b4226aedb64a5ab9432cb84d0f9250a1cde327cd0690bfd82c412bea5'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    if sha(AGG)!=EXPECTED:raise ValueError('aggregate changed')
    agg=json.loads(AGG.read_text());done_path=EXTRA/'COMPLETE.json';done=json.loads(done_path.read_text())
    if done['aggregate_sha256']!=EXPECTED or sha(EXTRA/'jobs.jsonl')!=done['jobs_sha256'] or sha(EXTRA/'run_grab_oversize_queue.py')!=done['script_sha256'] or sha(MAIN/'queue_manifest.json')!=done['queue_manifest_sha256']:raise ValueError('queue provenance mismatch')
    outcomes=[json.loads(x) for x in (EXTRA/'jobs.jsonl').read_text().splitlines()]
    if outcomes!=done['outcomes'] or len(outcomes)!=30:raise ValueError('missing or differing outcomes')
    originals={(x['subject'],x['member']):x for x in agg['unresolved']}
    if len(originals)!=30:raise ValueError('duplicate original unresolved')
    replacements={};caps=Counter();remaining=[]
    for result in outcomes:
        key=(result['subject'],result['member'])
        if key not in originals or key in replacements:raise ValueError('unexpected or duplicate retry')
        original=originals[key];size=original['npy_bytes']['contact.npy'];cap=result['cap_mib']
        if result['source_contact_npy_bytes']!=size or cap not in (192,256) or size>cap*1024**2 or (cap==256 and size<=192*1024**2):raise ValueError('incorrect cap')
        directory=EXTRA/result['directory'];rfile=directory/'report.json';r=json.loads(rfile.read_text());rowsfile=directory/'rows.jsonl';records=[json.loads(x) for x in rowsfile.read_text().splitlines()]
        sources=done['source_hashes'][str(cap)]
        if (sha(rfile)!=result['report_sha256'] or sha(rowsfile)!=r['records_sha256'] or len(records)!=1 or records[0]!=result['result'] or
            r['archive_sha256']!=next(j['sha256'] for j in json.loads((MAIN/'queue_manifest.json').read_text())['subjects'] if j['subject']==key[0]) or
            r['script_sha256']!=sources['program'] or r['reader_sha256']!=sources['reader'] or
            sha(directory/'audit_grab_subject_contacts.py')!=sources['program'] or sha(directory/'grab_restricted_reader.py')!=sources['reader']):raise ValueError('retry provenance mismatch')
        replacements[key]=records[0];caps[str(cap)]+=1
        if records[0]['status']!='ok':remaining.append({'subject':key[0],**records[0]})
    manifest=json.loads((MAIN/'queue_manifest.json').read_text())['subjects'];stage_manifest=STAGE/'manifest.json'
    if sha(stage_manifest)!=agg['stage_manifest_sha256']:raise ValueError('stage manifest changed')
    asset_entries={x['path']:x for x in json.loads(stage_manifest.read_text())['files']};checked={};counts=Counter();fps=Counter();genders=Counter();subjects={};allframes=0
    for job in manifest:
        subject=job['subject'];archive=ROOT/'data/GRAB'/f'grab__{subject}.zip'
        if sha(archive)!=job['sha256']:raise ValueError('archive changed')
        rows=[json.loads(x) for x in (MAIN/subject/'rows.jsonl').read_text().splitlines()]
        with zipfile.ZipFile(archive) as z:members=sorted(x for x in z.namelist() if x.endswith('.npz'))
        if sorted(x['member'] for x in rows)!=members or len(rows)!=job['count']:raise ValueError('coverage changed')
        subject_status=Counter();subject_fps=Counter()
        for row in rows:
            key=(subject,row['member'])
            if row['status']=='unresolved':
                if key not in replacements:raise ValueError('unmatched unresolved')
                row=replacements[key]
            subject_status[row['status']]+=1;counts[row['status']]+=1
            if row['status']!='ok':continue
            fps[str(row['fps'])]+=1;subject_fps[str(row['fps'])]+=1;genders[row['gender']]+=1;allframes+=row['frames']
            for field,hfield in (('template','template_sha256'),('mesh','mesh_sha256')):
                ref=PurePosixPath(row[field]);name=str(ref)
                if ref.is_absolute() or '..' in ref.parts or name not in asset_entries:raise ValueError('unsafe asset')
                if name not in checked:
                    path=(STAGE/name).resolve(strict=True)
                    if not path.is_relative_to(STAGE.resolve()) or sha(path)!=asset_entries[name]['sha256']:raise ValueError('asset hash mismatch')
                    checked[name]=sha(path)
                if row[hfield]!=checked[name]:raise ValueError('row asset mismatch')
        subjects[subject]={'total':len(rows),'counts':dict(subject_status),'fps_counts':dict(subject_fps)}
    if sum(counts.values())!=1117:raise ValueError('incorrect total')
    out={'script_sha256':sha(Path(__file__)),'aggregate_sha256':EXPECTED,'oversize_complete_sha256':sha(done_path),
         'oversize_jobs_sha256':done['jobs_sha256'],'stage_manifest_sha256':sha(stage_manifest),'subjects':subjects,
         'total':1117,'status_counts':dict(counts),'fps_counts':dict(fps),'gender_counts':dict(genders),
         'successful_frames':allframes,'assets_rehashed':len(checked),'retry_caps_mib':dict(caps),'remaining':remaining,
         'limitations':'Structural/timing-declaration checks of trusted archives only. Retried fields use 192/256MiB raw-field caps. No claim of malicious-pickle safety, real frame-rate, matching vertex order, physical contact or training readiness.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'total':out['total'],'status_counts':out['status_counts'],'fps_counts':out['fps_counts'],'retry_caps_mib':out['retry_caps_mib'],'remaining':len(remaining)}),flush=True)
if __name__=='__main__':main()
