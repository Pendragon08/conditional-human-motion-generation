"""Verify coverage, provenance and aggregate the split S3 contact audit."""
import hashlib,json,zipfile
from collections import Counter
from pathlib import Path,PurePosixPath

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE=ROOT/'grab_mesh_stage_20261030'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    archive=ROOT/'data/GRAB/grab__s3.zip';expected='c09458512f01e2c7415fb8a4d343e158041e0f8fa1bb8da08822a2bb30dd1a66'
    if sha(archive)!=expected:raise ValueError('archive changed')
    manifest=STAGE/'manifest.json';manifest_hash=sha(manifest)
    dirs=(ROOT/'grab_s3_contact_audit_pilot_20261030',ROOT/'grab_s3_contact_audit_remaining_20261030')
    allrows=[];report_hashes={}
    for directory in dirs:
        report_path=directory/'report.json';report=json.loads(report_path.read_text());rows_path=directory/'rows.jsonl'
        if report['subject']!='s3' or report['expected_sequences']!=125 or report['archive_sha256']!=expected or report['stage_manifest_sha256']!=manifest_hash:raise ValueError('source metadata mismatch')
        if sha(rows_path)!=report['records_sha256'] or sha(directory/'audit_grab_subject_contacts.py')!=report['script_sha256'] or sha(directory/'grab_restricted_reader.py')!=report['reader_sha256']:raise ValueError('audit source or rows changed')
        rows=[json.loads(line) for line in rows_path.read_text().splitlines()]
        if len(rows)!=report['scanned_sequences'] or Counter(r['status'] for r in rows)!=Counter(report['counts']):raise ValueError('counts differ')
        report_hashes[directory.name]=sha(report_path);allrows.extend(rows)
    extra=ROOT/'grab_s3_contact_oversize_20261030'
    extra_report=json.loads((extra/'report.json').read_text());extra_path=extra/'rows.jsonl'
    if (extra_report['subject']!='s3' or extra_report['archive_sha256']!=expected or
        extra_report['stage_manifest_sha256']!=manifest_hash or extra_report['expected_sequences']!=125 or
        sha(extra_path)!=extra_report['records_sha256'] or sha(extra/'audit_grab_subject_contacts_192.py')!=extra_report['script_sha256'] or
        sha(extra/'grab_restricted_reader.py')!=extra_report['reader_sha256']):raise ValueError('oversize audit provenance mismatch')
    replacement=[json.loads(x) for x in extra_path.read_text().splitlines()]
    if len(replacement)!=1 or replacement[0]['status']!='ok' or Counter(r['status'] for r in replacement)!=Counter(extra_report['counts']):raise ValueError('oversize audit did not succeed')
    matches=[i for i,row in enumerate(allrows) if row['member']==replacement[0]['member']]
    if len(matches)!=1 or allrows[matches[0]]['status']!='unresolved' or '128MiB cap' not in allrows[matches[0]]['error']:raise ValueError('unexpected replacement')
    first_failure=allrows[matches[0]];allrows[matches[0]]=replacement[0]
    report_hashes[extra.name]=sha(extra/'report.json')
    with zipfile.ZipFile(archive) as z:members=sorted(x for x in z.namelist() if x.endswith('.npz'))
    if len(allrows)!=125 or len({r['member'] for r in allrows})!=125 or sorted(r['member'] for r in allrows)!=members:raise ValueError('audit coverage missing or duplicated')
    fps=Counter();genders=Counter();meshes=set();frames=0;errors=[];verified={}
    manifest_docs=json.loads(manifest.read_text())['files'];manifest_entries={r['path']:r for r in manifest_docs}
    for row in allrows:
        if row['status']!='ok':errors.append(row);continue
        fps[str(row['fps'])]+=1;genders[row['gender']]+=1;meshes.add(row['mesh']);frames+=row['frames']
        for field,hfield in (('mesh','mesh_sha256'),('template','template_sha256')):
            ref=PurePosixPath(row[field]);key=str(ref)
            if ref.is_absolute() or '..' in ref.parts or key not in manifest_entries:raise ValueError('bad asset path')
            if key not in verified:
                path=(STAGE/key).resolve(strict=True)
                if not path.is_relative_to(STAGE.resolve()) or sha(path)!=manifest_entries[key]['sha256']:raise ValueError('asset changed')
                verified[key]=sha(path)
            if row[hfield]!=verified[key]:raise ValueError('record asset hash mismatch')
    result={'archive_sha256':expected,'stage_manifest_sha256':manifest_hash,'script_sha256':sha(Path(__file__)),
            'input_report_sha256':report_hashes,'sequences':125,'status_counts':dict(Counter(r['status'] for r in allrows)),
            'fps_counts':dict(fps),'gender_counts':dict(genders),'distinct_object_meshes':len(meshes),'validated_frames':frames,
            'verified_staged_assets':len(verified),'unresolved':errors,'resolved_oversize_original':first_failure,
            'limitations':'Restricted reader on trusted archive; contact dimensions, source-declared FPS and asset hashes only. Does not prove vertex order, actual contact, physics, real FPS, retargeting or training readiness.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
