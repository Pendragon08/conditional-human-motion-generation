"""Independently check queue provenance/coverage and inspect unresolved sizes."""
import hashlib,io,json,zipfile
from collections import Counter
from pathlib import Path,PurePosixPath

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
Q=ROOT/'grab_remaining_contact_queue_20261030';STAGE=ROOT/'grab_mesh_stage_20261030'
QUEUE_HASH='1aa60e4efec116de1efbf5f90818ae5dfbc89ba7d387c032a0f3d2ed933aa57c'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    manifest=Q/'queue_manifest.json';complete=Q/'COMPLETE.json';source_manifest=STAGE/'manifest.json'
    if sha(manifest)!=QUEUE_HASH:raise ValueError('queue manifest changed')
    jobs=json.loads(manifest.read_text())['subjects'];end=json.loads(complete.read_text())
    if end['manifest_sha256']!=QUEUE_HASH or sha(Q/'jobs.jsonl')!=end['jobs_sha256']:raise ValueError('queue completion provenance mismatch')
    if sha(Q/'audit_grab_subject_contacts.py')!=end['program_sha256'] or sha(Q/'grab_restricted_reader.py')!=end['reader_sha256']:raise ValueError('program changed')
    finished={x['subject']:x for x in end['subjects']}
    if len(jobs)!=8 or len(finished)!=8:raise ValueError('queue incomplete')
    stage_hash=sha(source_manifest);docs=json.loads(source_manifest.read_text())['files'];assets={x['path']:x for x in docs}
    if len(assets)!=len(docs):raise ValueError('duplicate stage asset')
    checked={};subjects={};errors=[];allfps=Counter();total=Counter()
    for job in jobs:
        s=job['subject'];directory=Q/s;archive=ROOT/'data/GRAB'/f'grab__{s}.zip';report_file=directory/'report.json';rowsfile=directory/'rows.jsonl'
        if sha(archive)!=job['sha256'] or sha(report_file)!=finished[s]['report_sha256']:raise ValueError('archive or report changed')
        report=json.loads(report_file.read_text());rows=[json.loads(x) for x in rowsfile.read_text().splitlines()]
        if (report['archive_sha256']!=job['sha256'] or report['stage_manifest_sha256']!=stage_hash or
            report['script_sha256']!=end['program_sha256'] or report['reader_sha256']!=end['reader_sha256'] or
            sha(directory/'audit_grab_subject_contacts.py')!=end['program_sha256'] or sha(directory/'grab_restricted_reader.py')!=end['reader_sha256'] or
            sha(rowsfile)!=report['records_sha256'] or Counter(x['status'] for x in rows)!=Counter(report['counts'])):raise ValueError('row provenance mismatch '+s)
        with zipfile.ZipFile(archive) as z:
            members=sorted(n for n in z.namelist() if n.endswith('.npz'))
            if len(rows)!=job['count'] or sorted(x['member'] for x in rows)!=members or len({x['member'] for x in rows})!=len(rows):raise ValueError('coverage mismatch '+s)
            fps=Counter();gender=Counter();subject_errors=[]
            for row in rows:
                if row['status']!='ok':
                    item={'subject':s,**row}
                    if 'exceeds 128MiB cap' in row.get('error',''):
                        info=z.getinfo(row['member']);item['outer_bytes']=info.file_size
                        if info.file_size<=128*1024**2:
                            with zipfile.ZipFile(io.BytesIO(z.read(info))) as nested:
                                item['npy_bytes']={x.filename:x.file_size for x in nested.infolist() if x.filename in ('body.npy','object.npy','contact.npy')}
                    errors.append(item);subject_errors.append(item);continue
                fps[str(row['fps'])]+=1;gender[row['gender']]+=1
                for field,hfield in (('mesh','mesh_sha256'),('template','template_sha256')):
                    ref=PurePosixPath(row[field]);key=str(ref)
                    if ref.is_absolute() or '..' in ref.parts or key not in assets:raise ValueError('bad staged asset path')
                    if key not in checked:
                        actual=(STAGE/key).resolve(strict=True)
                        if not actual.is_relative_to(STAGE.resolve()) or sha(actual)!=assets[key]['sha256']:raise ValueError('asset hash changed')
                        checked[key]=sha(actual)
                    if checked[key]!=row[hfield]:raise ValueError('row asset mismatch')
            allfps.update(fps);total.update(report['counts'])
            subjects[s]={'sequences':len(rows),'status_counts':report['counts'],'fps_counts':dict(fps),'gender_counts':dict(gender),'unresolved':len(subject_errors),'report_sha256':sha(report_file)}
    if sum(v['sequences'] for v in subjects.values())!=1117 or sum(total.values())!=1117:raise ValueError('unexpected total')
    out={'queue_manifest_sha256':QUEUE_HASH,'queue_complete_sha256':sha(complete),'stage_manifest_sha256':stage_hash,
         'script_sha256':sha(Path(__file__)),'subjects':subjects,'sequences':1117,'status_counts':dict(total),'fps_counts':dict(allfps),
         'unique_assets_rehashed':len(checked),'unresolved':errors,
         'limitations':'Technical structure and declared FPS on verified trusted archives; restricted reader and 128MiB caps. Oversize is unresolved, not evidence of corruption. Counts do not establish vertex order, true temporal sampling, contact force, physics, retargeting or training readiness.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'sequences':1117,'status_counts':out['status_counts'],'fps_counts':out['fps_counts'],'unresolved':len(errors),'unique_assets_rehashed':len(checked)}),flush=True)
if __name__=='__main__':main()
