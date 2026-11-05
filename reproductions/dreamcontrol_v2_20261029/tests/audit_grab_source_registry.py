"""Independently compare a completed GRAB registry to input bytes and audit rows."""
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN = ROOT / 'grab_source_registry_20261030'
MAIN = ROOT / 'grab_remaining_contact_queue_20261030'
EXTRA = ROOT / 'grab_oversize_contact_queue_v2_20261030'

def sha_stream(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()

def sha(p):
    with p.open('rb') as f:
        return sha_stream(f)

def obj(p):
    return json.loads(p.read_text(encoding='utf8'))

def lines(p):
    return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]

def main():
    report = obj(RUN / 'report.json')
    if sha(RUN / 'source_registry.jsonl') != report['registry_sha256'] or sha(RUN / 'build_grab_source_registry.py') != report['script_sha256']:
        raise ValueError('registry or builder hash mismatch')
    for path, expected in report['input_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('source input hash mismatch: ' + path)
    records = lines(RUN / 'source_registry.jsonl')
    if len(records) != 1335 or len({r['member'] for r in records}) != 1335:
        raise ValueError('record count or uniqueness mismatch')
    s2 = {}
    for path in ('grab_body_audit_20261030/all_structure_report.json', 'grab_oversize_audit_20261030/report.json'):
        for row in obj(ROOT / path)['sequences']:
            if row['member'] in s2: raise ValueError('duplicate s2 source')
            s2[row['member']] = row
    s3 = {}
    summary = obj(ROOT / 'grab_s3_contact_results_20261030/report.json')
    for dirname, digest in summary['input_report_sha256'].items():
        d = ROOT / dirname
        if sha(d / 'report.json') != digest: raise ValueError('s3 source report changed')
        for row in lines(d / 'rows.jsonl'):
            if row['member'] in s3 and not (s3[row['member']]['status'] == 'unresolved' and row['status'] == 'ok'):
                raise ValueError('unexpected s3 duplicate')
            s3[row['member']] = row
    extra = {(x['subject'], x['member']): x['result'] for x in lines(EXTRA / 'jobs.jsonl')}
    counts, duplicate_map, checked_assets = Counter(), defaultdict(list), {}
    archive_names = set()
    for subject in sorted(report['subject_counts'], key=lambda x: int(x[1:])):
        archive = ROOT / 'data/GRAB' / ('grab__' + subject + '.zip')
        if sha(archive) != report['archive_sha256'][subject]: raise ValueError('archive changed')
        if subject == 's2': sources = s2
        elif subject == 's3': sources = s3
        else:
            sources = {x['member']: x for x in lines(MAIN / subject / 'rows.jsonl')}
            for name, value in list(sources.items()):
                if value['status'] == 'unresolved': sources[name] = extra[(subject, name)]
        subset = [r for r in records if r['subject'] == subject]
        with zipfile.ZipFile(archive) as z:
            members = [i.filename for i in z.infolist() if i.filename.endswith('.npz')]
            if len(set(members)) != len(members) or set(members) != set(sources) or set(members) != {r['member'] for r in subset}:
                raise ValueError('member coverage mismatch: ' + subject)
            for row in subset:
                name = row['member']; src = sources[name]; info = z.getinfo(name)
                with z.open(info) as f: member_hash = sha_stream(f)
                expected_template = src['subject_template'] if subject == 's2' else src['template']
                expected_mesh = src['object_mesh'] if subject == 's2' else src['mesh']
                b = src['body_contact_shape'][1] if subject == 's2' else src['body_vertices']
                o = src['object_contact_shape'][1] if subject == 's2' else src['object_vertices']
                if (row['dataset'] != 'GRAB' or row['source_group'] != name or row['archive'] != archive.name or
                    row['archive_sha256'] != report['archive_sha256'][subject] or row['member_sha256'] != member_hash or
                    row['member_bytes'] != info.file_size or row['frames'] != src['frames'] or row['fps_declared'] != src['fps'] or
                    row['gender'] != src['gender'] or row['template'] != expected_template or row['object_mesh'] != expected_mesh or
                    row['body_vertices'] != b or row['object_vertices'] != o or row['training_ready'] is not False or
                    row['real_fps_verified'] is not False or row['vertex_order_verified'] is not False or row['contact_physics_verified'] is not False):
                    raise ValueError('row mismatch: ' + name)
                for field, sha_field in (('template','template_sha256'), ('object_mesh','object_mesh_sha256')):
                    asset = (ROOT / 'grab_mesh_stage_20261030' / row[field]).resolve(strict=True)
                    if not asset.is_relative_to((ROOT / 'grab_mesh_stage_20261030').resolve()):
                        raise ValueError('unsafe asset path: ' + name)
                    if asset not in checked_assets:
                        checked_assets[asset] = sha(asset)
                    if checked_assets[asset] != row[sha_field]:
                        raise ValueError('asset mismatch: ' + name)
                duplicate_map[member_hash].append(name); counts[subject] += 1
        archive_names.add(subject)
        print(json.dumps({'subject':subject,'validated':counts[subject]}), flush=True)
    duplicates = {k:v for k,v in duplicate_map.items() if len(v)>1}
    if dict(counts) != report['subject_counts'] or duplicates != report['byte_identical_member_groups'] or set(archive_names) != set(report['archive_sha256']):
        raise ValueError('aggregate mismatch')
    result = {'registry_sha256':report['registry_sha256'], 'source_report_sha256':sha(RUN / 'report.json'),
              'audit_script_sha256':sha(Path(__file__)), 'subjects':dict(counts), 'records':len(records),
              'byte_identical_member_groups':duplicates,
              'limitations':'Opaque byte and source-row verification only. Cannot detect semantic duplicates or verify real FPS, contact, physics, retargeting or training readiness.'}
    with (RUN / 'independent_audit.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({'complete': True, 'records': len(records), 'duplicates': len(duplicates)}), flush=True)

if __name__ == '__main__': main()
