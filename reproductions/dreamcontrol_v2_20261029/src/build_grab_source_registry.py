"""Build an auditable GRAB source inventory; does not decode or approve training data.

Run only on the checked GRAB archives and the prior structural audit outputs. Each
uncompressed NPZ member is hashed as opaque bytes; no pickle is opened here.
"""
import hashlib
import json
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE = ROOT / 'grab_mesh_stage_20261030'
MAIN = ROOT / 'grab_remaining_contact_queue_20261030'
EXTRA = ROOT / 'grab_oversize_contact_queue_v2_20261030'
OUT = Path(__file__).resolve().parent
S2_HASH = 'd77eb7d13cfb3e3d0b81f08ccf9d8d79320fa6541352e6e65173c56d4b01b9b6'
S3_HASH = 'c09458512f01e2c7415fb8a4d343e158041e0f8fa1bb8da08822a2bb30dd1a66'
PINS = {
    'grab_body_audit_20261030/all_structure_report.json': 'f9dda087749238b76985322105940a348f502ba25eee6618dfdea90af6b26969',
    'grab_oversize_audit_20261030/report.json': '640e4b4c30474c808bb7477758e1536651aa670873f6ba1799b59c9dce3da6cf',
    'grab_s2_contact_mesh_sizes_20261030/rows.jsonl': 'b0ab4544b037018fe7b5724df2f06ef0068cc7300a00636883eb37b7b4be884c',
    'grab_s3_contact_results_20261030/report.json': '2e820536b5e729eaddfa97aa46934d460e85cefcd9328536098bdd31f7c8a5cb',
    'grab_oversize_results_audit_20261030/report.json': '0118d08f87b0dce3136593815cfb0be77804a476b13b98de5be21a5848915b9f',
    'grab_mesh_stage_20261030/manifest.json': '2c98034afb63c38ffee86843484eac9f108fbef55b227032508e9223be2230f0',
    'grab_remaining_contact_queue_20261030/queue_manifest.json': '1aa60e4efec116de1efbf5f90818ae5dfbc89ba7d387c032a0f3d2ed933aa57c',
    'grab_remaining_contact_aggregate_20261030/report.json': '82f8fb7b4226aedb64a5ab9432cb84d0f9250a1cde327cd0690bfd82c412bea5',
}

def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()

def sha(path):
    with path.open('rb') as f:
        return digest(f)

def obj(path):
    return json.loads(path.read_text(encoding='utf8'))

def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]

def unique_records(records):
    result = {r['member']: r for r in records}
    if len(result) != len(records):
        raise ValueError('duplicate audit member')
    return result

def main():
    for name, expected in PINS.items():
        if sha(ROOT / name) != expected:
            raise ValueError('pinned input changed: ' + name)
    manifest = obj(STAGE / 'manifest.json')
    assets = {a['path']: a for a in manifest['files']}
    if len(assets) != len(manifest['files']):
        raise ValueError('duplicate staged asset')
    checked_assets = {}
    def check_asset(name, expected=None):
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or str(p) not in assets:
            raise ValueError('invalid asset path: ' + name)
        if name not in checked_assets:
            actual = (STAGE / name).resolve(strict=True)
            if not actual.is_relative_to(STAGE.resolve()) or sha(actual) != assets[name]['sha256']:
                raise ValueError('asset hash changed: ' + name)
            checked_assets[name] = assets[name]['sha256']
        if expected is not None and expected != checked_assets[name]:
            raise ValueError('audit asset hash mismatch: ' + name)
        return checked_assets[name]

    jobs = {j['subject']: j for j in obj(MAIN / 'queue_manifest.json')['subjects']}
    jobs.update({'s2': {'count': 93, 'sha256': S2_HASH}, 's3': {'count': 125, 'sha256': S3_HASH}})
    complete = obj(EXTRA / 'COMPLETE.json')
    previous = obj(ROOT / 'grab_oversize_results_audit_20261030/report.json')
    aggregate = obj(ROOT / 'grab_remaining_contact_aggregate_20261030/report.json')
    if sha(EXTRA / 'COMPLETE.json') != previous['oversize_complete_sha256'] or sha(EXTRA / 'jobs.jsonl') != previous['oversize_jobs_sha256']:
        raise ValueError('oversize outcomes changed')
    outcomes = rows(EXTRA / 'jobs.jsonl')
    if outcomes != complete['outcomes'] or len(outcomes) != 30:
        raise ValueError('oversize outcomes mismatch')
    replacements = {(x['subject'], x['member']): x['result'] for x in outcomes}
    if len(replacements) != 30:
        raise ValueError('duplicate oversize outcome')
    s3_summary = obj(ROOT / 'grab_s3_contact_results_20261030/report.json')
    s2_sizes = {x['sequence']: x for x in rows(ROOT / 'grab_s2_contact_mesh_sizes_20261030/rows.jsonl')}
    if len(s2_sizes) != 93 or any(not x['counts_match'] for x in s2_sizes.values()):
        raise ValueError('S2 size audit mismatch')
    source_hashes = dict(PINS)
    grouped_hashes = defaultdict(list)
    counts = Counter()
    output = OUT / 'source_registry.jsonl'
    with output.open('x', encoding='utf8', newline='\n') as dest:
        for subject in sorted(jobs, key=lambda s: int(s[1:])):
            job = jobs[subject]
            archive = ROOT / 'data/GRAB' / ('grab__' + subject + '.zip')
            if sha(archive) != job['sha256']:
                raise ValueError('archive hash changed: ' + subject)
            if subject == 's2':
                s2_reports = [obj(ROOT / 'grab_body_audit_20261030/all_structure_report.json'), obj(ROOT / 'grab_oversize_audit_20261030/report.json')]
                if any(x['archive_sha256'] != S2_HASH for x in s2_reports):
                    raise ValueError('S2 archive provenance mismatch')
                records = unique_records([r for rep in s2_reports for r in rep['sequences']])
                if set(records) != set(s2_sizes):
                    raise ValueError('S2 coverage mismatch')
            elif subject == 's3':
                records = {}
                for dirname, expected in s3_summary['input_report_sha256'].items():
                    directory = ROOT / dirname
                    report_path = directory / 'report.json'
                    if sha(report_path) != expected:
                        raise ValueError('S3 report changed')
                    report = obj(report_path)
                    rpath = directory / 'rows.jsonl'
                    if sha(rpath) != report['records_sha256'] or report['archive_sha256'] != S3_HASH:
                        raise ValueError('S3 rows changed')
                    source_hashes[str(rpath.relative_to(ROOT))] = report['records_sha256']
                    for row in rows(rpath):
                        old = records.get(row['member'])
                        if old and not (old['status'] == 'unresolved' and row['status'] == 'ok'):
                            raise ValueError('unexpected S3 replacement')
                        records[row['member']] = row
            else:
                directory = MAIN / subject
                rp = directory / 'report.json'
                report = obj(rp)
                row_path = directory / 'rows.jsonl'
                if sha(rp) != aggregate['subjects'][subject]['report_sha256'] or report['archive_sha256'] != job['sha256'] or sha(row_path) != report['records_sha256']:
                    raise ValueError('subject rows changed: ' + subject)
                source_hashes[str(rp.relative_to(ROOT))] = sha(rp)
                source_hashes[str(row_path.relative_to(ROOT))] = report['records_sha256']
                records = unique_records(rows(row_path))
                for name, row in list(records.items()):
                    if row['status'] == 'unresolved':
                        records[name] = replacements[(subject, name)]
            with zipfile.ZipFile(archive) as z:
                infos = [x for x in z.infolist() if x.filename.endswith('.npz')]
                members = [x.filename for x in infos]
                if len(members) != job['count'] or len(set(members)) != len(members) or set(members) != set(records):
                    raise ValueError('archive member coverage mismatch: ' + subject)
                for info in sorted(infos, key=lambda x: x.filename):
                    name = info.filename
                    p = PurePosixPath(name)
                    if p.is_absolute() or '..' in p.parts or len(p.parts) != 2 or p.parts[0] != subject:
                        raise ValueError('unsafe member path: ' + name)
                    r = records[name]
                    if subject == 's2':
                        template, mesh = r['subject_template'], r['object_mesh']
                        size = s2_sizes[name]
                        if size['mesh'] != mesh or size['contact_array_width'] != r['object_contact_shape'][1] or size['ply_vertices'] != r['object_contact_shape'][1]:
                            raise ValueError('S2 mesh audit mismatch')
                        body_n, object_n = r['body_contact_shape'][1], r['object_contact_shape'][1]
                        template_hash, mesh_hash = check_asset(template), check_asset(mesh)
                    else:
                        if r['status'] != 'ok':
                            raise ValueError('unresolved record: ' + name)
                        template, mesh = r['template'], r['mesh']
                        template_hash = check_asset(template, r['template_sha256'])
                        mesh_hash = check_asset(mesh, r['mesh_sha256'])
                        body_n, object_n = r['body_vertices'], r['object_vertices']
                    with z.open(info) as member:
                        member_sha = digest(member)
                    item = {'dataset': 'GRAB', 'subject': subject, 'source_group': name,
                            'archive': archive.name, 'archive_sha256': job['sha256'], 'member': name,
                            'member_sha256': member_sha, 'member_bytes': info.file_size,
                            'frames': r['frames'], 'fps_declared': r['fps'], 'gender': r['gender'],
                            'template': template, 'template_sha256': template_hash, 'body_vertices': body_n,
                            'object_mesh': mesh, 'object_mesh_sha256': mesh_hash, 'object_vertices': object_n,
                            'audit_scope': 'structure_only', 'real_fps_verified': False,
                            'vertex_order_verified': False, 'contact_physics_verified': False,
                            'training_ready': False}
                    dest.write(json.dumps(item, sort_keys=True, allow_nan=False) + '\n')
                    grouped_hashes[member_sha].append(name)
                    counts[subject] += 1
            dest.flush()
            os.fsync(dest.fileno())
            if sha(archive) != job['sha256']:
                raise ValueError('archive changed while reading: ' + subject)
            print(json.dumps({'subject': subject, 'records': counts[subject]}), flush=True)
    if sum(counts.values()) != 1335:
        raise ValueError('incomplete registry')
    for name, expected in source_hashes.items():
        if sha(ROOT / name) != expected:
            raise ValueError('input changed during run: ' + name)
    duplicates = {k: v for k, v in grouped_hashes.items() if len(v) > 1}
    report = {'script_sha256': sha(Path(__file__)), 'registry_sha256': sha(output),
              'input_sha256': source_hashes, 'archive_sha256': {k: v['sha256'] for k, v in jobs.items()},
              'subject_counts': dict(counts), 'records': sum(counts.values()),
              'assets_verified': len(checked_assets), 'byte_identical_member_groups': duplicates,
              'limitations': 'Source inventory only. Byte hashes do not detect semantic duplicates; declared FPS, vertex order, contact and physical quality remain unverified. No training split or training acceptance.'}
    with (OUT / 'report.json').open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({'complete': True, 'records': report['records'], 'duplicate_groups': len(duplicates)}), flush=True)

if __name__ == '__main__':
    main()
