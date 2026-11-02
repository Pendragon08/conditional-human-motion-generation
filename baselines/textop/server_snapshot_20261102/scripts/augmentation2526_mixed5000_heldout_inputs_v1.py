"""Input acceptance for a future mixed-5000 generated-reference evaluation.

The caller must authenticate its request and source files, own the shared lock,
and enforce resource and process supervision. This module does not launch a
simulator, load a model or claim that any trajectory is physically achievable.
"""
import hashlib
import json
import os
import pathlib

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
HOME = pathlib.Path('/home/ouyanghao/textop-user100k-evaluation-20261029')
VERIFY = 'user_augmentation2526_mixed5000_cpu_verify_v1'
EXPORT = 'user_augmentation2526_mixed5000_cpu_export_v1'
FK = 'user_ldm100k_generated_fk_v2'
ACTOR_SHA = 'f2ee798dd0ae3a9b7878c720742d444aa09747ea39829ed96a173f7e2de829c7'
MANIFEST_SHA = 'f277d4df6888f7253235c9a5bf32ddf1a960be3a6b6c48a58e0333449dfb0453'


def owned(path, root):
    path = pathlib.Path(path)
    if not path.is_absolute() or path.resolve() != path or root not in path.parents:
        raise ValueError('path outside accepted private tree')
    if path.is_symlink() or path.stat().st_uid != 1013:
        raise ValueError('unexpected private file ownership')
    return path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def accept(binding):
    """Return all 1,880 fixed generated references and accepted actor metadata.

    Digest bindings must come from authenticated prior actual closure records,
    never merely from the files being checked. Every reference is included;
    no physical outcome is used to select inputs. File checks are repeated
    under the caller's lock, but do not protect against unrelated writers.
    """
    if os.getuid() != 1013 or not isinstance(binding, dict):
        raise ValueError('invalid identity or binding')
    base = ROOT / 'step34_20261029'
    receipts = ROOT / 'receipts'
    verify_paths = dict(request=receipts/(VERIFY+'_request.json'), driver=receipts/(VERIFY+'_driver.json'),
                        outer=receipts/(VERIFY+'_outer_exit.json'), receipt=base/VERIFY/'receipt.json',
                        exit=base/(VERIFY+'_supervision')/'exit.json',
                        process=base/(VERIFY+'_supervision')/'process_exit.json',
                        resources=base/(VERIFY+'_supervision')/'resources.jsonl',
                        actor=base/EXPORT/'actor_RMS.private.pth')
    expected = binding['verify_hashes']
    if set(expected) != set(verify_paths) or expected['actor'] != ACTOR_SHA:
        raise ValueError('incomplete or incorrect independent verification binding')
    for key, path in verify_paths.items():
        if sha(owned(path, ROOT)) != expected[key]:
            raise ValueError('verification source differs: ' + key)
    receipt, outer = read(verify_paths['receipt']), read(verify_paths['outer'])
    exit_info, process = read(verify_paths['exit']), read(verify_paths['process'])
    if not (receipt['status'] == 'MIXED_PPO5000_ACTOR_INDEPENDENT_CPU_VERIFY_FINISHED'
            and receipt['output_writers_closed'] is True and receipt['independent_process_verification'] is True
            and receipt['result']['inference']['sha256'] == ACTOR_SHA
            and receipt['result']['optimizer_updates_performed'] == 0
            and outer['actual_returncode'] == 0 and outer['child_reaped'] is True
            and outer['request_sha256'] == expected['request']
            and outer['child_pid'] == process['supervisor_pid']
            and exit_info['request_sha256'] == expected['request']
            and exit_info['receipt_sha256'] == expected['receipt']
            and exit_info['process_exit_sha256'] == expected['process']
            and exit_info['resource_sha256'] == expected['resources']
            and all(x['supervision_passed'] and x['child_reaped'] and x['child_exit_code'] == 0
                    and not x['timed_out'] and not x['sample_errors'] for x in (exit_info, process))):
        raise ValueError('independent verification did not close')

    fk_paths = dict(receipt=HOME/FK/'receipt.json', exit=HOME/(FK+'_supervision')/'exit.json',
                    process=HOME/(FK+'_supervision')/'process_exit.json',
                    outer=receipts/('ldm100k_generated_fk_'+FK+'_outer_exit.json'))
    if set(binding['fk_hashes']) != set(fk_paths):
        raise ValueError('incomplete generated FK closure binding')
    for key, path in fk_paths.items():
        if sha(owned(path, ROOT if key == 'outer' else HOME)) != binding['fk_hashes'][key]:
            raise ValueError('generated FK closure differs: ' + key)
    fk, fk_exit, fk_process, fk_outer = (read(fk_paths[k]) for k in ('receipt', 'exit', 'process', 'outer'))
    if not (fk['status'] == 'USER_LDM100K_GENERATED_ISAAC_FK_FINISHED'
            and fk['output_writers_closed'] is True and fk['exported_trials'] == 1880
            and fk['physics_steps'] == 0 and fk_outer['actual_returncode'] == 0
            and all(x['supervision_passed'] and x['child_reaped'] and x['child_exit_code'] == 0
                    and not x['timed_out'] and not x['sample_errors'] for x in (fk_exit, fk_process))):
        raise ValueError('generated FK did not close')
    manifest = owned(fk['private_manifest']['path'], HOME)
    if fk['private_manifest']['sha256'] != MANIFEST_SHA or sha(manifest) != MANIFEST_SHA:
        raise ValueError('generated reference manifest differs')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    from ppo5000_heldout_manifest_contract import validate
    validate(rows, binding['generation_seeds'])
    for row in rows:
        path = owned(row['file'], HOME)
        stat = path.stat()
        if stat.st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('generated reference bytes differ')
        after = path.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('generated reference changed while reading')
    return dict(actor_path=str(verify_paths['actor']), actor_sha256=ACTOR_SHA,
                training_binding=receipt['training_binding'], manifest_sha256=MANIFEST_SHA,
                rows=rows, source_kind='heldout-generated-reference',
                reference_count=len(rows), physics_success_not_yet_measured=True)
