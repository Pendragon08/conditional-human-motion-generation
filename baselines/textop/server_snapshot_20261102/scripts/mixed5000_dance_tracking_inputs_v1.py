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

    fk_run='user_mixed5000_dance_fk_library_v1';fkout=base/fk_run;fklog=base/(fk_run+'_supervision')
    fk_paths=dict(request=receipts/(fk_run+'_request.json'),driver=receipts/(fk_run+'_driver.json'),outer=receipts/(fk_run+'_outer_exit.json'),receipt=fkout/'receipt.json',exit=fklog/'exit.json',process=fklog/'process_exit.json',resources=fklog/'resources.jsonl')
    expected_fk=binding['fk_closure_hashes']
    if set(fk_paths)!=set(expected_fk):raise ValueError('incomplete FK closure')
    for key,path in fk_paths.items():
        if sha(owned(path,ROOT))!=expected_fk[key]:raise ValueError('FK closure changed: '+key)
    fk=read(fk_paths['receipt']);outer=read(fk_paths['outer']);ex=read(fk_paths['exit']);pr=read(fk_paths['process'])
    if not(fk['status']=='PPO5000_DANCE_VAL_REFERENCE_FK_FINISHED' and fk['output_writers_closed'] and fk['exported_clips']==16 and fk['exported_frames']==15517 and outer['actual_returncode']==0 and outer['child_reaped'] and ex['request_sha256']==expected_fk['request'] and ex['receipt_sha256']==expected_fk['receipt'] and ex['process_exit_sha256']==expected_fk['process'] and ex['resource_sha256']==expected_fk['resources'] and all(x['supervision_passed'] and x['child_reaped'] and x['child_exit_code']==0 and not x['timed_out'] and not x['sample_errors'] for x in (ex,pr))):raise ValueError('FK closure failed')
    manifest=owned(fkout/'references.private.jsonl',ROOT)
    if sha(manifest)!=binding['fk_manifest_sha256'] or fk['private_manifest']['sha256']!=binding['fk_manifest_sha256']:raise ValueError('dance manifest changed')
    original=[json.loads(line) for line in manifest.read_text().splitlines()];files=binding['audited_files']
    if len(original)!=16 or len(files)!=16:raise ValueError('wrong reference count')
    rows=[]
    for x,audit in zip(original,files):
        path=owned(x['file'],ROOT)
        if path.parent!=fkout or str(path)!=audit['file'] or path.stat().st_size!=audit['bytes'] or sha(path)!=audit['sha256'] or x['sha256']!=audit['sha256'] or x['frames']!=audit['frames']:raise ValueError('dance reference differs')
        identity='|'.join([x['source_family'],str(x['annotation_index']),str(x['first_pose_index']),str(x['last_pose_index_inclusive'])])
        rows.append(dict(file=str(path),frames=x['frames'],segment_id=hashlib.sha256(identity.encode()).hexdigest(),seed=20261103,mode='original_val_dance',reference_fps=50,source_sha256=x['source_sha256'],first_pose_index=x['first_pose_index'],last_pose_index_inclusive=x['last_pose_index_inclusive']))
    if len({x['segment_id'] for x in rows})!=16:raise ValueError('duplicate dance source window')
    return dict(actor_path=str(verify_paths['actor']),actor_sha256=ACTOR_SHA,training_binding=receipt['training_binding'],manifest_sha256=binding['fk_manifest_sha256'],rows=rows,source_kind='original-heldout-dance-reference',reference_count=16,physics_success_not_yet_measured=True)
