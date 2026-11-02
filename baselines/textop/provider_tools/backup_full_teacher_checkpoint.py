"""Back up a stable formal teacher checkpoint without deleting any version.

The complete checkpoint stays private on server and D:. GitHub gets a separate
inference-only export without per-motion sampling arrays or optimizer state.
"""
import argparse
import base64
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
import traceback
from session import SSH, GH, remote

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
RUN = 'filter_policy_paper69_resume5000_to40000_20261026_v1'
REPOSITORY = 'Pendragon08/conditional-human-motion-generation'
TAG = 'textop-filter-policy29d-20261026'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--completed', type=int, required=True)
    a = ap.parse_args()
    assert a.completed in range(10000, 40001, 5000)
    import msvcrt
    lock = (D / 'formal_filter_checkpoint_backup.lock').open('a+b')
    if lock.seek(0, 2) == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    folder = B / 'results/full_teacher_checkpoints_20261026' / ('round_' + str(a.completed))
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / 'backup.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'completed': a.completed}

    def persist():
        state['updated_unix'] = time.time()
        state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
        (D / ('full_teacher_round{}_backup.json'.format(a.completed))).write_text(json.dumps(state, indent=2), encoding='utf-8')

    env = os.environ.copy()
    env.update(HTTPS_PROXY='http://127.0.0.1:7897', HTTP_PROXY='http://127.0.0.1:7897', GH_PROMPT_DISABLED='1')

    def gh(*args):
        return subprocess.run([GH, *args], env=env, capture_output=True, text=True,
                              encoding='utf-8', timeout=900)

    try:
        spec = json.loads((B/'configs/full-teacher-backup-spec.json').read_text(encoding='utf-8'))
        expected_spec = sha(B/'configs/full-teacher-backup-spec.json')
        script = r'''
import pathlib,os,json,hashlib,subprocess,sys
ROOT=pathlib.Path(ROOT_LITERAL);assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sys.path.insert(0,str(ROOT/'scripts'))
from full_teacher_backup_protocol import preflight,sha
sp=ROOT/'configs/full-teacher-backup-spec.json';assert sha(sp)==EXPECTED_SPEC
s=json.loads(sp.read_text());assert s['run']==RUN_LITERAL and s['backup_enabled']
pre=preflight(ROOT,s,COMPLETED);assert pre['ready'],pre
out=ROOT/'checkpoint_backups'/('round_'+str(COMPLETED))/s['audit_run'];audit=out/'audit.json';adapter=out/'adapter_proof.json'
record=ROOT/'receipts'/('full_teacher_checkpoint_cpu_'+str(COMPLETED)+'_execution.json')
if not audit.exists():
 assert not out.exists() and not record.exists(),'Keep incomplete audit; inspect rather than retry'
 result=subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'scripts/inspect_full_teacher_checkpoint.py'),'--completed',str(COMPLETED)],capture_output=True,text=True,timeout=180,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 execution={'child_exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'sdk_runs':0}
 with record.open('x') as f:json.dump(execution,f,indent=2)
 assert result.returncode==0,(result.stderr,result.stdout)
else:execution=json.loads(record.read_text())
proof=json.loads(audit.read_text());ap=json.loads(adapter.read_text())
assert execution['child_exit_code']==0 and proof['run']==s['run'] and proof['completed_learning_iterations']==COMPLETED
assert proof['source_sha256']==s['bound_backup_files']['scripts/inspect_formal_filter_checkpoint_v2.py']
assert ap['spec_sha256']==EXPECTED_SPEC and ap['audit_sha256']==sha(audit)
assert ap['adapter_sha256']==s['bound_backup_files']['scripts/inspect_full_teacher_checkpoint.py']
assert ap['protocol_sha256']==s['bound_backup_files']['scripts/full_teacher_backup_protocol.py']
after=preflight(ROOT,s,COMPLETED);assert after['ready'] and after['metadata']==pre['metadata']
print(json.dumps({'audit':proof,'execution':execution,'adapter':ap,'new_run_provenance_verified':True,'preflight':after}))
'''.replace('ROOT_LITERAL',repr(ROOT)).replace('RUN_LITERAL',repr(RUN)).replace('EXPECTED_SPEC',repr(expected_spec)).replace('COMPLETED',str(a.completed))
        audited = json.loads(remote(script, timeout=210))
        (folder / 'cpu_audit.json').write_text(json.dumps(audited, indent=2), encoding='utf-8')
        proof = audited['audit']
        state.update(status='CPU_AUDITED_DOWNLOAD_PENDING', audit=proof, adapter=audited['adapter'], run=RUN, data_deleted=False)
        persist()
        targets = {}
        for key, category in [('full_checkpoint', 'private'), ('public_inference', 'public')]:
            item = proof[key]
            dest = D / 'checkpoints' / category / RUN
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / pathlib.PurePosixPath(item['path']).name
            if target.exists():
                assert target.stat().st_size == item['bytes'] and sha(target) == item['sha256']
            else:
                assert shutil.disk_usage(dest).free > item['bytes'] + 1024**3
                partial = target.with_name(target.name + '.partial')
                assert not partial.exists(), 'Retain interrupted transfer for inspection'
                args = ['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-i',
                        'C:/Users/Pende/.ssh/id_ed25519', 'ouyanghao@10.147.17.32:' + item['path'], str(partial)]
                transfer = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', timeout=900)
                (folder / (category + '_transfer.json')).write_text(json.dumps({'exit_code': transfer.returncode,
                    'stdout': transfer.stdout, 'stderr': transfer.stderr}, indent=2), encoding='utf-8')
                assert transfer.returncode == 0, transfer.stderr
                assert partial.stat().st_size == item['bytes'] and sha(partial) == item['sha256']
                partial.replace(target)
            targets[key] = str(target)
        state.update(status='PRIVATE_FULL_AND_PUBLIC_INFERENCE_LOCAL_VERIFIED', local_paths=targets)
        persist()
        response = gh('api', 'repos/' + REPOSITORY + '/releases/tags/' + TAG)
        if response.returncode:
            assert '404' in response.stderr, response.stderr
            notes = folder / 'release_notes.md'
            notes.write_text('TextOp 29DoF privileged filtering-policy inference snapshots.\n\n'
                'Training target: 40,000 iterations on 8,278 available motion sources. These are intermediate checkpoints, not completed paper or closed-loop results. The 557D privileged actor/critic uses a declared validated static solid-floor contact representation; exact author geometry is unavailable.\n\n'
                'Public exports contain network and observation-normalizer tensors plus aggregate iteration metadata. They omit per-motion sampling statistics, optimizer state, raw motion arrays and text/source indexes. Full training-resume checkpoints remain privately backed up on the server and local D: drive. No server checkpoint is removed.\n', encoding='utf-8')
            created = gh('release', 'create', TAG, '--repo', REPOSITORY, '--target', 'codex/textop-reproduction-20261024',
                         '--title', 'TextOp 29DoF privileged filter policy (training in progress)', '--prerelease',
                         '--notes-file', str(notes))
            assert created.returncode == 0, created.stderr
            response = gh('api', 'repos/' + REPOSITORY + '/releases/tags/' + TAG)
        assert response.returncode == 0, response.stderr
        release = json.loads(response.stdout)
        public = pathlib.Path(targets['public_inference'])
        matches = [x for x in release['assets'] if x['name'] == public.name]
        if not matches:
            uploaded = gh('release', 'upload', TAG, str(public), '--repo', REPOSITORY)
            assert uploaded.returncode == 0, uploaded.stderr
        response = gh('api', 'repos/' + REPOSITORY + '/releases/tags/' + TAG)
        assert response.returncode == 0, response.stderr
        asset = next(x for x in json.loads(response.stdout)['assets'] if x['name'] == public.name)
        assert asset['size'] == proof['public_inference']['bytes']
        assert asset['digest'] == 'sha256:' + proof['public_inference']['sha256']
        # A final server hash check after transfers/publication proves the source still exists unchanged.
        final = json.loads(remote('''import pathlib,hashlib,json
p=pathlib.Path(%r)
print(json.dumps({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
''' % proof['full_checkpoint']['path'], timeout=45))
        assert final['sha256'] == proof['full_checkpoint']['sha256'] and final['bytes'] == proof['full_checkpoint']['bytes']
        state.update(status='FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED',
                     github_release=release['html_url'], github_asset=asset['browser_download_url'],
                     github_digest=asset['digest'], github_asset_id=asset['id'],
                     original_server_checkpoint_retained=True, full_checkpoint_publicly_uploaded=False,
                     private_full_two_location_sha_verified=True, inference_three_location_sha_verified=True,
                     convergence_claimed=False, full_generator_started=False)
        state.pop('error', None)
        persist()
        print(json.dumps({k: state[k] for k in ['status', 'completed', 'github_release',
              'private_full_two_location_sha_verified', 'inference_three_location_sha_verified', 'data_deleted']}))
    except Exception:
        state['error'] = traceback.format_exc()
        persist()
        raise


if __name__ == '__main__':
    main()
