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
RUN = 'filter_policy_paper69_solid_floor_20261026_v1'
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
    assert a.completed in [1000] + list(range(5000, 40001, 5000))
    import msvcrt
    lock = (D / 'formal_filter_checkpoint_backup.lock').open('a+b')
    if lock.seek(0, 2) == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    folder = B / 'results/formal_filter_checkpoints_20261026' / ('round_' + str(a.completed))
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / 'backup.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'completed': a.completed}

    def persist():
        state['updated_unix'] = time.time()
        state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
        (D / ('formal_filter_round{}_backup.json'.format(a.completed))).write_text(json.dumps(state, indent=2), encoding='utf-8')

    env = os.environ.copy()
    env.update(HTTPS_PROXY='http://127.0.0.1:7897', HTTP_PROXY='http://127.0.0.1:7897', GH_PROMPT_DISABLED='1')

    def gh(*args):
        return subprocess.run([GH, *args], env=env, capture_output=True, text=True,
                              encoding='utf-8', timeout=900)

    try:
        source = B / 'scripts/inspect_formal_filter_checkpoint_v2.py'
        expected = sha(source)
        bound = json.loads((B / 'configs/formal-contact-activation.json').read_text(encoding='utf-8'))['deployed_files']
        assert 'scripts/inspect_formal_filter_checkpoint_v2.py' not in bound
        script = r'''
import pathlib,os,json,hashlib,base64,subprocess,time
ROOT=pathlib.Path(ROOT_LITERAL);bound=BOUND;assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/n)==s for n,s in bound.items())
script=ROOT/'scripts/inspect_formal_filter_checkpoint_v2.py';data=base64.b64decode(SOURCE)
assert hashlib.sha256(data).hexdigest()==EXPECTED
if script.exists():assert sha(script)==EXPECTED
else:
 with script.open('xb') as f:f.write(data)
if COMPLETED==40000:
 current=json.loads((ROOT/'tracker_validation/filter_policy_paper69_solid_floor_20261026_v1/receipt.json').read_text())
 stopped=json.loads((ROOT/'receipts/filter_policy_filter_policy_paper69_solid_floor_20261026_v1_exit.json').read_text())
 assert current['status']=='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED' and current['output_writers_closed'] is True
 assert stopped['child_exit_code']==0 and stopped['timed_out'] is False
out=ROOT/'checkpoint_backups'/('round_'+str(COMPLETED))/'author_v2';audit=out/'audit.json'
if not audit.exists():
 assert not out.exists(),'Failed/incomplete prior audit preserved; inspect before retry'
 result=subprocess.run([str(ROOT/'env/bin/python'),str(script),'--completed',str(COMPLETED)],capture_output=True,text=True,timeout=180,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 execution={'child_exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'sdk_runs':0}
 record=ROOT/'receipts'/('formal_checkpoint_cpu_'+str(COMPLETED)+'_author_v2_execution.json')
 assert not record.exists();record.write_text(json.dumps(execution,indent=2))
 assert result.returncode==0,(result.stderr,result.stdout)
else:execution=json.loads((ROOT/'receipts'/('formal_checkpoint_cpu_'+str(COMPLETED)+'_author_v2_execution.json')).read_text())
proof=json.loads(audit.read_text());assert proof['source_sha256']==EXPECTED and execution['child_exit_code']==0
assert all(sha(ROOT/n)==s for n,s in bound.items())
print(json.dumps({'audit':proof,'execution':execution,'frozen_runtime_files_verified':len(bound)}))
'''.replace('ROOT_LITERAL', repr(ROOT)).replace('BOUND', repr(bound)).replace('SOURCE', repr(base64.b64encode(source.read_bytes()).decode())).replace('EXPECTED', repr(expected)).replace('COMPLETED', str(a.completed))
        audited = json.loads(remote(script, timeout=210))
        (folder / 'cpu_audit.json').write_text(json.dumps(audited, indent=2), encoding='utf-8')
        proof = audited['audit']
        state.update(status='CPU_AUDITED_DOWNLOAD_PENDING', audit=proof, data_deleted=False)
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
