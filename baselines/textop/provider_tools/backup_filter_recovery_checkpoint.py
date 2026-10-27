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
from session import SSH, GH
from filter_recovery_windows_paths import guard_local

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
RUN = 'filter_policy_paper69_resume10000_filter_recovery_20261027_v1'
REPOSITORY = 'Pendragon08/conditional-human-motion-generation'
TAG = 'textop-filter-recovery29d-20261027'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def recorded_client(script, evidence, timeout=240, command=None, env=None):
    """Retain emitted client streams in direct files, without reader threads.

    SSH/SCP/GitHub outcome is separate from the remote CPU supervisor. If a
    client cannot be reaped, its retained files may grow and are not final.
    """
    evidence = pathlib.Path(evidence)
    intent = evidence.with_name(evidence.stem + '_intent.json')
    stream_paths = {k:evidence.with_name(evidence.stem+'.'+k+'.txt') for k in ('stdout','stderr')}
    assert all(not os.path.lexists(str(p)) for p in [evidence,intent]+list(stream_paths.values()))
    command = SSH + ['python3 -'] if command is None else list(command)
    start = dict(command=command, timeout_seconds=timeout, stage='SSH_CLIENT_DISPATCH_INTENT',
                 script_sha256=hashlib.sha256(script.encode('utf-8')).hexdigest())
    with intent.open('x', encoding='utf-8') as f:
        json.dump(start, f, indent=2); f.flush(); os.fsync(f.fileno())
    proc = None
    record = dict(start, client_pid=None, actual_client_exit_code=None, client_reaped=False,
        timed_out=False, stdout='', stderr='', error=None, remote_child_outcome='UNKNOWN_UNTIL_SERVER_PROOF',
        direct_stream_files={k:str(v) for k,v in stream_paths.items()}, streams_final=False,output_may_grow=True)
    def snapshot():
        result={}
        for key,path in stream_paths.items():
            try: result[key]=path.read_text(encoding='utf-8',errors='replace')
            except BaseException as read_error:
                record.setdefault('stream_snapshot_errors',[]).append({'path':str(path),'error':repr(read_error)})
                if record['error'] is None:record['error']=repr(read_error)
                result[key]=''
        return result
    with stream_paths['stdout'].open('xb') as stdout_file, stream_paths['stderr'].open('xb') as stderr_file:
        try:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout_file, stderr=stderr_file,
                                    text=True, encoding='utf-8',env=env)
            record['client_pid'] = proc.pid
            proc.communicate(script, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            record['timed_out'] = True
            record['error'] = repr(error)
            record['first_timeout_streams'] = snapshot()
            proc.kill()  # Only our directly spawned client, never a remote training PID.
            try: proc.communicate(timeout=10)
            except BaseException as cleanup_error:
                record['cleanup_error'] = repr(cleanup_error)
                record['second_timeout_streams'] = snapshot()
        except BaseException:
            record['error'] = traceback.format_exc()
            if proc is not None and proc.poll() is None:
                proc.kill()
                try: proc.communicate(timeout=10)
                except BaseException as cleanup_error: record['cleanup_error'] = repr(cleanup_error)
        finally:
            if proc is not None:
                record['actual_client_exit_code'] = proc.poll()
                record['client_reaped'] = proc.returncode is not None
            record['streams_final'] = record['client_reaped'] or proc is None
            record['output_may_grow'] = not record['streams_final']
            for f in (stdout_file,stderr_file): f.flush(); os.fsync(f.fileno())
            try: record.update(snapshot())
            except BaseException as read_error:
                record['stream_snapshot_error']=repr(read_error)
                if record['error'] is None:record['error']=repr(read_error)
            with evidence.open('x', encoding='utf-8') as f:
                json.dump(record, f, indent=2); f.flush(); os.fsync(f.fileno())
    return record


def recorded_remote(script,evidence,timeout=240,command=None,env=None):
    record=recorded_client(script,evidence,timeout,command,env)
    assert record['actual_client_exit_code'] == 0 and record['client_reaped'] and not record['timed_out'] and record['error'] is None,str(evidence)
    return record['stdout']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--completed', type=int, required=True)
    ap.add_argument('--preflight-only', action='store_true')
    a = ap.parse_args()
    assert a.completed in range(15000, 40001, 5000)
    guard_local(D,D,directory=True)
    import msvcrt
    lock_path=D / 'formal_filter_checkpoint_backup.lock'
    guard_local(D,lock_path)
    lock = lock_path.open('a+b')
    if lock.seek(0, 2) == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    # Readiness does not create a backup directory or import Torch.
    spec_path = B / 'configs/filter-recovery-backup-spec.json'
    local_spec=json.loads(spec_path.read_text(encoding='utf-8'))
    for name,digest in local_spec['bound_local_files'].items():
        assert sha(B/name)==digest,name
    expected_spec = sha(spec_path)
    code = "import sys,json,pathlib; r=pathlib.Path(%r); sys.path.insert(0,str(r/'scripts')); from filter_recovery_backup_protocol import preflight,sha; p=r/'configs/filter-recovery-backup-spec.json'; assert sha(p)==%r; s=json.loads(p.read_text()); print(json.dumps(preflight(r,s,%d)))" % (ROOT,expected_spec,a.completed)
    observation=B/'results/filter_recovery_checkpoint_tools_20261027/observations'
    observation.mkdir(exist_ok=True)
    ready = json.loads(recorded_remote(code,observation/('preflight_'+str(time.time_ns())+'.json'),timeout=90))
    if a.preflight_only or not ready['ready']:
        print(json.dumps(ready)); return
    assert json.loads(spec_path.read_text(encoding='utf-8'))['backup_enabled'] is True
    folder = B / 'results/filter_recovery_checkpoints_20261027' / ('round_' + str(a.completed))
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / 'backup.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'completed': a.completed}

    def persist():
        state['updated_unix'] = time.time()
        state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
        mirror=D / ('filter_recovery_round{}_backup.json'.format(a.completed))
        guard_local(D,mirror)
        mirror.write_text(json.dumps(state, indent=2), encoding='utf-8')

    env = os.environ.copy()
    env.update(HTTPS_PROXY='http://127.0.0.1:7897', HTTP_PROXY='http://127.0.0.1:7897', GH_PROMPT_DISABLED='1')

    gh_sequence=[0]
    def gh(*args):
        gh_sequence[0]+=1
        record=recorded_client('',folder/('github_client_'+str(gh_sequence[0])+'.json'),timeout=900,command=[GH,*args],env=env)
        assert record['client_reaped'] and not record['timed_out'] and record['error'] is None
        return subprocess.CompletedProcess([GH,*args],record['actual_client_exit_code'],record['stdout'],record['stderr'])

    try:
        spec = json.loads((B/'configs/filter-recovery-backup-spec.json').read_text(encoding='utf-8'))
        expected_spec = sha(B/'configs/filter-recovery-backup-spec.json')
        script = r'''
import pathlib,os,json,hashlib,subprocess,sys
ROOT=pathlib.Path(ROOT_LITERAL);assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sys.path.insert(0,str(ROOT/'scripts'))
from filter_recovery_backup_protocol import preflight,sha
sp=ROOT/'configs/filter-recovery-backup-spec.json';assert sha(sp)==EXPECTED_SPEC
s=json.loads(sp.read_text());assert s['run']==RUN_LITERAL and s['backup_enabled']
pre=preflight(ROOT,s,COMPLETED);assert pre['ready'],pre
out=ROOT/'checkpoint_backups'/('round_'+str(COMPLETED))/s['audit_run'];audit=out/'audit.json';adapter=out/'adapter_proof.json'
record=ROOT/'receipts'/('filter_recovery_checkpoint_cpu_'+str(COMPLETED)+'_execution.json')
intent=record.with_name(record.stem+'_intent.json')
if not audit.exists():
 assert not out.exists() and not record.exists() and not intent.exists(),'Keep incomplete audit; inspect rather than retry'
 from filter_recovery_destination import guard_destination
 guard_destination(ROOT,out)
 from contact_process_supervision import run_supervised
 from contact_filter_resources import sample_resources
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 command=[str(ROOT/'env/bin/python'),str(ROOT/'scripts/inspect_filter_recovery_checkpoint.py'),'--completed',str(COMPLETED)]
 with intent.open('x') as f:
  json.dump({'status':'CPU_AUDIT_DISPATCH_INTENT','run':s['run'],'completed':COMPLETED,'command':command,'spec_sha256':EXPECTED_SPEC,'preflight':pre},f,indent=2);f.flush();os.fsync(f.fileno())
 execution=run_supervised(command,ROOT/'logs'/('filter_recovery_checkpoint_cpu_'+str(COMPLETED)+'.log'),ROOT/'logs'/('filter_recovery_checkpoint_cpu_'+str(COMPLETED)+'_resources.jsonl'),record,
  sampler=lambda pid:sample_resources(ROOT,pid,minimum_ram_bytes=1024**3,minimum_disk_bytes=3*1024**3),timeout_seconds=180,sample_interval=15,grace_seconds=10)
 assert execution['supervision_passed'],execution
else:execution=json.loads(record.read_text())
proof=json.loads(audit.read_text());ap=json.loads(adapter.read_text())
summary=json.loads((out/'relaxed_summary.json').read_text())
assert ap['relaxed_summary_sha256']==sha(out/'relaxed_summary.json')
assert summary['run']==s['run'] and summary['completed_learning_iterations']==COMPLETED
assert summary['source_count']==8278 and summary['checkpoint_sha256']==proof['full_checkpoint']['sha256']
assert summary['dataset_accepted'] is False and summary['strict_evaluation_completed'] is False
private=pathlib.Path(summary['private_decisions']['path']);assert private.parent==out and sha(private)==summary['private_decisions']['sha256']
assert execution['child_exit_code']==0 and execution['child_reaped'] and not execution['timed_out'] and execution['supervision_passed']
assert proof['run']==s['run'] and proof['completed_learning_iterations']==COMPLETED
assert proof['source_sha256']==s['bound_backup_files']['scripts/inspect_formal_filter_checkpoint_v2.py']
assert ap['spec_sha256']==EXPECTED_SPEC and ap['audit_sha256']==sha(audit)
assert ap['adapter_sha256']==s['bound_backup_files']['scripts/inspect_filter_recovery_checkpoint.py']
assert ap['protocol_sha256']==s['bound_backup_files']['scripts/filter_recovery_backup_protocol.py']
after=preflight(ROOT,s,COMPLETED);assert after['ready'] and after['metadata']==pre['metadata']
print(json.dumps({'audit':proof,'execution':execution,'adapter':ap,'new_run_provenance_verified':True,'preflight':after,'relaxed_summary':summary}))
'''.replace('ROOT_LITERAL',repr(ROOT)).replace('RUN_LITERAL',repr(RUN)).replace('EXPECTED_SPEC',repr(expected_spec)).replace('COMPLETED',str(a.completed))
        audited = json.loads(recorded_remote(script, folder / 'ssh_cpu_audit_execution.json', timeout=300))
        (folder / 'cpu_audit.json').write_text(json.dumps(audited, indent=2), encoding='utf-8')
        proof = audited['audit']
        (folder / 'relaxed_summary.json').write_text(json.dumps(audited['relaxed_summary'], indent=2), encoding='utf-8')
        proof['private_relaxed_statistics'] = audited['relaxed_summary']['private_decisions']
        state.update(status='CPU_AUDITED_DOWNLOAD_PENDING', audit=proof, adapter=audited['adapter'], run=RUN, data_deleted=False)
        persist()
        targets = {}
        for key, category in [('full_checkpoint', 'private'), ('public_inference', 'public'), ('private_relaxed_statistics', 'private_statistics')]:
            item = proof[key]
            dest = D / 'checkpoints' / category / RUN
            if category == 'private_statistics':
                dest = D / 'datasets/filter_recovery_statistics_private_20261027' / ('round_' + str(a.completed))
            guard_local(D,dest,directory=True)
            dest.mkdir(parents=True, exist_ok=True)
            guard_local(D,dest,directory=True)
            target = dest / pathlib.PurePosixPath(item['path']).name
            guard_local(D,target)
            if target.exists():
                assert target.stat().st_size == item['bytes'] and sha(target) == item['sha256']
            else:
                assert shutil.disk_usage(dest).free > item['bytes'] + spec['local_private_reserve_bytes']
                partial = target.with_name(target.name + '.partial')
                guard_local(D,partial,must_absent=True)
                args = ['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-i',
                        'C:/Users/Pende/.ssh/id_ed25519', 'ouyanghao@10.147.17.32:' + item['path'], str(partial)]
                recorded_remote('',folder/(category+'_transfer.json'),timeout=900,command=args)
                guard_local(D,partial)
                assert partial.stat().st_size == item['bytes'] and sha(partial) == item['sha256']
                guard_local(D,target,must_absent=True)
                partial.replace(target)
                guard_local(D,target)
            targets[key] = str(target)
        state.update(status='PRIVATE_FULL_AND_PUBLIC_INFERENCE_LOCAL_VERIFIED', local_paths=targets)
        persist()
        response = gh('api', 'repos/' + REPOSITORY + '/releases/tags/' + TAG)
        if response.returncode:
            assert '404' in response.stderr, response.stderr
            notes = folder / 'release_notes.md'
            notes.write_text('TextOp 29DoF privileged filtering-policy inference snapshots.\n\n'
                'Continued from the verified 10,000-round checkpoint on 8,278 sources. The next review is at 15,000 rounds; the 40,000-round paper reference is retained. These are intermediate checkpoints, not completed paper or closed-loop results. The 557D privileged actor/critic uses a declared validated static solid-floor contact representation; exact author geometry is unavailable.\n\n'
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
        final = json.loads(recorded_remote('''import pathlib,hashlib,json
p=pathlib.Path(%r)
print(json.dumps({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
''' % proof['full_checkpoint']['path'], folder/'ssh_final_source_execution.json',timeout=45))
        assert final['sha256'] == proof['full_checkpoint']['sha256'] and final['bytes'] == proof['full_checkpoint']['bytes']
        state.update(status='FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED',
                     github_release=release['html_url'], github_asset=asset['browser_download_url'],
                     github_digest=asset['digest'], github_asset_id=asset['id'],
                     original_server_checkpoint_retained=True, full_checkpoint_publicly_uploaded=False,
                     private_full_two_location_sha_verified=True, inference_three_location_sha_verified=True,
                     private_relaxed_statistics_two_location_sha_verified=True,
                     relaxed_counts=audited['relaxed_summary']['counts'], strict_evaluation_completed=False,
                     dataset_accepted=False, convergence_claimed=False, full_generator_started=False)
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
