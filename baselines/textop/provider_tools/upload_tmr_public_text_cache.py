"""Bound, signed, once-only upload of already verified public text-model files.

Never redownloads or changes the original failed direct-HF job. Existing upload
intents or partial files reject dispatch; reconciliation precedes any new run.
"""
import argparse
import hashlib
import json
import msvcrt
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback
from backup_filter_recovery_checkpoint import recorded_remote
from filter_recovery_windows_paths import guard_local

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
O = B / 'results/semantic_evaluator_20261029/public_text_upload_v1'
p = argparse.ArgumentParser()
p.add_argument('--worker', action='store_true')
a = p.parse_args()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def birth(process):
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k.GetProcessTimes.restype = wintypes.BOOL
    ts = [wintypes.FILETIME() for _ in range(4)]
    assert k.GetProcessTimes(wintypes.HANDLE(int(process._handle)), *[ctypes.byref(x) for x in ts])
    return (ts[0].dwHighDateTime << 32) | ts[0].dwLowDateTime


if not a.worker:
    O.mkdir(exist_ok=False)
    cache_root = B / 'results/semantic_evaluator_20261029/public_text_cache_v1'
    cache = json.loads((cache_root / 'receipt.json').read_bytes())
    assert cache['status'] == 'PUBLIC_TMR_TEXT_CACHE_D_BYTES_VERIFIED' and cache['output_writers_closed']
    assert json.loads((cache_root / 'exit.json').read_bytes())['actual_returncode'] == 0
    assert sum(x['bytes'] for x in cache['completed_files']) == 707323471 and len(cache['completed_files']) == 11
    # Cache's foreground exec client was actually reaped before invoking this uploader.
    write(O / 'source_binding.json', dict(cache_receipt_sha256=digest(cache_root / 'receipt.json'),
        cache_exit_sha256=digest(cache_root / 'exit.json'), files=cache['completed_files'],
        source_sha256=digest(pathlib.Path(__file__)), public_weights_only=True))
    cmd = [sys.executable, '-B', str(pathlib.Path(__file__).resolve()), '--worker']
    with (O / 'worker.log').open('xb') as log:
        q = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    dispatch = dict(status='PUBLIC_TMR_UPLOAD_DISPATCHED_NOT_ACCEPTED', worker_pid=q.pid,
        worker_start_filetime=birth(q), command=cmd, timeout_seconds=21600)
    write(O / 'dispatch.json', dispatch)
    print(json.dumps(dispatch))
    raise SystemExit(0)

lock = (O / 'upload.lock').open('a+b')
lock.seek(0)
msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
client = None
try:
    bound = json.loads((O / 'source_binding.json').read_bytes())
    assert digest(pathlib.Path(__file__)) == bound['source_sha256']
    rows = []
    for item in bound['files']:
        path = pathlib.Path(item['path'])
        owner = guard_local(D, path)
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        relative = path.relative_to(D / 'models/evaluation_text_tmr_20261029').as_posix()
        assert relative.split('/')[0] in ['text_model_0', 'text_model_1'] and relative.count('/') == 1
        st = path.stat()
        rows.append(dict(relative=relative, path=str(path), bytes=item['bytes'], sha256=item['sha256'],
            source_fingerprint=dict(size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns), owner=owner))
    assert shutil.disk_usage(D).free > 2 * 1024**3
    files = [{k: x[k] for k in ['relative', 'bytes', 'sha256']} for x in rows]
    init = r'''
import json,pathlib,os,sys,shutil,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import owned,read,sha
parent=owned(r/'evaluator_tmr_20261029');out=parent/'text_models_import_v1';assert not out.exists()
old=read(r/'receipts/tmr_text_models_tmr_text_models_v1_outer_exit.json');assert old['actual_returncode']==1
files=FILES_LITERAL;total=sum(x['bytes'] for x in files);assert total==707323471 and len(files)==11
assert shutil.disk_usage(parent).free>2*1024**3+128*1024**2+total+16*1024**2
out.mkdir(mode=0o700)
for name in ['text_model_0','text_model_1']:(out/name).mkdir(mode=0o700)
spec=dict(run='text_models_import_v1',files=files,total_bytes=total,source_binding_sha256=SOURCE_SHA_LITERAL,output_writers_closed=False)
with (out/'import_intent.json').open('x') as f:json.dump(spec,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='NEW_PUBLIC_TMR_IMPORT_DIRECTORY_PREPARED',path=str(out),intent_sha256=sha(out/'import_intent.json'),data_free_bytes=shutil.disk_usage(parent).free,
 available_host_ram_bytes=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024,
 gpu1_snapshot=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free,memory.used','--format=csv,noheader,nounits'],text=True).strip())))
'''.replace('FILES_LITERAL', repr(files)).replace('SOURCE_SHA_LITERAL', repr(digest(O / 'source_binding.json')))
    before = json.loads(recorded_remote(init, O / 'before_client.json', timeout=180))
    write(O / 'before.json', before)
    remote_root = before['path']
    commands = []
    for item in rows:
        commands.append('put "' + item['path'].replace('\\', '/') + '" "' + remote_root + '/' + item['relative'] + '.partial"')
    cmd = ['sftp', '-R', '8', '-B', '32768', '-b', '-', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
        '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=12', '-i', 'C:/Users/Pende/.ssh/id_ed25519', 'ouyanghao@10.147.17.32']
    sample_code = r'''
import os,pathlib,json,sys,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));from ldm100k_verified_input import owned
out=owned(r/'evaluator_tmr_20261029/text_models_import_v1');rows=[]
for rel in RELATIVE_LITERAL:
 p=out/(rel+'.partial');rows.append(dict(relative=rel,bytes=p.stat().st_size if p.exists() else 0))
free=shutil.disk_usage(out).free;assert free>2*1024**3+128*1024**2
print(json.dumps(dict(files=rows,data_free_bytes=free,total_written_bytes=sum(x['bytes'] for x in rows))))
'''.replace('RELATIVE_LITERAL', repr([x['relative'] for x in rows]))
    with (O / 'upload.stdout.txt').open('xb') as stdout, (O / 'upload.stderr.txt').open('xb') as stderr, (O / 'resources.jsonl').open('x') as resources:
        client = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        write(O / 'client_intent.json', dict(command=cmd, pid=client.pid, start_filetime=birth(client), started_unix=time.time(), timeout_seconds=21600))
        client.stdin.write(('\n'.join(commands) + '\n').encode('utf-8'))
        client.stdin.close()
        started = time.monotonic()
        seq = 0
        while client.poll() is None:
            sample = json.loads(recorded_remote(sample_code, O / ('s' + str(seq) + '.json'), timeout=90))
            seq += 1
            assert sample['total_written_bytes'] <= 707323471 and shutil.disk_usage(D).free > 2 * 1024**3
            resources.write(json.dumps(dict(unix=time.time(), remote=sample, client_pid=client.pid, D_free_bytes=shutil.disk_usage(D).free)) + '\n')
            resources.flush()
            assert time.monotonic() - started < 21600
            try:
                client.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        write(O / 'client_outcome.json', dict(actual_client_exit_code=client.returncode, client_reaped=True, streams_final=True))
        assert client.returncode == 0
    for item in rows:
        p = pathlib.Path(item['path'])
        st = p.stat()
        assert item['source_fingerprint'] == dict(size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)
        assert digest(p) == item['sha256']
    finalize = r'''
import json,pathlib,os,sys,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import owned,sha,read
out=owned(r/'evaluator_tmr_20261029/text_models_import_v1');intent=read(out/'import_intent.json');done=[]
for item in intent['files']:
 p=owned(out/(item['relative']+'.partial'));assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 with p.open('r+b') as f:f.flush();os.fsync(f.fileno())
 final=out/item['relative'];assert not final.exists();p.rename(final)
 assert owned(final).stat().st_size==item['bytes'] and sha(final)==item['sha256']
 st=final.stat();done.append(dict(path=str(final),relative=item['relative'],bytes=st.st_size,sha256=sha(final),fingerprint=dict(device=st.st_dev,inode=st.st_ino,size=st.st_size,mtime_ns=st.st_mtime_ns,uid=st.st_uid)))
assert shutil.disk_usage(out).free>2*1024**3+128*1024**2
proof=dict(status='PUBLIC_TMR_TEXT_FILES_SERVER_BYTES_VERIFIED',files=done,total_bytes=sum(x['bytes'] for x in done),source_binding_sha256=intent['source_binding_sha256'],
 model_payload_loads=0,optimizer_updates=0,output_writers_closed=True,data_free_bytes=shutil.disk_usage(out).free)
with (out/'receipt.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(proof,receipt_sha256=sha(out/'receipt.json'),receipt_path=str(out/'receipt.json'))))
'''
    result = json.loads(recorded_remote(finalize, O / 'after_client.json', timeout=240))
    write(O / 'receipt.json', result)
    write(O / 'worker_exit.json', dict(actual_returncode=0, status='PUBLIC_TMR_UPLOAD_BYTE_VERIFIED', client_reaped=True, streams_final=True))
except BaseException as error:
    if client is not None and client.poll() is None:
        client.kill()
        client.wait(timeout=15)
    write(O / 'worker_exit.json', dict(actual_returncode=1, status='FAILED_PUBLIC_TMR_UPLOAD_ALL_PARTIALS_RETAINED',
        error=repr(error), traceback=traceback.format_exc(), client_reaped=client is None or client.poll() is not None))
    raise
finally:
    lock.close()
