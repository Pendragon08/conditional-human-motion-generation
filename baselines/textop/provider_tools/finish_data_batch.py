"""Collect this already-running data batch, then archive private outputs locally.

No training launch, process signals, deletion, or publication of licensed assets.
Exits after the one explicit batch finishes (or after six hours).
"""
import datetime, hashlib, json, os, pathlib, subprocess, sys, time
from session import SSH, remote

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
LOCAL = pathlib.Path('D:/具身智能/TextOp复现_20261024')
SERVER = '/data/code/ouyanghao/oyh/textop-reproduction-20261024/data_preparation_20261025'

def run(name):
    result = subprocess.run([sys.executable, str(HERE/name)], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(name + ': ' + result.stderr[-2000:])
    print(result.stdout[-1200:], flush=True)

def archive_private(batch,expected):
    target = LOCAL/'datasets'/('paper29_'+batch+'_20261025.private.tar.gz')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return {'path': str(target), 'status': 'existing_archive_preserved'}
    partial = target.with_suffix(target.suffix+'.partial')
    if partial.exists():
        raise FileExistsError('Previous partial archive preserved for inspection')
    # Binary pipe avoids PowerShell transcoding and creates no server tar copy.
    command = ['tar', '-C', SERVER, '-czf', '-', batch,
               'audit_summary.json', 'available_raw_pairs.private.json',
               'train_sequence_manifest.private.jsonl', 'val_sequence_manifest.private.jsonl']
    import shlex
    with partial.open('xb') as output:
        result = subprocess.run([*SSH, ' '.join(shlex.quote(x) for x in command)], stdout=output, stderr=subprocess.PIPE, timeout=3600)
    if result.returncode:
        raise RuntimeError('Private archive transfer failed: '+result.stderr.decode(errors='replace')[-1000:])
    import tarfile
    with tarfile.open(partial) as archive:
        members=archive.getnames()
        assert batch+'/progress.json' in members
        assert sum(name.endswith('.private.npz') for name in members)==expected
        # Read every member to verify compression integrity, retaining no extra copy.
        for member in archive:
            if member.isfile():
                with archive.extractfile(member) as source:
                    while source.read(8*1024**2): pass
    digest=hashlib.sha256()
    with partial.open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):digest.update(block)
    partial.rename(target)
    receipt={'path':str(target),'bytes':target.stat().st_size,'sha256':digest.hexdigest(),
             'motion_files':expected,'license':'Private local backup; never publish this archive to GitHub',
             'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (LOCAL/(batch+'_private_data_backup_receipt.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    public={k:v for k,v in receipt.items() if k!='path'}
    (BASE/('results/data_preparation_20261025/local_backup_'+batch+'.json')).write_text(json.dumps(public,indent=2),encoding='utf-8')
    return receipt

def main():
    last_published=-1
    for _ in range(360):
        try:
            run('collect_data_preparation.py')
            aggregate=json.loads((BASE/'results/data_preparation_20261025/latest.json').read_text(encoding='utf-8'))
            progress=aggregate['receipts'].get('batch500',{})
            processed=progress.get('processed',0)
            finished=progress.get('status')!='RUNNING' and processed>=progress.get('total_requested',500)
            extra=aggregate['receipts'].get('batch143',{})
            local=aggregate['receipts'].get('local_raw_expansion',{})
            extra_done=extra.get('status')!='RUNNING' and extra.get('processed',0)>=143
            local_done=local.get('status') in ['LOCAL_EXTRACTION_COMPLETE','LOCAL_EXTRACTION_PARTIAL_UNMATCHED','STOPPED_WITH_ERROR']
            if finished:
                if progress.get('completed_sequences')==500 and progress.get('failed_sequences')==0:
                    print(json.dumps(archive_private('available500_v1',500),ensure_ascii=False),flush=True)
            if extra_done:
                if extra.get('completed_sequences')==143 and extra.get('failed_sequences')==0:
                    print(json.dumps(archive_private('available143_v1',143),ensure_ascii=False),flush=True)
            if finished and extra_done and local_done:
                run('collect.py')
                run('publish.py')
                print('DATA_BATCH_COLLECTED_FILTER_STILL_PENDING',flush=True)
                return
            milestone=(processed+extra.get('processed',0)+local.get('completed',0))//100
            if milestone>last_published:
                run('publish.py');last_published=milestone
        except Exception as error:
            print(repr(error),flush=True)
        time.sleep(60)
    raise TimeoutError('Six-hour collection limit reached; no training or data deletion performed')

if __name__=='__main__': main()
