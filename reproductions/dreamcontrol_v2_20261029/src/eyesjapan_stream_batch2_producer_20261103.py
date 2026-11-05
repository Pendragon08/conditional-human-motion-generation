"""Stream selected archive members without creating another archive copy."""
import hashlib, json, subprocess, sys, tarfile
from pathlib import Path
from bmlmovi_stream_protocol_20261102 import write_frame,end_stream,MAX_PAYLOAD

ROOT=Path(__file__).resolve().parent.parent
ARCHIVE=Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives/EyesJapanDataset_smplx_g.tar.bz2')
REMOTE='/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029'
PY='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'
def main():
    mode=sys.argv[1]
    if mode != 'batch2': raise ValueError('unknown mode')
    queue=ROOT/'data'/('eyesjapan_stream_'+mode+'_queue_20261103.json')
    q=json.loads(queue.read_text(encoding='utf8')); expected={r['member']:r for r in q['members']};seen=set()
    if q['training_ready'] is not False or len(expected)!=len(q['members']): raise ValueError('invalid queue')
    before=ARCHIVE.stat();out=ROOT/'reports'/('eyesjapan_stream_'+mode+'_producer_20261103.jsonl')
    remote_queue=REMOTE+'/eyesjapan_stream_'+mode+'_queue_20261103.json'
    remote_out=REMOTE+'/eyesjapan_stream_'+mode+'_20261103'
    command=f'cd {REMOTE} && nice -n 15 {PY} bmlmovi_stream_consumer_20261102.py {remote_queue} {remote_out}'
    with out.open('x',encoding='utf8',buffering=1) as log, (ROOT/'reports'/('eyesjapan_stream_'+mode+'_ssh_20261103.log')).open('xb') as err:
      proc=subprocess.Popen(['ssh','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','10.147.17.32',command],stdin=subprocess.PIPE,stdout=err,stderr=err)
      try:
        with tarfile.open(ARCHIVE,'r|bz2') as tar:
          for item in tar:
            row=expected.get(item.name)
            if row is None: continue
            if item.name in seen or not item.isfile() or item.size!=row['member_size'] or item.size>MAX_PAYLOAD: raise ValueError('unexpected member')
            f=tar.extractfile(item)
            if f is None: raise ValueError('cannot read member')
            payload=f.read(MAX_PAYLOAD+1);digest=hashlib.sha256(payload).hexdigest()
            if len(payload)!=item.size or digest!=row['source_sha256']: raise ValueError('source member changed')
            write_frame(proc.stdin,{'member':item.name,'sha256':digest},payload);seen.add(item.name)
            log.write(json.dumps({'event':'sent_not_completed','member':item.name,'sha256':digest})+'\n')
            if len(seen)==len(expected):break
        if seen!=set(expected):raise ValueError('missing members')
        after=ARCHIVE.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('source archive changed')
        end_stream(proc.stdin);proc.stdin.close();code=proc.wait()
        log.write(json.dumps({'event':'ssh_exit','code':code,'sent':len(seen)})+'\n')
        if code:raise RuntimeError('remote consumer failed')
      except BaseException:
        proc.stdin.close();proc.wait();raise
if __name__=='__main__':main()
