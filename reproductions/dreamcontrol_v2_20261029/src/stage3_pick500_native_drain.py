"""Drain verified generated candidates in native batches while producers continue.

Only data admission is performed. Stops before final500-library verification and
formal training, or reports a measured deficit for new-seed replacement sampling.
"""
import json
import os
from pathlib import Path
import subprocess
import time

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
W=O/'native_drain_controller_v1';W.mkdir(exist_ok=False)
(W/'launch.json').open('x').write(json.dumps({'pid':os.getpid(),'scope':'bounded reference admission, no policy updates','target':500}))
active=O/'active_native_controller.json'
if active.exists():
    old=json.loads(active.read_text());p=Path('/proc')/str(old['pid'])/'cmdline'
    if p.exists() and 'stage3_pick500_native_drain.py' in p.read_bytes().decode(errors='replace'):raise RuntimeError('native drain already active')
active.write_text(json.dumps({'pid':os.getpid(),'directory':str(W),'source':str(Path(__file__).resolve())},indent=2))
began=time.monotonic();runs=0


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines(keepends=True) if x.endswith('\n')] if p.exists() else []


def save(value):
    value.update(pid=os.getpid(),elapsed_seconds=time.monotonic()-began,formal_training_started=False)
    t=W/'status.tmp';t.write_text(json.dumps(value,indent=2));t.replace(W/'status.json')


try:
    while time.monotonic()-began<12000:
        registry=json.loads((O/'native_pipeline_registry.json').read_text())
        known={0,1,2,3};admitted=4
        for d in sorted((O/'native_batches').glob('batch_[0-9]*')):
            if not (d/'summary.json').exists():raise RuntimeError('native batch incomplete; preserve and inspect '+str(d))
            report=json.loads((d/'summary.json').read_text());assert report['status']=='native_batch_completed'
            known.update(report['candidate_ids']);admitted+=report['admitted']
        if admitted>=500:
            save({'status':'target_reached_pending_final_library_audit','admitted':admitted,'native_batches':runs});break
        source=[]
        for file in registry['optimizer_record_paths']:source.extend(rows(Path(file)))
        if len({x['candidate_id'] for x in source})!=len(source):raise ValueError('duplicate optimizer records')
        ready=sum(x['passed_requested_kinematic_checks'] and x['candidate_id'] not in known for x in source)
        done=all((Path(x)/'report.json').exists() for x in registry['producer_dirs']+registry['optimizer_terminal_dirs'])
        if ready>=32 or (done and ready>=2):
            cmd=['/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python','-u',str(S/'src/stage3_pick500_native_batch_v2.py'),'--batch-size','32']
            save({'status':'running_native_batch','admitted':admitted,'ready_candidates':ready,'native_batches':runs})
            with (W/('batch_call_'+str(runs).zfill(3)+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                (W/('batch_child_'+str(runs).zfill(3)+'.json')).open('x').write(json.dumps({'pid':child.pid,'command':cmd}))
                code=child.wait()
            (W/('batch_exit_'+str(runs).zfill(3)+'.json')).open('x').write(json.dumps({'returncode':code,'child_reaped':True}))
            if code:raise RuntimeError('native batch guard/child failed; preserve original log')
            runs+=1;continue
        if done:
            save({'status':'initial_sources_drained_replacement_sampling_required','admitted':admitted,
                'target':500,'deficit':500-admitted,'pending_singleton':ready,'native_batches':runs});break
        save({'status':'waiting_for_more_completed_candidates','admitted':admitted,'ready_candidates':ready,'native_batches':runs})
        time.sleep(15)
    else:raise RuntimeError('bounded admission controller wait expired')
except BaseException as e:
    save({'status':'failed_preserved_batch_evidence','error':repr(e),'native_batches':runs});raise
