"""Admit one immutable batch of completed generated29DoF candidates in PhysX.

Does not train, alter the running producer/optimizer or filter source datasets.
Repeated invocations skip already completed native batches and preserve rejects.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import numpy as np

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
FLOOR=8*1024**3+456*1024**2


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def json_rows(p):
    if not p.exists():return []
    return [json.loads(x) for x in p.read_text().splitlines(keepends=True) if x.endswith('\n')]


def resources(native_pid):
    table={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid=,rss=,args='],text=True).splitlines():
        fields=line.strip().split(None,3)
        if len(fields)==4:table[int(fields[0])]=(int(fields[1]),int(fields[2]),fields[3])
    roots={native_pid,os.getpid()}
    for pid,script in [(1402368,'stage3_sample_pick29_batch.py'),(1405357,'stage3_pick500_postopt_follow_v3.py')]:
        if pid in table and script in table[pid][2] and str(R) in table[pid][2]:roots.add(pid)
    own=set(roots)
    while True:
        before=len(own);own.update(pid for pid,(ppid,_,_) in table.items() if ppid in own)
        if len(own)==before:break
    rss=sum(table[p][1] for p in own if p in table)*1024
    gpu=0
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    for line in output.splitlines():
        pid,used=[x.strip() for x in line.split(',')]
        if int(pid) in own:
            if not used.isdigit():raise RuntimeError('own GPU accounting unavailable')
            gpu+=int(used)*1024**2
    return {'own_pids':sorted(own),'own_rss_bytes':rss,'own_gpu_bytes':gpu,'free_bytes':shutil.disk_usage('/data').free}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();assert 2<=a.batch_size<=32
    root=O/'native_batches';root.mkdir(exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pilot=json.loads((O/'native_pilot_v1/reference_bank.json').read_text())
        assert len(pilot['references'])==4
        completed=[];seen={0,1,2,3};all_refs=list(pilot['references'])
        directories=sorted(root.glob('batch_[0-9]*'))
        for directory in directories:
            summary=directory/'summary.json'
            if not summary.exists():raise RuntimeError('prior batch has no verified summary; inspect preserved output: '+str(directory))
            old=json.loads(summary.read_text());assert old['status']=='native_batch_completed'
            assert not seen.intersection(old['candidate_ids'])
            native=directory/'native/report.json';assert sha(native)==old['native_report_sha256']
            completed.append(old);seen.update(old['candidate_ids'])
            manifest=directory/'native/reference_bank.json'
            if manifest.exists():
                assert sha(manifest)==old['native_manifest_sha256']
                all_refs.extend(json.loads(manifest.read_text())['references'])
        rows=json_rows(O/'postopt_remaining496_jit_v2/records.jsonl')+json_rows(O/'postopt_remaining488_jit_v3/records.jsonl')
        if len({x['candidate_id'] for x in rows})!=len(rows):raise ValueError('duplicate optimizer IDs')
        selected=[x for x in rows if x['candidate_id'] not in seen and x['passed_requested_kinematic_checks']][:a.batch_size]
        if len(selected)<2:
            print(json.dumps({'status':'waiting_for_more_completed_candidates','ready':len(selected)}));return
        sample_ids=[]
        for row in selected:
            q=Path(row['output']).resolve();assert q.is_relative_to(O.resolve()) and sha(q)==row['output_sha256']
            report=q.parent/'report.json';assert json.loads(report.read_text())==row
            source=Path(row['sample']);assert sha(source)==row['sample_sha256']
            with np.load(source,allow_pickle=False) as z:
                m=json.loads(z['metadata_utf8'].tobytes())
                assert hashlib.sha256(z['features'].tobytes()).hexdigest()==m['feature_content_sha256']==row['source_feature_content_sha256']
                assert m['candidate_id']==row['candidate_id'] and m['initial_noise_sha256']==row['source_initial_noise_sha256']
                assert np.allclose(m['task_object_position'],row['task_object_position'],atol=1e-12)
            sample_ids.append(row['candidate_id'])
        assert shutil.disk_usage('/data').free>FLOOR+32*1024**2
        d=root/('batch_'+str(len(directories)+1).zfill(4));d.mkdir(exist_ok=False)
        spec=json.loads((R/'stage3_supermimic29_20261103/runtime_bridge_spec.json').read_text())
        cmd=[spec['python'],'-u',str(S/'src/stage3_pick500_replay.py'),'--headless','--device','cuda:1','--output-dir',str(d/'native')]
        for row in selected:cmd.extend(['--candidate-dir',str(Path(row['output']).parent)])
        (d/'inputs.json').open('x').write(json.dumps({'candidate_ids':sample_ids,'rows':selected,'command':cmd,
            'source_sha256':sha(S/'src/stage3_pick500_replay.py'),'scope':'native geometry replay, not PD tracking or task success'},indent=2))
        with (d/'child.log').open('x') as log:
            child=subprocess.Popen(['timeout','--kill-after=20s','240']+cmd,env={**os.environ,**spec['env']},cwd=spec['cwd'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (d/'launch.json').open('x').write(json.dumps({'coordinator_pid':os.getpid(),'child_pid':child.pid,'command':cmd}))
            peak_gpu=peak_rss=0;protection=None
            try:
                while child.poll() is None:
                    snap=resources(child.pid);peak_gpu=max(peak_gpu,snap['own_gpu_bytes']);peak_rss=max(peak_rss,snap['own_rss_bytes'])
                    with (d/'resources.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
                    if snap['own_gpu_bytes']>12*1024**3 or snap['own_rss_bytes']>32*1024**3 or snap['free_bytes']<FLOOR:
                        protection=snap;raise RuntimeError('owned job resource protection')
                    time.sleep(2)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                (d/'exit.json').open('x').write(json.dumps({'returncode':child.returncode,'protection':protection,'failed_preserved':True}))
                raise
            code=child.wait()
        (d/'exit.json').open('x').write(json.dumps({'returncode':code,'child_reaped':True}))
        assert code==0,'native child failed; preserve batch and inspect original log'
        report=json.loads((d/'native/report.json').read_text())
        assert report['status']=='native_replay_completed' and report['candidates']==len(selected) and report['frames']==120
        new_refs=[]
        for i,row in enumerate(selected):
            admission=json.loads((d/'native'/('candidate_'+str(i)+'_admission.json')).read_text())
            assert admission['kinematic_report_sha256']==sha(Path(row['output']).parent/'report.json')
            assert np.allclose(admission['task_object_position'],row['task_object_position'],atol=1e-12)
            assert admission['native_replay_frames']==120 and admission['eligible_for_calibrated_scene_reference']==all(admission['native_checks'].values())
            if admission['eligible_for_calibrated_scene_reference']:
                proof=d/'native'/('candidate_'+str(i)+'_admission.json')
                new_refs.append({'path':row['output'],'sha256':row['output_sha256'],'postopt_report':str(proof),
                    'postopt_report_sha256':sha(proof),'kinematic_report':str(Path(row['output']).parent/'report.json'),
                    'grasp_time_s':2.,'task_object_position':row['task_object_position'],
                    'task_object_quaternion_wxyz':row['task_object_quaternion_wxyz']})
        manifest=d/'native/reference_bank.json'
        if manifest.exists():assert json.loads(manifest.read_text())['references']==new_refs
        elif new_refs:
            single={k:v for k,v in pilot.items() if k!='references'};single['references']=new_refs
            manifest.open('x').write(json.dumps(single,indent=2))
        assert len(new_refs)==report['admitted_references']
        all_refs.extend(new_refs)
        hashes=set()
        for item in all_refs:
            assert sha(item['path'])==item['sha256'] and sha(item['postopt_report'])==item['postopt_report_sha256']
            with np.load(item['path'],allow_pickle=False) as z:
                content=b''.join(np.ascontiguousarray(z[k]).tobytes() for k in ['q','root_pos','root_quat','hand_closed'])
            fingerprint=hashlib.sha256(content).hexdigest();assert fingerprint not in hashes,'duplicate admitted robot-state content';hashes.add(fingerprint)
        summary={'status':'native_batch_completed','candidate_ids':sample_ids,'native_checked':len(selected),'admitted':len(new_refs),
            'native_rejected':len(selected)-len(new_refs),'native_report_sha256':sha(d/'native/report.json'),
            'native_manifest_sha256':sha(manifest) if manifest.exists() else None,
            'own_jobs_gpu_peak_bytes':peak_gpu,'own_jobs_rss_peak_bytes':peak_rss,'formal_training_started':False}
        (d/'summary.json').open('x').write(json.dumps(summary,indent=2))
        combined={k:v for k,v in pilot.items() if k!='references'};combined['references']=all_refs
        combined['reference_target_count']=500
        (d/'partial_reference_bank.json').open('x').write(json.dumps(combined,indent=2))
        raw_count=len(json_rows(O/'candidate_pilot_v1/records.jsonl'))+len(json_rows(O/'candidate_remaining496_v1/records.jsonl'))
        progress={'status':'growing_reference_library','generated_candidates':raw_count,
            'completed_optimizer_records':4+len(rows),'kinematic_passed':4+sum(x['passed_requested_kinematic_checks'] for x in rows),
            'native_checked':4+sum(x['native_checked'] for x in completed)+len(selected),
            'native_admitted':len(all_refs),'target_admitted':500,'still_needed':max(0,500-len(all_refs)),
            'latest_batch':str(d),'current_partial_manifest':str(d/'partial_reference_bank.json'),
            'current_partial_manifest_sha256':sha(d/'partial_reference_bank.json'),
            'formal_reference_count_satisfied':len(all_refs)>=500,'formal_training_started':False,
            'scope':'geometric/task-scene admission only, not PD tracking or Pick success'}
        tmp=O/'library_progress.tmp';tmp.write_text(json.dumps(progress,indent=2));tmp.replace(O/'library_progress.json')
        print(json.dumps({'batch':summary,'library':progress}),flush=True)


if __name__=='__main__':main()
