"""Read-only completed-arm snapshot; not a final gate or efficacy conclusion."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/condition_feedback_formal_v28_20261012'
ARMS=('raw','ordinary','random_best','native_single','native_best','proposed',
      'no_feedback','shuffled','planned_prefix','tracking_only')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main(pid):
    alive=False
    try:
        os.kill(pid,0);alive=True
    except ProcessLookupError: pass
    cmdline=Path('/proc')/str(pid)/'cmdline'
    command=cmdline.read_bytes().decode(errors='replace').replace('\x00',' ') if cmdline.exists() else None
    if alive: assert command and 'run_condition_feedback_formal_v28_20261012.py' in command
    entries=[]
    for seed in (11,12,13):
        for arm in ARMS:
            path=OUT/'raw'/('test_eval_'+arm+'_s'+str(seed))
            if not (path/'receipt.json').exists(): continue
            receipt=read(path/'receipt.json');binding=read(path/'lineage.json')
            assert receipt['status']=='complete' and receipt['cases']==144
            assert all(sha(path/name)==digest for name,digest in binding['outputs'].items())
            rows=read(path/'metrics.json')
            selected=OUT/'selected'/(arm+'_s'+str(seed)+'.json')
            choices=read(selected)['choices'];lineage=read(selected.with_suffix('.lineage.json'))
            assert lineage['selection_json']==sha(selected)
            assert [r['case_id'] for r in rows]==[r['case_id'] for r in choices]
            result=dict(arm=arm,seed=seed,attempts=len(rows),
                completed=sum(r['completed'] for r in rows),
                construction_completed=sum(c['construction_complete'] for c in choices),
                condition_pass=sum(r['condition_pass'] for r in rows),
                physical_pass=sum(r['physical_pass'] for r in rows),
                tracking_pass=sum(r['tracking_pass'] for r in rows),
                joint_pass=sum(r['joint_pass'] and c['construction_complete'] for r,c in zip(rows,choices)),
                early_terminated=sum(r['terminated'] for r in rows))
            values=[r['endpoint_error_m'] for r in rows if isinstance(r.get('endpoint_error_m'),(int,float)) and math.isfinite(r['endpoint_error_m'])]
            result['endpoint_error_mean_m']=sum(values)/len(values) if values else None
            result['endpoint_measured_count']=len(values)
            result['metadata_sha256']=sha(path/'metadata.json');result['metrics_sha256']=sha(path/'metrics.json')
            entries.append(result)
    payload=dict(time=time.time(),pid=pid,process_alive=alive,verified_command=command,
        queue_state=read(OUT/'queue_state.json'),complete_replay_arm_seed_batches=len(entries),
        expected_replay_arm_seed_batches=30,entries=entries,final_gate=False,
        innovation_confirmed=False,semantic_diversity_not_assessed_here=True,
        qualification='Partial completed-arm snapshot; no changes to live execution or thresholds',
        source_sha256=sha(Path(__file__)))
    destination=OUT/('progress_snapshot_'+str(time.time_ns())+'.json')
    destination.write_text(json.dumps(payload,indent=2),encoding='utf8')
    print(json.dumps(payload,indent=2))
    print('SNAPSHOT_PATH',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True)
    main(p.parse_args().pid)
