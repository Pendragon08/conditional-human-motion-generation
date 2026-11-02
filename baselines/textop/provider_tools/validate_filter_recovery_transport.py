"""Local synthetic subprocess transport fixtures; no SSH or checkpoint operations."""
import argparse
import json
import pathlib
import sys
from backup_filter_recovery_checkpoint import recorded_remote

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert a.run.isascii() and a.run.isidentifier()
b=pathlib.Path(__file__).resolve().parents[1]
out=b/'results/filter_recovery_checkpoint_tools_20261027'/a.run
assert not out.exists();out.mkdir()
cases=[('normal',"print('normal fixture')",10,[sys.executable,'-']),
 ('nonzero',"import sys;print('retained stdout');print('retained stderr',file=sys.stderr);raise SystemExit(7)",10,[sys.executable,'-']),
 ('timeout',"import time;print('before timeout',flush=True);time.sleep(60)",.2,[sys.executable,'-']),
 ('spawn_error','',10,[str(out/'intentionally_missing_executable')])]
records={}
for name,script,timeout,command in cases:
    path=out/(name+'.json')
    try:
        value=recorded_remote(script,path,timeout,command)
        assert name=='normal' and value.strip()=='normal fixture'
    except AssertionError:
        assert name!='normal'
    record=json.loads(path.read_text(encoding='utf-8'));records[name]=record
    assert path.with_name(path.stem+'_intent.json').exists()
    assert record['remote_child_outcome']=='UNKNOWN_UNTIL_SERVER_PROOF'
    for key in ('stdout','stderr'):
        direct=pathlib.Path(record['direct_stream_files'][key])
        assert direct.is_file() and direct.read_text(encoding='utf-8',errors='replace')==record[key]
    assert record['streams_final'] and not record['output_may_grow']
assert records['normal']['actual_client_exit_code']==0 and records['normal']['client_reaped']
assert records['nonzero']['actual_client_exit_code']==7 and records['nonzero']['client_reaped']
assert 'retained stdout' in records['nonzero']['stdout'] and 'retained stderr' in records['nonzero']['stderr']
assert records['timeout']['timed_out'] and records['timeout']['client_reaped'] and records['timeout']['actual_client_exit_code']!=0
assert 'before timeout' in records['timeout']['stdout']
assert 'before timeout' in records['timeout']['first_timeout_streams']['stdout']
assert records['spawn_error']['client_pid'] is None and records['spawn_error']['error']
summary=dict(status='RECOVERY_LOCAL_TRANSPORT_FIXTURES_PASSED',cases=4,
 outcomes={n:{k:r[k] for k in ('actual_client_exit_code','client_reaped','timed_out')} for n,r in records.items()},
 ssh_calls=0,real_checkpoint_loads=0,sdk_runs=0,training_signals=0)
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
