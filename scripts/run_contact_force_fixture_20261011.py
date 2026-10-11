"""Budgeted warm-runtime box contact fixture; failures are retained."""
import argparse,json,subprocess,time,shutil
from pathlib import Path
import run_reset_alignment_checks_v4 as runtime
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/contact_force_fixture_20261011'
PREVIOUS=runtime.PREVIOUS+[ROOT/('results/'+s) for s in (
 'reset_alignment_v4_20261011','condition_feedback_probe_20261011',
 'condition_feedback_probe_v2_20261011','contact_sampling_v5_20261011',
 'contact_sampling_v6_20261011','contact_sampling_v7_20261011','contact_sampling_v8_20261011')]
def save(p,x): p.write_text(json.dumps(x,indent=2))
def main(check):
 import fcntl
 OUT.mkdir(exist_ok=True)
 with (OUT/'queue.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  files=['scripts/contact_force_fixture_20261011.py','scripts/run_contact_force_fixture_20261011.py',
   'scripts/run_reset_alignment_checks_v4.py','configs/env-closd-baseline-20261010.json',
   'docs/contact_force_fixture_20261011.md']
  hashes={p:runtime.sha(ROOT/p) for p in files}
  version=OUT/'versions.json'
  if version.exists(): assert json.loads(version.read_text())==hashes
  else: save(version,hashes)
  ledger_path=OUT/'budget_ledger.json'
  ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else []
  for label in (['repeat_a'] if check else ['repeat_a','repeat_b']):
   dest=OUT/label
   if (dest/'summary.json').exists():
    assert any(x['label']==label and x['exit_code']==0 for x in ledger)
    continue
   assert not dest.exists(),'Incomplete attempt preserved'
   previous=[x for d in PREVIOUS if (d/'budget_ledger.json').exists() for x in json.loads((d/'budget_ledger.json').read_text())]
   assert sum(x['seconds'] for x in previous+ledger)<72*3600
   assert sum(x['seconds'] for x in previous+ledger if x['stage']=='measurement')<4*3600
   assert shutil.disk_usage('/data').free>20*1024**3
   assert sum(p.stat().st_size for d in PREVIOUS+[OUT] for p in d.rglob('*') if p.is_file())<15*1024**3
   runtime.OUT=OUT; resources=runtime.resource_guard()
   save(OUT/'state.json',dict(status='RUNNING',label=label,resources=resources))
   cmd=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/contact_force_fixture_20261011.py'),'--output',str(dest)]
   begin=time.monotonic()
   with (OUT/(label+'.log')).open('w') as log:
    try: code=subprocess.run(cmd,env=runtime.environment(),stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
    except subprocess.TimeoutExpired: code=124
   ledger.append(dict(label=label,stage='measurement',seconds=time.monotonic()-begin,exit_code=code));save(ledger_path,ledger)
   if code:
    save(OUT/'state.json',dict(status='FAILED_RETAINED',label=label));raise RuntimeError('Fixture failure preserved')
   result=json.loads((dest/'summary.json').read_text())
   assert result['source_sha256']==hashes['scripts/contact_force_fixture_20261011.py']
   assert runtime.sha(dest/'fixture_arrays.npz')==result['arrays_sha256']
   print('FIXTURE_DOCUMENT_EXECUTION_COMPLETE',label,result['status'],flush=True)
  save(OUT/'state.json',dict(status='WITNESS_COMPLETE' if check else 'REPEATS_COMPLETE'))
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true');main(p.parse_args().check)
