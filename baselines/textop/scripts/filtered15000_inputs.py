"""Bind new generator inputs to actual private strict passes and CPU proof."""
import hashlib,json,pathlib
from strict_tracking_coverage import verify_trial
CHECKPOINT='66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88'

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))

def load_selection(spec):
 root=pathlib.Path(spec['root']);path=pathlib.Path(spec['strict_audit_receipt'])
 assert path.parent.parent==root/'filtered15000_20261027' and path==path.resolve()
 assert sha(path)==spec['strict_audit_receipt_sha256']
 a=read(path);assert a['status']=='CLOSED_FULL_STRICT_RESULT_CPU_AUDIT_PASSED'
 assert a['checkpoint_sha256']==CHECKPOINT and a['source_count']==a['attempted']==8278 and a['unattempted']==0
 assert a['complete_passes']+a['failed']==8278 and a['complete_passes']>0
 assert a['strict_result_accepted'] and a['original_processes_exited'] and not a['filtered_corpus_accepted']
 assert a['new_model_loads']==a['new_gpu_contexts']==a['new_sdk_runs']==a['optimizer_updates']==0
 ep=root/'receipts'/('teacher15000_full_audit_'+path.parent.name+'_exit.json')
 assert sha(ep)==spec['strict_audit_supervisor_sha256'];e=read(ep)
 assert e['audit_receipt_sha256']==sha(path) and e['child_reaped'] and e['child_exit_code']==0 and e['supervision_passed']
 assert not e['timed_out'] and e['supervisor_error'] is None and not e.get('cleanup_error') and e['sample_errors']==[]
 pp=root/'receipts'/('teacher15000_full_audit_'+path.parent.name+'_process_exit.json');assert sha(pp)==e['process_exit_sha256']
 pr=read(pp);assert pr['child_reaped'] and pr['child_exit_code']==0 and pr['supervision_passed'] and not pr['timed_out']
 outer=root/'receipts'/('teacher15000_full_audit_'+path.parent.name+'_outer_exit.json');assert read(outer)['actual_returncode']==0
 entry=a['private_artifacts']['accepted_sources.private.json'];p=pathlib.Path(entry['path'])
 assert p.parent==path.parent and p==p.resolve() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
 selected=read(p);assert len(selected)==a['complete_passes'] and len({x['canonical_source'] for x in selected})==len(selected)
 for x in selected:
  t=x['trial'];verify_trial(t,x['frames'],0);assert t['complete'] and not t['failed'] and x['strict_complete_pass']
  assert hashlib.sha256(x['canonical_source'].encode()).hexdigest()==x['motion_id']
 return a,{x['canonical_source']:x for x in selected}

def select_prepared_rows(rows,selection):
 assert len(rows)==len({x['canonical_source'] for x in rows})==8278
 lookup={x['canonical_source']:x for x in rows};assert set(selection)<=set(lookup)
 result=[]
 for source,item in sorted(selection.items(),key=lambda kv:kv[1]['global_index']):
  row=lookup[source];assert row['split']==item['split'] and row['frames50']==item['frames']
  assert row['dof']==29 and row['features']==69
  result.append(row)
 assert len(result)==len(selection)>0
 return result
