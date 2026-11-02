"""Keep polling light; full trace validation belongs to the durable CPU audit."""
import pathlib,ast
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
p=B/'provider_tools/collect_teacher15000_strict.py';s=p.read_text(encoding='utf-8')
start=s.index("if 'outer' in f and 'exit' in f and 'receipt' in f and f['outer']['actual_returncode']==0:")
end=s.index('print(json.dumps(result))',start)
s=s[:start]+'''result['full_execution_closed']=bool('outer' in f and 'exit' in f and 'receipt' in f and f['outer']['actual_returncode']==0 and f['receipt'].get('output_writers_closed'))
result['closed_trace_cpu_audit_required']=result['full_execution_closed']
result['private_trace_files_read']=0
if result['full_execution_closed']:
 result['closed_evidence_metadata']={k:result['sha256'][k] for k in ['receipt','process','exit','outer']}
 result['next_action']='Use the new durable CPU audit before accepting physical decisions or constructing generator data'
''' +s[end:]
s=s.replace('Read actual frozen-policy replay state and preserve a unique SSH outcome.','Read only lightweight full replay metadata; never open private trace arrays.')
target=B/'provider_tools/collect_teacher15000_strict_light.py';assert not target.exists();ast.parse(s);target.write_text(s,encoding='utf-8')
print('New light collector prepared; no SSH dispatch or trace read')
