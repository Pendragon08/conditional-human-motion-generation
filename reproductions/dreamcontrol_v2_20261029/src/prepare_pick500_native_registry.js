const fs=require('fs'),path=require('path');
let s=fs.readFileSync(path.join(__dirname,'stage3_pick500_native_batch.py'),'utf8');
function rep(a,b){if(!s.includes(a))throw Error('source differs');s=s.replace(a,b);}
rep("    for pid,script in [(1402368,'stage3_sample_pick29_batch.py'),(1405357,'stage3_pick500_postopt_follow_v3.py')]:",
`    registry=O/'native_pipeline_registry.json'
    extra=json.loads(registry.read_text()).get('active_jobs',[]) if registry.exists() else []
    tracked=[(1402368,'stage3_sample_pick29_batch.py'),(1405357,'stage3_pick500_postopt_follow_v3.py')]
    tracked+= [(x['pid'],x['script']) for x in extra]
    for pid,script in tracked:`);
rep("        rows=json_rows(O/'postopt_remaining496_jit_v2/records.jsonl')+json_rows(O/'postopt_remaining488_jit_v3/records.jsonl')",
`        registry_path=O/'native_pipeline_registry.json'
        registry=json.loads(registry_path.read_text()) if registry_path.exists() else {}
        optimizer_paths=registry.get('optimizer_record_paths',[str(O/'postopt_remaining496_jit_v2/records.jsonl'),str(O/'postopt_remaining488_jit_v3/records.jsonl')])
        rows=[]
        for value in optimizer_paths:
            path=Path(value).resolve();assert path.is_relative_to(O.resolve());rows.extend(json_rows(path))`);
rep("        raw_count=len(json_rows(O/'candidate_pilot_v1/records.jsonl'))+len(json_rows(O/'candidate_remaining496_v1/records.jsonl'))",
`        producers=registry.get('producer_dirs',[str(O/'candidate_pilot_v1'),str(O/'candidate_remaining496_v1')])
        raw_count=0
        for value in producers:
            path=Path(value).resolve();assert path.is_relative_to(O.resolve());raw_count+=len(json_rows(path/'records.jsonl'))`);
rep("'completed_optimizer_records':4+len(rows)","'completed_optimizer_records':4+len(rows)+len(registry.get('pre_v3_rejected_ids',[11]))");
fs.writeFileSync(path.join(__dirname,'stage3_pick500_native_batch_v2.py'),s);
console.log('prepared registry-driven immutable native batch v2');
