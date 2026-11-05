const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),{execFileSync}=require('node:child_process');
const L=path.resolve(__dirname,'..'), R='/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029';
const stage=R+'/repair_prior29_v2_20261105', src=path.join(__dirname,'repair_prior29_v2_20261105');
const cfg={baseline_config:R+'/stage2_prior29_20261103/config.json',project_root:R,stage_dir:stage,
 run_name:'grab_omni29_mdm_floor_l2_guided_v2',max_steps:10000,max_sample_batch_size:32,
 sample_gpu_memory_fraction:0.1,checkpoint_size_budget:576*1024**2,disk_floor_bytes:8963227648,
 implementation_note:'Independent four-issue repair; fixed upstream L2 guide with floor-coordinate transform and shared task masks. Same project data and initialization; not paper Full-Mix.',
 checkpoint_sizing:{baseline_measured_bytes:503839702,per_checkpoint_budget_bytes:576*1024**2,
  planned_retained:3,spare_allowances:2,minimum_start_bytes:11983126528,
  rationale:'New full Adam/RNG serialization must fit measured budget before formal start; 576MiB is 19.9% above prior complete payload, three retained plus two spare allowances. Hard 11GiB startup and runtime floor unchanged.'}};
const entries=[];
for(const name of fs.readdirSync(src).sort()){
 if(!name.endsWith('.py')&&!name.endsWith('.md')&&!name.endsWith('.json'))continue;
 if(name==='pipeline_controller_v2.py')continue;
 const b=fs.readFileSync(path.join(src,name)); entries.push({path:'src/'+name,sha:crypto.createHash('sha256').update(b).digest('hex'),data:b.toString('base64')});
}
for(const name of ['test_repair_prior29_postopt_v2.py','test_repair_prior29_eval_protocol_v2.py']){
 const b=fs.readFileSync(path.join(L,'tests',name));entries.push({path:'tests/'+name,sha:crypto.createHash('sha256').update(b).digest('hex'),data:b.toString('base64')});
}
const cb=Buffer.from(JSON.stringify(cfg,null,2)+'\n');entries.push({path:'config.json',sha:crypto.createHash('sha256').update(cb).digest('hex'),data:cb.toString('base64')});
const binding={schema:'repair_prior29_source_binding_v1',files:Object.fromEntries(entries.map(e=>[e.path,e.sha]))};
entries.push({path:'source_binding.json',data:Buffer.from(JSON.stringify(binding,null,2)+'\n').toString('base64')});
const code=`import sys,json,base64,hashlib,pathlib\np=json.load(sys.stdin)\ns=pathlib.Path(p['stage'])\ns.mkdir(exist_ok=False)\nfor e in p['entries']:\n t=s/e['path'];t.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(e['data'])\n if 'sha' in e: assert hashlib.sha256(b).hexdigest()==e['sha']\n with t.open('xb') as f: f.write(b)\n(s/'runs').mkdir()\nprint(json.dumps({'stage':str(s),'files':len(p['entries']),'binding_sha256':hashlib.sha256((s/'source_binding.json').read_bytes()).hexdigest()}))\n`;
const command="python3 -c 'import base64;exec(base64.b64decode(\""+Buffer.from(code).toString('base64')+"\"))'";
const output=execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32',command],{input:JSON.stringify({stage,entries}),encoding:'utf8',maxBuffer:1024*1024,timeout:120000});
const out=path.join(L,'reports','repair_prior29_v2_20261105');fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'deployment_receipt.json'),output,{flag:'wx'});fs.writeFileSync(path.join(out,'config.json'),cb,{flag:'wx'});fs.writeFileSync(path.join(out,'source_binding.json'),JSON.stringify(binding,null,2),{flag:'wx'});console.log(output);
