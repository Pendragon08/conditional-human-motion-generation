// Process the remainder once from a traceable fixed queue; no bulk source archive.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{spawn}=require('child_process'),{once}=require('events');
const root=path.resolve(__dirname,'..'),reports=path.join(root,'reports/grab_dex_motionhub_20261103');
const readJsonl=p=>fs.readFileSync(p,'utf8').trim().split(/\r?\n/).map(JSON.parse);
const pairs=readJsonl(path.join(reports,'candidate_pairs.jsonl'));
const prior=[...readJsonl(path.join(reports,'batch10_records.jsonl')),...readJsonl(path.join(reports,'batch100_records.jsonl'))];
const rows=readJsonl(path.join(root,'reports/grab_source_registry_20261030.jsonl'));
const byId=new Map(rows.map(x=>[x.member.replace(/\.npz$/,''),x]));
const done=new Set(prior.map(x=>x.id));done.add('s9/wineglass_pass_1');
if(done.size!==111||prior.some(x=>x.status!=='technical_preview'))throw Error('prior coverage');
const selected=pairs.filter(x=>!done.has(x.id));
if(selected.length!==1224||new Set(selected.map(x=>x.id)).size!==1224)throw Error('remaining coverage');
const queue=Buffer.from(selected.map(x=>JSON.stringify({id:x.id,robot_relative_path:x.robot_relative_path,source_member_sha256:x.source_member_sha256,text_sha256:x.text_sha256})).join('\n')+'\n');
fs.writeFileSync(path.join(reports,'batch1224_queue.jsonl'),queue,{flag:'wx'});
const cmd='cd /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029; DEXWEAVE_BATCH_ID=batch1224 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python dexweave_stream_robot_batch_v2_20261103.py';
const child=spawn('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=8','10.147.17.32',cmd],{stdio:['pipe','inherit','inherit']});
async function get(url){let last;for(let i=0;i<4;i++){try{const r=await fetch(url,{signal:AbortSignal.timeout(60000)});if(!r.ok)throw Error(`HTTP ${r.status}`);return Buffer.from(await r.arrayBuffer())}catch(e){last=e;await new Promise(resolve=>setTimeout(resolve,(i+1)*1500))}}throw last}
async function write(b){if(!child.stdin.write(b))await once(child.stdin,'drain')}
async function main(){
 for(const row of selected){
  const source=byId.get(row.id);if(!source||source.member_sha256!==row.source_member_sha256)throw Error('source registry');
  const base=`https://huggingface.co/api/datasets/${row.robot_repository}/tree/${row.robot_revision}/${path.posix.dirname(row.robot_relative_path)}?expand=true`;
  const meta=JSON.parse((await get(base)).toString()).find(x=>x.path===row.robot_relative_path);
  if(!meta||!meta.lfs?.oid||meta.size>30000000)throw Error('missing pinned LFS metadata');
  const raw=await get(`https://huggingface.co/datasets/${row.robot_repository}/resolve/${row.robot_revision}/${row.robot_relative_path}`);
  if(raw.length!==meta.size||crypto.createHash('sha256').update(raw).digest('hex')!==meta.lfs.oid)throw Error('publisher hash mismatch');
  const h={id:row.id,bytes:raw.length,published_sha256:meta.lfs.oid,source_member_sha256:row.source_member_sha256,source_frames:source.frames,text_sha256:row.text_sha256,captions:row.captions,risk_codes:row.risk_codes};
  await write(Buffer.from(JSON.stringify(h)+'\n'));await write(raw);
 }
 child.stdin.end();const [code]=await once(child,'close');if(code!==0)throw Error(`server exited ${code}`);
 fs.writeFileSync(path.join(reports,'batch1224_producer_status.json'),JSON.stringify({status:'completed',requested:selected.length,queue_sha256:crypto.createHash('sha256').update(queue).digest('hex')},null,2),{flag:'wx'});
}
main().catch(e=>{console.error(e);try{fs.writeFileSync(path.join(reports,'batch1224_producer_status.json'),JSON.stringify({status:'error',error:String(e)},null,2),{flag:'wx'})}catch{}child.stdin.destroy();process.exitCode=1});
