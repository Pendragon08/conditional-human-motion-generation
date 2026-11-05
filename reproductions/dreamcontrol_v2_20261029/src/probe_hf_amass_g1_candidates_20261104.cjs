const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const out=path.resolve('work/dreamcontrol_v2_reproduction_20261029/reports/hf_amass_g1_search_20261104');fs.mkdirSync(out,{recursive:true});
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
async function bounded(url,cap=1000000,firstLine=false){
 const r=await fetch(url,{signal:AbortSignal.timeout(30000)});if(!r.ok)return {url,status:r.status};
 const reader=r.body.getReader();let chunks=[],length=0;
 while(true){const v=await reader.read();if(v.done)break;chunks.push(Buffer.from(v.value));length+=v.value.byteLength;
  if(firstLine&&Buffer.concat(chunks).includes(10)){await reader.cancel();break;}
  if(length>cap){await reader.cancel();throw Error('bounded read cap: '+url);}}
 let b=Buffer.concat(chunks);if(firstLine)b=b.subarray(0,b.indexOf(10)<0?b.length:b.indexOf(10));
 return {url,status:r.status,commit:r.headers.get('x-repo-commit'),etag:r.headers.get('etag'),bytes:b.length,sha256:sha(b),buffer:b};
}
(async()=>{
 const repos=['JianuoCao/TEXEDO','fleaven/Retargeted_AMASS_for_robotics','ember-lab-berkeley/AMASS_Retargeted_for_G1','wenfag/AMASS_Retargeted_for_G1','Yochish/TextOp-Data','Camellia-Boy/TextOp-Data-Whole','openhe/g1-retargeted-motions','junmokane/AMASS_retarget_g1'];
 const results=[];
 for(let i=0;i<repos.length;i+=3){const batch=await Promise.allSettled(repos.slice(i,i+3).map(async repo=>{
  const r=await bounded(`https://huggingface.co/datasets/${repo}/resolve/main/README.md`,200000);
  const folder=path.join(out,repo.replace('/','__'));fs.mkdirSync(folder,{recursive:true});
  if(r.buffer){fs.writeFileSync(path.join(folder,'README.md'),r.buffer);r.excerpt=r.buffer.toString().slice(0,13000);delete r.buffer;}
  return {repo,...r};}));batch.forEach((r,k)=>results.push(r.status==='fulfilled'?r.value:{repo:repos[i+k],error:String(r.reason)}));}
 const tex=results.find(r=>r.repo==='JianuoCao/TEXEDO');const revision=tex.commit||'main';const folder=path.join(out,'JianuoCao__TEXEDO');
 for(const file of ['metadata/dataset_summary.json','prepare_texedo_dataset.py','data/train.jsonl']){
  const r=await bounded(`https://huggingface.co/datasets/JianuoCao/TEXEDO/resolve/${revision}/${file}`,1000000,file.endsWith('jsonl'));
  if(r.buffer){fs.writeFileSync(path.join(folder,file.replaceAll('/','__')),r.buffer);r.excerpt=r.buffer.toString();delete r.buffer;}
  results.push({repo:'JianuoCao/TEXEDO',file,...r});
 }
 const info={queried_at_utc:new Date().toISOString(),scope:'bounded README/metadata and first JSONL record, not a bulk download',results};
 fs.writeFileSync(path.join(out,'candidate_cards_and_texedo_contract.json'),JSON.stringify(info,null,2));process.stdout.write(JSON.stringify(info));
})().catch(e=>{process.stderr.write(String(e));process.exitCode=1;});
