const fs=require('node:fs');const path=require('node:path');
const out=path.resolve('work/dreamcontrol_v2_reproduction_20261029/reports/hf_amass_g1_search_20261104');fs.mkdirSync(out,{recursive:true});
const requests=[...['DreamControl','ReactiveBFM','HumanML3D','AMASS','g1-retarget','GMR','TextOp'].map(term=>({kind:'datasets',term})),...['DreamControl','ReactiveBFM'].map(term=>({kind:'models',term}))];
const results=[];
(async()=>{
for(let start=0;start<requests.length;start+=3){
 const batch=await Promise.allSettled(requests.slice(start,start+3).map(async q=>{
  const url=`https://huggingface.co/api/${q.kind}?search=${encodeURIComponent(q.term)}&limit=100`;
  const response=await fetch(url,{signal:AbortSignal.timeout(30000)});if(!response.ok)throw Error(`${url}: ${response.status}`);
  const data=await response.json();return {...q,url,count:data.length,repositories:data.map(x=>({id:x.id,private:x.private,gated:x.gated,downloads:x.downloads,likes:x.likes}))};
 }));batch.forEach((r,i)=>results.push(r.status==='fulfilled'?r.value:{...requests[start+i],error:String(r.reason)}));
}
const record={queried_at_utc:new Date().toISOString(),scope:'public HF dataset/model name search, at most100 per term, no motion arrays downloaded',results};
fs.writeFileSync(path.join(out,'name_search.json'),JSON.stringify(record,null,2));
process.stdout.write(JSON.stringify(record));
})().catch(e=>{process.stderr.write(String(e));process.exitCode=1;});
