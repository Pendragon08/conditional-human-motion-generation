const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const out=path.resolve('work/dreamcontrol_v2_reproduction_20261029/reports/hf_amass_g1_search_20261104/JianuoCao__TEXEDO');
const cards=JSON.parse(fs.readFileSync(path.join(out,'../candidate_cards_and_texedo_contract.json'),'utf8'));
const commit=cards.results.find(x=>x.repo==='JianuoCao/TEXEDO'&&!x.file).commit;
const samples=[{source:'amass',id:'001169'},{source:'claw',id:'020308'}];
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
async function get(file){const url=`https://huggingface.co/datasets/JianuoCao/TEXEDO/resolve/${commit}/${file}`;
 const r=await fetch(url,{signal:AbortSignal.timeout(30000)});if(!r.ok)throw Error(r.status+' '+url);
 const rd=r.body.getReader();let parts=[],size=0;while(true){const v=await rd.read();if(v.done)break;parts.push(Buffer.from(v.value));size+=v.value.byteLength;if(size>4*1024**2){await rd.cancel();throw Error('small sample cap');}}
 const b=Buffer.concat(parts);const target=path.join(out,file.replaceAll('/','__'));fs.writeFileSync(target,b);return {file,url,bytes:b.length,sha256:sha(b),buffer:b};}
function inspect(b){
 if(!b.subarray(0,6).equals(Buffer.from([147,78,85,77,80,89])))throw Error('not NPY');
 const major=b[6],start=major===1?10:12,len=major===1?b.readUInt16LE(8):b.readUInt32LE(8),header=b.subarray(start,start+len).toString();
 const descr=header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/)[1];
 const shape=header.match(/['"]shape['"]\s*:\s*\(([^)]+)\)/)[1].split(',').map(s=>s.trim()).filter(Boolean).map(Number);
 if(!/fortran_order['"]?\s*:\s*False/.test(header)||shape.length!==2||shape[1]!==36||!['<f4','<f8'].includes(descr))throw Error('unexpected source header '+header);
 const width=descr==='<f4'?4:8,n=shape[0]*36;const value=i=>width===4?b.readFloatLE(start+len+i*width):b.readDoubleLE(start+len+i*width);
 if(b.length!==start+len+n*width)throw Error('NPY byte length');let normError=0,finite=true;
 for(let t=0;t<shape[0];t++){let norm=0;for(let c=0;c<36;c++){let x=value(t*36+c);finite=finite&&Number.isFinite(x);if(c>=3&&c<7)norm+=x*x;}normError=Math.max(normError,Math.abs(Math.sqrt(norm)-1));}
 if(!finite)throw Error('nonfinite sample');return {shape,dtype:descr,body_joint_columns:29,all_finite:finite,root_quaternion_norm_max_error:normError};
}
(async()=>{
 const results=[];
 for(const s of samples){const prefix=s.id.slice(0,3);const m=await get(`motions/${s.source}/${prefix}/${s.id}.npy`),t=await get(`texts/${s.source}/${prefix}/${s.id}.txt`);
  const captions=t.buffer.toString().split(/\r?\n/).filter(Boolean).map(line=>line.split('#')[0]);const info=inspect(m.buffer);delete m.buffer;delete t.buffer;
  results.push({...s,motion:m,text:t,...info,caption_count:captions.length,caption_examples:captions.slice(0,3)});
 }
 const index=JSON.parse(fs.readFileSync(path.join(out,'data__train.jsonl'),'utf8'));const first=results[0];
 if(first.id!==index.id||first.shape[0]!==index.num_frames||first.caption_count!==index.num_texts||JSON.stringify(first.caption_examples)!==JSON.stringify(index.captions.map(x=>x.caption)))throw Error('index/source sample pair mismatch');
 const report={queried_at_utc:new Date().toISOString(),repo:'JianuoCao/TEXEDO',revision:commit,results,amass_first_index_text_and_shape_binding:true,
  scope:'two bounded downloadable file pairs, not corpus acceptance or physical/semantic validation; FPS from paper50Hz still requires release clock confirmation'};
 fs.writeFileSync(path.join(out,'two_pair_sample_probe.json'),JSON.stringify(report,null,2));process.stdout.write(JSON.stringify(report));
})().catch(e=>{process.stderr.write(String(e));process.exitCode=1;});
