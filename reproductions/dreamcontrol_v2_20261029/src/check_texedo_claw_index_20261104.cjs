const fs=require('node:fs'),path=require('node:path');
const out=path.resolve('work/dreamcontrol_v2_reproduction_20261029/reports/hf_amass_g1_search_20261104/JianuoCao__TEXEDO');
const commit=JSON.parse(fs.readFileSync(path.join(out,'two_pair_sample_probe.json'),'utf8')).revision;
(async()=>{
 const url=`https://huggingface.co/datasets/JianuoCao/TEXEDO/resolve/${commit}/data/train.jsonl`;
 const r=await fetch(url,{signal:AbortSignal.timeout(30000)});if(!r.ok)throw Error(r.status);
 const rd=r.body.getReader(),decoder=new TextDecoder();let pending='',bytes=0,lines=0,match=null;
 while(!match){const v=await rd.read();if(v.done)break;bytes+=v.value.byteLength;if(bytes>32*1024**2){await rd.cancel();throw Error('index read cap');}
  pending+=decoder.decode(v.value,{stream:true});let at;while((at=pending.indexOf('\n'))>=0){const line=pending.slice(0,at);pending=pending.slice(at+1);if(!line.trim())continue;lines++;const row=JSON.parse(line);if(row.id==='020308'){match=row;break;}}}
 await rd.cancel();if(!match)throw Error('target not in scanned training index');
 const captions=fs.readFileSync(path.join(out,'texts__claw__020__020308.txt'),'utf8').split(/\r?\n/).filter(Boolean).map(x=>x.split('#')[0]);
 const same=JSON.stringify(captions)===JSON.stringify(match.captions.map(x=>x.caption));
 const report={revision:commit,scanned_index_bytes:bytes,scanned_rows:lines,record:match,actual_txt_caption_count:captions.length,
  actual_jsonl_caption_count:match.captions.length,num_texts_field:match.num_texts,caption_arrays_equal:same,
  conclusion:same?'actual captions and text file match; num_texts scalar can be stale metadata':'caption mismatch requires source-specific investigation before adoption'};
 fs.writeFileSync(path.join(out,'claw_sample_index_binding.json'),JSON.stringify(report,null,2));process.stdout.write(JSON.stringify(report));
})().catch(e=>{process.stderr.write(String(e));process.exitCode=1;});
