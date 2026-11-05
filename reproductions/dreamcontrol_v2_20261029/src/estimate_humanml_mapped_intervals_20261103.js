// Hypothetical scope analysis only. No clips are processed, skipped or accepted.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=path.resolve(__dirname,'..');
const paths={bml:'data/bmlmovi_smplx_inventory_20261102/records.jsonl',eyes:'data/eyesjapan_smplx_inventory_20261102/records.jsonl',kit:'data/kit_smplx_stream_inventory_20261102/records.jsonl',kit_large:'data/kit_smplx_large_members_20261102/records.jsonl'};
const bytes=p=>fs.readFileSync(path.join(root,p));const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const input=Object.fromEntries(Object.entries(paths).map(([k,p])=>[k,bytes(p)]));
const lines=b=>b.toString('utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
const kit=new Map(lines(input.kit).filter(x=>x.poses_shape).map(x=>[x.member,x]));
for(const x of lines(input.kit_large)){if(kit.has(x.member))throw Error('duplicate KIT source');kit.set(x.member,x)}
function calc(rows,fpsPolicy){
 let full=0,kept=0,keptPadded=0,rowCount=0,invalid=0,outOfBounds=0,mirrored=0,mergedCount=0,declaredMismatches=0;
 for(const r of rows){
  if(!r.poses_shape||!Number.isInteger(r.poses_shape[0]))throw Error('missing source shape');
  const n=r.poses_shape[0],fps=fpsPolicy==='kit_dart_assumed_100' ? 100 : r.declared_fps;
  if(!Number.isFinite(fps)||fps<=0)throw Error('invalid declared fps');
  if(fpsPolicy==='kit_dart_assumed_100'&&r.declared_fps!==100)declaredMismatches++;
  full+=n;const ranges=[];
  for(const row of r.nominal_rows){
   rowCount++;if(String(row.motion_id).startsWith('M'))mirrored++;
   const s=Number(row.start),e=Number(row.end);
   if(!Number.isFinite(s)||!Number.isFinite(e)||s<0||e<=s){invalid++;continue;}
   const a=Math.floor(s*fps/20),b=Math.ceil(e*fps/20);
   if(a>=n||b>n){outOfBounds++;continue;}
   ranges.push([a,b]);
  }
  function union(pad){
   const arr=ranges.map(([a,b])=>[Math.max(0,a-pad),Math.min(n,b+pad)]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);let total=0,end=-1,count=0;
   for(const [a,b] of arr){if(a>end){total+=b-a;end=b;count++;}else if(b>end){total+=b-end;end=b;}}
   return [total,count];
  }
  const [a,c]=union(0),[b]=union(8);kept+=a;keptPadded+=b;mergedCount+=c;
 }
 return {source_count:rows.length,full_source_frames:full,mapping_rows:rowCount,mirrored_rows:mirrored,invalid_rows:invalid,out_of_bounds_rows:outOfBounds,merged_intervals:mergedCount,union_frames:kept,union_plus_8_source_frames_context:keptPadded,ratio_with_context:keptPadded/full,source_headers_disagree_with_assumption:declaredMismatches};
}
const groups={BMLmovi:calc(lines(input.bml),'declared_fps'),EyesJapan:calc(lines(input.eyes),'declared_fps'),KIT:calc([...kit.values()],'kit_dart_assumed_100')};
if(groups.BMLmovi.source_count!==1801||groups.EyesJapan.source_count!==749||groups.KIT.source_count!==4231)throw Error('inventory coverage');
const out={source_sha256:Object.fromEntries(Object.entries(input).map(([k,b])=>[k,sha(b)])),groups,
 assumptions:['Candidate HumanML mapping indices treated as half-open 20Hz frames, plus 8-source-frame context for the illustrative count.','KIT uses third-party DART 100Hz hypothesis, contrary to source headers declaring 120Hz; no source is modified.','Mirrored text rows merge for source-frame counting; mirrored outputs still require a validated conversion and correct text.','Annotations and named sources are not independently validated; invalid/out-of-range intervals excluded only from this estimate.','GMR depends on initialization: 8-frame context does not establish equivalence to full-source retargeting.','Does not include GRAB, other AMASS sources, semantic review, dynamics or actual timing.'],training_ready:false};
const dst=path.join(root,'reports','humanml_mapped_interval_estimate_20261103.json');fs.writeFileSync(dst,JSON.stringify(out,null,2),{flag:'wx'});console.log(JSON.stringify(out,null,2));
