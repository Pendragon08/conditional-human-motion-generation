// Resumable 32 MiB publication parts. Buffers in RAM; never stages large disk files.
const fs=require('node:fs'),path=require('node:path'),https=require('node:https'),crypto=require('node:crypto'),zlib=require('node:zlib');
const {execFileSync,spawn}=require('node:child_process');
const reader=require('./publication_tar_reader_20261105.cjs');
const {root,dir,inventory,token,source,tar}=reader;
const repo='Pendragon08/conditional-human-motion-generation',release=434987063,chunkBytes=32*1024*1024;
const mode=process.argv[2];if(!['weights','evidence','logs'].includes(mode))throw Error('Expected weights, evidence, or logs');
const prefix={weights:'archived-weights-v3',evidence:'local-evidence-v3',logs:'completed-logs-review-v3'}[mode];
const receiptPath=path.join(dir,prefix+'-receipt.json');
const receipt=fs.existsSync(receiptPath)?JSON.parse(fs.readFileSync(receiptPath)):{mode,prefix,status:'in_progress',parts:[]};
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const api=require('./publication_remote_api_20261105.cjs').api;
function save(){const tmp=receiptPath+'.tmp';fs.writeFileSync(tmp,JSON.stringify(receipt,null,2));fs.renameSync(tmp,receiptPath);}
function listAssets(){const rows=[];for(let p=1;;p++){const batch=api(`releases/${release}/assets?per_page=100&page=${p}`);rows.push(...batch);if(batch.length<100)return rows;}}
let assets=new Map(listAssets().map(a=>[a.name,a]));
function relayPost(name,buffer){return new Promise((resolve,reject)=>{
 const remote='/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/publication_snapshot_20261105/relay_release_chunk_buffered_v2_20261105.py';
 const child=spawn('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','10.147.17.32','python3','-u',remote],{stdio:['pipe','pipe','pipe']});let out='',err='',settled=false;
 const done=(e,r)=>{if(settled)return;settled=true;clearTimeout(timer);e?reject(e):resolve(r);};
 const timer=setTimeout(()=>{child.kill();done(Error('SSH relay upload timeout'));},3600000);
 child.stdout.on('data',b=>out+=b);child.stderr.on('data',b=>err+=b);child.on('error',e=>done(e));child.stdin.on('error',e=>done(e));
 child.on('close',code=>{if(code!==0)return done(Error('SSH relay failed: '+err.slice(0,500)));try{const a=JSON.parse(out);if(a.error)return done(Error(a.error));done(null,a);}catch(e){done(e);}});
 child.stdin.write(JSON.stringify({token,repo,release,name,bytes:buffer.length})+'\n');child.stdin.end(buffer);
 });}
function post(name,buffer){return new Promise((resolve,reject)=>{
 const url=new URL(`https://uploads.github.com/repos/${repo}/releases/${release}/assets`);url.searchParams.set('name',name);let settled=false;
 const done=(err,value)=>{if(settled)return;settled=true;clearTimeout(timer);err?reject(err):resolve(value);};
 const req=https.request(url,{method:'POST',headers:{Authorization:'Bearer '+token,'User-Agent':'DreamControl-reproduction-publication','Content-Type':'application/octet-stream','Content-Length':buffer.length,Accept:'application/vnd.github+json'}},res=>{let body='';res.on('data',b=>body+=b);res.on('end',()=>{if(res.statusCode!==201)return done(Error('HTTP '+res.statusCode+': '+body.slice(0,300)));try{done(null,JSON.parse(body));}catch(e){done(e);}});});
 const timer=setTimeout(()=>{req.destroy();done(Error('Upload timeout'));},240000);
 req.on('error',e=>done(e));req.on('close',()=>{if(!settled)done(Error('Connection closed before upload confirmation'));});req.end(buffer);
 });}
async function upload(index,buffer){
 const name=prefix+'.part'+String(index).padStart(4,'0'),digest=sha(buffer);let asset=assets.get(name);
 for(let attempt=1;attempt<=5;attempt++){
  try{
   if(asset?.state==='uploaded'){if(asset.size!==buffer.length||asset.digest!=='sha256:'+digest)throw Error('Existing uploaded part differs: '+name);}
   else {if(asset){if(asset.state!=='starter')throw Error('Unexpected asset state');api('releases/assets/'+asset.id,'DELETE');}asset=await (process.argv.includes('--relay')?relayPost(name,buffer):post(name,buffer));}
   if(asset.state!=='uploaded'||asset.size!==buffer.length||asset.digest!=='sha256:'+digest)throw Error('GitHub part verification mismatch');
   assets.set(name,asset);const record={index,name,id:asset.id,bytes:buffer.length,sha256:digest,url:asset.browser_download_url,verified:true};receipt.parts=receipt.parts.filter(r=>r.index!==index);receipt.parts.push(record);receipt.parts.sort((a,b)=>a.index-b.index);save();console.log(JSON.stringify({mode,verified_part:index,bytes:buffer.length}));return;
  }catch(e){if(/differs|Unexpected asset/.test(String(e))||attempt===5)throw e;console.log(JSON.stringify({mode,part:index,retry:attempt,error:String(e)}));await new Promise(r=>setTimeout(r,Math.min(attempt*3000,15000)));asset=listAssets().find(a=>a.name===name);}
 }
}
function tarHeader(name,size){const b=Buffer.alloc(512);b.write(name);b.write('0000644\0',100);b.write('0000000\0',108);b.write('0000000\0',116);b.write(size.toString(8).padStart(11,'0')+'\0',124);b.write('00000000000\0',136);b.fill(32,148,156);b[156]=48;b.write('ustar\0',257);b.write('00',263);b.write(b.reduce((s,x)=>s+x,0).toString(8).padStart(6,'0')+'\0 ',148);return b;}
async function* evidence(rows,manifest){yield tarHeader('MANIFEST.json',manifest.length);yield manifest;if(manifest.length%512)yield Buffer.alloc(512-manifest.length%512);for(const r of rows){yield tarHeader('files/'+r.sha256,r.bytes);if(r.bytes)yield* source(r);if(r.bytes%512)yield Buffer.alloc(512-r.bytes%512);}yield Buffer.alloc(1024);}
async function run(){
 let input,rows;
 if(mode==='weights'){rows=inventory.archived_weights;const m=Buffer.from(JSON.stringify({format:'project-checkpoints-v1',rows:rows.map(r=>({original_path:r.original_remote_path,member:'weights/'+r.sha256+'.pt',bytes:r.bytes,sha256:r.sha256}))},null,2));input=tar(rows,m);}
 if(mode==='evidence'){
  const planFile=path.join(dir,'local-evidence-v3-plan.json.gz');if(fs.existsSync(planFile))rows=JSON.parse(zlib.gunzipSync(fs.readFileSync(planFile)));else{rows=inventory.files.filter(r=>!r.publication_hold).map(r=>{const b=fs.readFileSync(r.local_path);return {...r,bytes:b.length,sha256:sha(b),published_path:r.relative_path||('external/'+path.basename(path.dirname(r.local_path))+'/'+path.basename(r.local_path))};});fs.writeFileSync(planFile,zlib.gzipSync(JSON.stringify(rows)));}
  const m=Buffer.from(JSON.stringify({format:'project-local-evidence-v1',rows:rows.map(r=>({original_path:r.published_path,member:'files/'+r.sha256,bytes:r.bytes,sha256:r.sha256}))},null,2));input=evidence(rows,m);
 }
 if(mode==='logs'){
  const file=path.join(root,'training_space_archive_20261105/completed_logs_and_review_png/v1/payload.tar');rows=[{local_path:file,bytes:352358400,sha256:'c491df125d039fe7357595176b90e576bcbc94a6f5321e8f32259344b4fdf1d4',relative_path:'completed_logs_and_review_png/v1/payload.tar'}];input=source(rows[0]);
 }
 if(receipt.error){receipt.previous_errors=[...(receipt.previous_errors||[]),receipt.error];delete receipt.error;}receipt.status='in_progress';receipt.source_file_count=rows.length;receipt.source_file_bytes=rows.reduce((s,r)=>s+r.bytes,0);save();
 let buffer=Buffer.alloc(chunkBytes),used=0,index=0,total=0;const completeHash=crypto.createHash('sha256'),pending=new Set();
 async function queue(b){const n=++index;const p=upload(n,b);pending.add(p);p.then(()=>pending.delete(p),()=>{});if(pending.size>=2)await Promise.race(pending);}
 for await(const b of input){completeHash.update(b);total+=b.length;let pos=0;while(pos<b.length){const n=Math.min(buffer.length-used,b.length-pos);b.copy(buffer,used,pos,pos+n);used+=n;pos+=n;if(used===buffer.length){await queue(buffer);buffer=Buffer.alloc(chunkBytes);used=0;}}}
 if(used)await queue(buffer.subarray(0,used));await Promise.all(pending);
 Object.assign(receipt,{status:'completed_all_parts_and_source_hashes_verified',part_count:index,reconstructed_bytes:total,reconstructed_sha256:completeHash.digest('hex'),reconstructed_name:prefix+'.tar',completed_utc:new Date().toISOString()});save();console.log(JSON.stringify({mode,status:receipt.status,parts:index,bytes:total}));
}
run().catch(e=>{receipt.status='failed_preserved';receipt.error=String(e);save();console.error(String(e));process.exitCode=1;});
