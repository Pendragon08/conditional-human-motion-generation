// Stream only archived project checkpoints to the authorized GitHub release.
// Does not extract, delete, or modify any source archive. Credentials stay in memory.
const fs=require('node:fs'),path=require('node:path'),https=require('node:https'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');const {once}=require('node:events');
const root=path.resolve(__dirname,'..'),dir=path.join(root,'reports/publication_inventory_20261105');
const inventory=JSON.parse(fs.readFileSync(path.join(dir,'local_inventory.json')));
const release=434987063,repo='Pendragon08/conditional-human-motion-generation';
const token=execFileSync('gh',['auth','token'],{encoding:'utf8',stdio:['ignore','pipe','pipe']}).trim();
const receiptPath=path.join(dir,'local_results_upload_receipt.json');
const receipt=fs.existsSync(receiptPath)?JSON.parse(fs.readFileSync(receiptPath)): {release,repo,status:'in_progress',parts:[]};
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const archiveIndexes=new Map();
function indexTar(file){
 if(archiveIndexes.has(file))return archiveIndexes.get(file);
 const fd=fs.openSync(file,'r'),total=fs.fstatSync(fd).size,result=new Map();let pos=0,pax={},longName=null;
 const str=(b,a,n)=>b.subarray(a,a+n).toString('utf8').replace(/\0.*$/s,'');
 while(pos+512<=total){const h=Buffer.alloc(512);fs.readSync(fd,h,0,512,pos);if(h.every(b=>b===0))break;
  const size=parseInt(str(h,124,12).trim()||'0',8),type=String.fromCharCode(h[156]||48);let name=str(h,0,100),prefix=str(h,345,155);if(prefix)name=prefix+'/'+name;
  if(type==='x'||type==='g'){const b=Buffer.alloc(size);fs.readSync(fd,b,0,size,pos+512);let k=0;while(k<b.length){const space=b.indexOf(32,k),len=Number(b.subarray(k,space).toString()),line=b.subarray(space+1,k+len-1).toString();if(!len)throw Error('Invalid PAX');const eq=line.indexOf('=');pax[line.slice(0,eq)]=line.slice(eq+1);k+=len;}}
  else if(type==='L'){const b=Buffer.alloc(size);fs.readSync(fd,b,0,size,pos+512);longName=b.toString().replace(/\0.*$/s,'');}
  else {name=pax.path||longName||name;if(type==='0'||type==='\0')result.set(name,{offset:pos+512,size});pax={};longName=null;}
  pos+=512+Math.ceil(size/512)*512;
 }fs.closeSync(fd);archiveIndexes.set(file,result);return result;
}
function header(name,size){
 if(Buffer.byteLength(name)>100)throw Error('Tar member too long');const b=Buffer.alloc(512);b.write(name);b.write('0000644\0',100);b.write('0000000\0',108);b.write('0000000\0',116);b.write(size.toString(8).padStart(11,'0')+'\0',124);b.write('00000000000\0',136);b.fill(32,148,156);b[156]=48;b.write('ustar\0',257);b.write('00',263);const sum=b.reduce((s,x)=>s+x,0);b.write(sum.toString(8).padStart(6,'0')+'\0 ',148);return b;
}
const tarSize=s=>512+Math.ceil(s/512)*512;
async function* source(row){
 const file=row.local_path||row.archive_path,before=fs.statSync(file,{bigint:true});let start=0;
 if(row.archive_path){const member=indexTar(file).get(row.member);if(!member||member.size!==row.bytes)throw Error('Archive member mismatch: '+row.relative_path);start=member.offset;}
 else if(Number(before.size)!==row.bytes)throw Error('File size mismatch');
 const h=crypto.createHash('sha256');let n=0;
 if(row.bytes===0){if(row.sha256!==h.digest('hex'))throw Error('Empty file hash mismatch');return;}
 for await(const b of fs.createReadStream(file,{start,end:start+row.bytes-1,highWaterMark:1024*1024})){n+=b.length;h.update(b);yield b;}
 const after=fs.statSync(file,{bigint:true});if(n!==row.bytes||h.digest('hex')!==row.sha256||before.ino!==after.ino||before.mtimeNs!==after.mtimeNs||before.size!==after.size)throw Error('Source verification failed: '+row.relative_path);
}
async function* tar(rows,manifest){
 yield header('MANIFEST.json',manifest.length);yield manifest;if(manifest.length%512)yield Buffer.alloc(512-manifest.length%512);
 for(const r of rows){yield header('files/'+r.sha256,r.bytes);yield* source(r);if(r.bytes%512)yield Buffer.alloc(512-r.bytes%512);}yield Buffer.alloc(1024);
}
async function upload(name,size,stream){
 const url=new URL(`https://uploads.github.com/repos/${repo}/releases/${release}/assets`);url.searchParams.set('name',name);
 const req=https.request(url,{method:'POST',headers:{Authorization:'Bearer '+token,'User-Agent':'DreamControl-reproduction-publication','Content-Type':'application/x-tar','Content-Length':size,Accept:'application/vnd.github+json'}});
 let result;const response=new Promise((resolve,reject)=>{req.on('response',res=>{let s='';res.on('data',b=>s+=b);res.on('end',()=>{if(res.statusCode!==201)return reject(Error('Upload HTTP '+res.statusCode+': '+s.slice(0,800)));resolve(JSON.parse(s));});});req.on('error',reject);});response.catch(()=>{});
 const h=crypto.createHash('sha256');let sent=0,last=Date.now();
 try{for await(const b of stream){h.update(b);sent+=b.length;if(!req.write(b))await once(req,'drain');if(Date.now()-last>30000){console.log(JSON.stringify({asset:name,sent,total:size}));last=Date.now();}}req.end();result=await response;}
 catch(e){req.destroy();throw e;}
 const digest=h.digest('hex');if(sent!==size||result.size!==size||result.digest!=='sha256:'+digest)throw Error('GitHub digest/size mismatch');
 return {name,size,sha256:digest,id:result.id,url:result.browser_download_url,verified_utc:new Date().toISOString()};
}
(async()=>{
 const rows=inventory.files.filter(r=>!r.publication_hold).map(r=>{const bytes=fs.readFileSync(r.local_path);return {...r,bytes:bytes.length,sha256:hash(bytes),original_remote_path:r.relative_path||('external/'+path.basename(path.dirname(r.local_path))+'/'+path.basename(r.local_path))};}),parts=[];let batch=[],size=0;for(const r of rows){if(size+r.bytes>1000*1024*1024&&batch.length){parts.push(batch);batch=[];size=0;}batch.push(r);size+=r.bytes;}if(batch.length)parts.push(batch);
 for(let i=0;i<parts.length;i++){
  const name='local-code-reports-media-v2-part'+String(i+1).padStart(2,'0')+'.tar';if(receipt.parts.some(p=>p.name===name))continue;
  const batch=parts[i],manifest=Buffer.from(JSON.stringify({format:'project-local-evidence-v1',rows:batch.map(r=>({original_path:r.original_remote_path,member:'files/'+r.sha256,bytes:r.bytes,sha256:r.sha256}))},null,2));
  const bytes=tarSize(manifest.length)+batch.reduce((s,r)=>s+tarSize(r.bytes),0)+1024;
  const uploaded=await upload(name,bytes,tar(batch,manifest));uploaded.weight_count=batch.length;receipt.parts.push(uploaded);fs.writeFileSync(receiptPath,JSON.stringify(receipt,null,2));console.log(JSON.stringify(uploaded));
 }
 receipt.status='all_local_inventory_files_uploaded_verified';receipt.weight_count=rows.length;receipt.weight_bytes=rows.reduce((s,r)=>s+r.bytes,0);fs.writeFileSync(receiptPath,JSON.stringify(receipt,null,2));console.log(JSON.stringify({status:receipt.status,parts:receipt.parts.length}));
})().catch(e=>{receipt.status='failed_preserved';receipt.error=String(e);fs.writeFileSync(receiptPath,JSON.stringify(receipt,null,2));console.error(String(e));process.exitCode=1;});
