const fs=require('node:fs'),https=require('node:https'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');const {once}=require('node:events');
const repo='Pendragon08/conditional-human-motion-generation',release=434987063;
const file=path.resolve(process.argv[2]),name=process.argv[3]||path.basename(file);
if(!fs.statSync(file).isFile())throw Error('Not a regular file');
const token=execFileSync('gh',['auth','token'],{encoding:'utf8',stdio:['ignore','pipe','pipe']}).trim();
const out=path.resolve(__dirname,'../reports/publication_inventory_20261105/local_extra_assets.json');
async function shaFile(){const h=crypto.createHash('sha256');for await(const b of fs.createReadStream(file))h.update(b);return h.digest('hex');}
(async()=>{
 const before=fs.statSync(file,{bigint:true}),sha=await shaFile(),size=Number(before.size);
 const existing=JSON.parse(execFileSync('gh',['api',`repos/${repo}/releases/${release}/assets?per_page=100`],{encoding:'utf8',maxBuffer:8*1024*1024})).find(x=>x.name===name);
 let asset=existing;if(existing&&(existing.size!==size||existing.digest!=='sha256:'+sha))throw Error('Different existing asset; preserved');
 if(!asset){
  const url=new URL(`https://uploads.github.com/repos/${repo}/releases/${release}/assets`);url.searchParams.set('name',name);
  const type=name.endsWith('.mp4')?'video/mp4':name.endsWith('.webm')?'video/webm':'application/octet-stream';
  const req=https.request(url,{method:'POST',headers:{Authorization:'Bearer '+token,'User-Agent':'DreamControl-reproduction-publication','Content-Type':type,'Content-Length':size,Accept:'application/vnd.github+json'}});
  const result=new Promise((ok,bad)=>{req.on('error',bad);req.on('response',res=>{let b='';res.on('data',x=>b+=x);res.on('end',()=>res.statusCode===201?ok(JSON.parse(b)):bad(Error('Upload HTTP '+res.statusCode+': '+b.slice(0,500))));});});result.catch(()=>{});
  const h=crypto.createHash('sha256');try{for await(const b of fs.createReadStream(file)){h.update(b);if(!req.write(b))await once(req,'drain');}req.end();asset=await result;}catch(e){req.destroy();throw e;}
  if(h.digest('hex')!==sha)throw Error('File changed while uploading');
 }
 const after=fs.statSync(file,{bigint:true});if(after.mtimeNs!==before.mtimeNs||after.size!==before.size||asset.size!==size||asset.digest!=='sha256:'+sha)throw Error('Asset verification mismatch');
 const receipt={name,bytes:size,sha256:sha,id:asset.id,url:asset.browser_download_url,verified:true};const list=fs.existsSync(out)?JSON.parse(fs.readFileSync(out)):[];if(!list.some(x=>x.id===receipt.id))list.push(receipt);fs.writeFileSync(out,JSON.stringify(list,null,2));console.log(JSON.stringify(receipt));
})().catch(e=>{console.error(String(e));process.exitCode=1;});
