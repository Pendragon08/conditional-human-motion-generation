// Stream the published OMOMO visualization tar and extract only 15 previously
// unreviewed long-clip source videos; no whole archive is stored. Names and
// nominal clocks are evidence for review, not identity or semantic acceptance.
const https=require('https'),zlib=require('zlib'),fs=require('fs'),path=require('path'),crypto=require('crypto'),{Transform}=require('stream');
const output=path.resolve(process.argv[2]);
if(!fs.existsSync(output)||fs.readdirSync(output).length)throw Error('output must be a new empty directory');
const wanted=new Set(['sub12_largebox_034','sub12_largebox_035','sub12_largebox_036','sub12_largebox_038','sub12_largebox_042','sub12_largebox_043','sub12_largebox_045','sub12_largebox_047','sub12_largebox_072','sub12_largebox_074','sub12_largebox_079','sub12_largebox_080','sub12_largebox_081','sub12_largebox_082','sub12_largebox_086']);
const url='https://drive.usercontent.google.com/download?id=1ek-Kgvtg_NpRKKrfz1WRPy7CKVjHw5wW&export=download&confirm=t';
const log=x=>fs.appendFileSync(path.join(output,'events.jsonl'),JSON.stringify({...x,at:new Date().toISOString()})+'\n');
function diskGuard(){const s=fs.statfsSync(output);if(Number(s.bavail)*Number(s.bsize)<1024**3+100*1024**2)throw Error('local disk safety reserve');}
diskGuard();let inputBytes=0,nextProgress=100*1024**2,found=0,buffer=Buffer.alloc(0),remain=0,padding=0,selected=null,parts=[],finished=false;
const rate=4*1024*1024,start=Date.now();
const throttle=new Transform({transform(chunk,enc,cb){inputBytes+=chunk.length;const delay=Math.max(0,inputBytes/rate*1000-(Date.now()-start));setTimeout(()=>cb(null,chunk),delay);}});
const request=https.get(url,res=>{
 log({event:'response',status:res.statusCode,length:res.headers['content-length'],last_modified:res.headers['last-modified']});
 if(res.statusCode!==200||Number(res.headers['content-length'])!==3030804060){log({event:'failed',reason:'unexpected response',inputBytes});res.destroy();process.exitCode=1;return;}
 res.on('data',()=>{if(inputBytes>=nextProgress){diskGuard();log({event:'progress',found,inputBytes});nextProgress+=100*1024**2;}});
 const gz=zlib.createGunzip();res.pipe(throttle).pipe(gz);
 gz.on('data',chunk=>{
  buffer=Buffer.concat([buffer,chunk]);
  while(true){
   if(remain){const n=Math.min(remain,buffer.length);if(selected)parts.push(buffer.subarray(0,n));buffer=buffer.subarray(n);remain-=n;if(remain)return;
    if(selected){diskGuard();const bytes=Buffer.concat(parts);const target=path.join(output,selected+'.mp4');fs.writeFileSync(target,bytes,{flag:'wx'});log({event:'extracted',sample:selected,bytes:bytes.length,sha256:crypto.createHash('sha256').update(bytes).digest('hex'),compressed_bytes_read:inputBytes});wanted.delete(selected);selected=null;parts=[];found++;if(!wanted.size){finished=true;log({event:'complete',found,inputBytes});res.destroy();throttle.destroy();gz.destroy();return;}}
   }
   if(padding){const n=Math.min(padding,buffer.length);buffer=buffer.subarray(n);padding-=n;if(padding)return;}
   if(buffer.length<512)return;
   const h=buffer.subarray(0,512);buffer=buffer.subarray(512);const name=h.subarray(0,100).toString().split('\0')[0];
   if(!name){finished=true;log({event:'archive_end',found,missing:[...wanted],inputBytes});res.destroy();throttle.destroy();gz.destroy();return;}
   const raw=h.subarray(124,136).toString().replace(/\0/g,'').trim(),size=parseInt(raw||'0',8);if(!Number.isSafeInteger(size)||size<0)throw Error('invalid tar header');
   remain=size;padding=(512-(size%512))%512;const base=path.posix.basename(name,'.mp4');
   if(name.endsWith('.mp4')&&wanted.has(base)){if(size>10000000)throw Error('video exceeds 10 MB limit');selected=base;parts=[];log({event:'matched',name,size,inputBytes});}
  }
 });
 const fail=e=>{if(!finished){log({event:'failed',message:e.message,inputBytes,found});process.exitCode=1;}};
 gz.on('error',fail);res.on('error',fail);throttle.on('error',fail);
});
request.setTimeout(60000,()=>request.destroy(Error('network timeout')));
request.on('error',e=>{if(!finished){log({event:'failed',message:e.message,inputBytes,found});process.exitCode=1;}});
process.on('uncaughtException',e=>{if(!finished)log({event:'failed',message:e.message,inputBytes,found});process.exitCode=1;request.destroy();throttle.destroy();});
