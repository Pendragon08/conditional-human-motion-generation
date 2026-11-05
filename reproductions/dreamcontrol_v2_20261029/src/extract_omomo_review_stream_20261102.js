// Extract selected source videos sequentially without storing the 3 GB archive.
// This supplies review material only; identity, timing and licensing need review.
const https = require('https');
const zlib = require('zlib');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const output = path.resolve(process.argv[2]);
const wanted = new Set(['sub16_largebox_036','sub16_largebox_023','sub3_largebox_011','sub16_largebox_026','sub8_largebox_040','sub12_largebox_002','sub16_largebox_049']);
const url = 'https://drive.usercontent.google.com/download?id=1ek-Kgvtg_NpRKKrfz1WRPy7CKVjHw5wW&export=download&confirm=t';
const log = x => fs.appendFileSync(path.join(output,'events.jsonl'),JSON.stringify({...x,at:new Date().toISOString()})+'\n');
let inputBytes=0, found=0, buffer=Buffer.alloc(0), remain=0, padding=0, selected=null, parts=[];
https.get(url, res => {
  log({event:'response',status:res.statusCode,length:res.headers['content-length'],last_modified:res.headers['last-modified']});
  if(res.statusCode!==200 || Number(res.headers['content-length'])!==3030804060){ res.destroy(); return; }
  res.on('data', c => { inputBytes+=c.length; });
  const gz=zlib.createGunzip(); res.pipe(gz);
  gz.on('data', chunk => {
    buffer=Buffer.concat([buffer,chunk]);
    while (true) {
      if(remain){ const n=Math.min(remain,buffer.length); if(selected) parts.push(buffer.subarray(0,n)); buffer=buffer.subarray(n); remain-=n; if(remain)return;
        if(selected){ const bytes=Buffer.concat(parts); const target=path.join(output,selected+'.mp4'); fs.writeFileSync(target,bytes,{flag:'wx'}); log({event:'extracted',sample:selected,bytes:bytes.length,sha256:crypto.createHash('sha256').update(bytes).digest('hex'),compressed_bytes_read:inputBytes}); wanted.delete(selected); selected=null;parts=[]; found++; if(!wanted.size){log({event:'complete',found,inputBytes});res.destroy();gz.destroy();return;} }
      }
      if(padding){const n=Math.min(padding,buffer.length);buffer=buffer.subarray(n);padding-=n;if(padding)return;}
      if(buffer.length<512)return;
      const h=buffer.subarray(0,512);buffer=buffer.subarray(512);
      const name=h.subarray(0,100).toString().split('\0')[0]; if(!name){log({event:'archive_end',found,missing:[...wanted],inputBytes});res.destroy();gz.destroy();return;}
      const raw=h.subarray(124,136).toString().replace(/\0/g,'').trim();const size=parseInt(raw||'0',8);if(!Number.isSafeInteger(size)||size<0)throw Error('invalid tar header');
      remain=size;padding=(512-(size%512))%512;
      const base=path.posix.basename(name,'.mp4');
      if(name.endsWith('.mp4') && wanted.has(base)){if(size>10000000)throw Error('video exceeds 10 MB limit'); selected=base;parts=[];log({event:'matched',name,size,inputBytes});}
    }
  });
  gz.on('error', e => log({event:'error',message:e.message,inputBytes}));
  res.on('error', e => log({event:'error',message:e.message,inputBytes}));
}).on('error',e=>log({event:'error',message:e.message,inputBytes}));
process.on('uncaughtException', e=>{log({event:'fatal',message:e.message,inputBytes});process.exitCode=1;});
