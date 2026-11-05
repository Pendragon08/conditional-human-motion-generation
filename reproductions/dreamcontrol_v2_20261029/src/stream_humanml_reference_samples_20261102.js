// Stream three small diagnostic members from a third-party uncompressed tar.
// The mirror retains upstream noncommercial licensing; this is not training data.
const https = require('https'), fs = require('fs'), path = require('path'), crypto = require('crypto');
const out = path.resolve(process.argv[2]);
const url = 'https://huggingface.co/datasets/zyyy12138/motion-smd-data/resolve/113b64cb43c0e0c25ec09f93542ec92f575ad148/humanml3d/new_joints.tar';
const expectedSize = 1111336960, targets = new Set(['004271.npy','007555.npy','012314.npy']);
const official = '739df5e2e025d21ae5ce01ea7e0a8d3038448a46e88865d95a638e6242ee8b6b';
fs.mkdirSync(out, {recursive:false});
const log = x => fs.appendFileSync(path.join(out,'events.jsonl'), JSON.stringify(x)+'\n');
let inputBytes=0, members=0, lastProgress=0, buffer=Buffer.alloc(0), remain=0, pad=0, selected=null, parts=[], found={};
function fail(message) { log({event:'failure',message,inputBytes}); console.error(message); process.exitCode=1; }
function get(u,depth=0) {
  if(depth>5) return fail('too many redirects');
  https.get(u,{headers:{'Accept-Encoding':'identity'}},res=>{
    if ([301,302,303,307,308].includes(res.statusCode)) { const next=new URL(res.headers.location,u).href;res.resume();get(next,depth+1);return; }
    if (res.statusCode!==200 || Number(res.headers['content-length'])!==expectedSize) { res.resume();fail('bad HTTP status or content length');return; }
    log({event:'response',status:res.statusCode,length:expectedSize});
    res.on('data',chunk=>{
      inputBytes+=chunk.length;
      if (inputBytes>expectedSize) {res.destroy();fail('input exceeds declared size');return;}
      if(inputBytes-lastProgress>=64*1024**2){lastProgress=inputBytes;log({event:'progress',inputBytes,members});}
      buffer=Buffer.concat([buffer,chunk]);
      while(true){
        if(remain){const n=Math.min(remain,buffer.length);if(selected)parts.push(buffer.subarray(0,n));buffer=buffer.subarray(n);remain-=n;if(remain)return;
          if(selected){const bytes=Buffer.concat(parts), digest=crypto.createHash('sha256').update(bytes).digest('hex');
            fs.writeFileSync(path.join(out,selected),bytes,{flag:'wx'});found[selected]={member:currentName,bytes:bytes.length,sha256:digest};
            log({event:'extracted',sample:selected,bytes:bytes.length,sha256:digest,inputBytes});targets.delete(selected);selected=null;parts=[];
            if(!targets.size){const report={revision:'113b64cb43c0e0c25ec09f93542ec92f575ad148',archive_size:expectedSize,inputBytes,members,found,
              publisher_archive_sha256:'a261219891b1003b88c66f4d94ce46b6e192bc857fbaff8bcfbb63a908803219',full_archive_sha256_verified:false,
              official_sample_sha256:official,one_example_bytes_match_official:found['012314.npy'].sha256===official,training_ready:false,
              limitations:['Third-party mirror, not an official release.','One matching sample does not authenticate other members.',
                'Normalized human joints are not raw markers and do not prove real time or licensing.']};
              fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2),{flag:'wx'});log({event:'complete',inputBytes});
              console.log(JSON.stringify(report));res.destroy();return;}
          }
        }
        if(pad){const n=Math.min(pad,buffer.length);buffer=buffer.subarray(n);pad-=n;if(pad)return;}
        if(buffer.length<512)return;
        const header=buffer.subarray(0,512);buffer=buffer.subarray(512);
        const name=header.subarray(0,100).toString().split('\0')[0];if(!name){res.destroy();fail('end of tar before all samples');return;}
        const prefix=header.subarray(345,500).toString().split('\0')[0];currentName=prefix?prefix+'/'+name:name;
        const raw=header.subarray(124,136).toString().replace(/\0/g,'').trim(),size=parseInt(raw||'0',8);members++;
        if(!Number.isSafeInteger(size)||size<0){res.destroy();fail('invalid tar header size');return;}
        remain=size;pad=(512-size%512)%512;
        const base=path.posix.basename(currentName),type=header[156];
        if(targets.has(base)&&(type===0||type===48)){
          if(size<=0||size>1024*1024){res.destroy();fail('invalid target size');return;}
          selected=base;parts=[];
        }
      }
    });
    res.on('error',e=>fail(e.message));
    res.on('end',()=>{if(targets.size)fail('stream ended before all targets');});
  }).on('error',e=>fail(e.message));
}
let currentName='';get(url);
