// Restore one exact archive member; keep the verified local tar and refuse overwrite.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),cp=require('child_process');
const A=path.resolve(__dirname,'../parallel_archive_20261104'),m=JSON.parse(fs.readFileSync(path.join(A,'manifest.json'),'utf8'));
const member=process.argv[2],row=m.rows.find(x=>x.member===member);if(!row)throw Error('Pass one exact member from parallel_archive_20261104/manifest.json');
const fd=fs.openSync(m.archive_path,'r');let pos=0,offset;
while(true){const b=Buffer.alloc(512);if(fs.readSync(fd,b,0,512,pos)!==512||b.every(x=>x===0))break;
 const str=(a,z)=>b.subarray(a,z).toString().replace(/\0.*$/s,'');const n=(str(345,500)?str(345,500)+'/':'')+str(0,100),size=parseInt(str(124,136).trim(),8);
 if(n===member){if(size!==row.bytes)throw Error('Archive size differs');offset=pos+512;break;}pos+=512+Math.ceil(size/512)*512;
}fs.closeSync(fd);if(offset===undefined)throw Error('Member missing');
const h=crypto.createHash('sha256');
(async()=>{
 for await(const b of fs.createReadStream(m.archive_path,{start:offset,end:offset+row.bytes-1}))h.update(b);
 if(h.digest('hex')!==row.sha256)throw Error('Local member SHA mismatch');
 const payload=Buffer.from(JSON.stringify(row)).toString('base64');
 const script=String.raw`import base64,json,hashlib,shutil,sys,os
from pathlib import Path
r=json.loads(base64.b64decode('PAYLOAD'));p=Path(r['remote_path']);root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029').resolve()
assert root in p.resolve().parents and not p.exists()
assert shutil.disk_usage('/data').free-r['bytes']>=8*1024**3+456*1024**2
tmp=p.with_name(p.name+'.restore_'+r['sha256'][:12]);h=hashlib.sha256();n=0
with tmp.open('xb') as f:
 while True:
  b=sys.stdin.buffer.read(4*1024**2)
  if not b:break
  f.write(b);h.update(b);n+=len(b)
assert n==r['bytes'] and h.hexdigest()==r['sha256'];os.link(tmp,p);tmp.unlink()
print(json.dumps({'restored':str(p),'sha256':r['sha256'],'local_archive_retained':True}))
`.replace('PAYLOAD',payload);
 const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
 const child=cp.spawn('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -c '+quote(script)],{stdio:['pipe','inherit','inherit']});
 fs.createReadStream(m.archive_path,{start:offset,end:offset+row.bytes-1}).pipe(child.stdin);
 child.on('close',code=>{process.exitCode=code;});
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
