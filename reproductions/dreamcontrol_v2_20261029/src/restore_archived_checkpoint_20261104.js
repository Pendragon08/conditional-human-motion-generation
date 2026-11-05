// Restore one explicitly indexed inactive checkpoint without changing its bytes.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{execFileSync}=require('child_process');
const L=path.resolve(__dirname,'..'),manifest=path.join(L,'checkpoint_archive_20261104/manifest.json');
const relative=process.argv[2];if(!relative)throw Error('Pass the exact relative_path from the archive manifest');
const j=JSON.parse(fs.readFileSync(manifest,'utf8')),row=j.rows.find(r=>r.relative_path===relative);if(!row||!row.local_verified)throw Error('Checkpoint not in verified archive');
const HOST='10.147.17.32',P='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python';
const remote=script=>execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,P+' -'],{input:script,encoding:'utf8',timeout:60000});
const encoded=Buffer.from(JSON.stringify(row)).toString('base64');
(async()=>{
 const h=crypto.createHash('sha256');for await(const b of fs.createReadStream(row.local_path))h.update(b);
 if(h.digest('hex')!==row.sha256||fs.statSync(row.local_path).size!==row.bytes)throw Error('Archive bytes do not match; no restore');
 const check=JSON.parse(remote(String.raw`import base64,json,hashlib,shutil
from pathlib import Path
r=json.loads(base64.b64decode('PAYLOAD'));p=Path(r['remote_path']);root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029').resolve()
assert p.resolve().is_relative_to(root)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
if p.exists():
 assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'different existing checkpoint; overwrite refused'
 print(json.dumps({'already_restored':True}))
else:
 assert shutil.disk_usage('/data').free-r['bytes']>=8*1024**3+456*1024**2
 tmp=p.with_name(p.name+'.restore_'+r['sha256'][:12]+'.tmp');assert not tmp.exists()
 print(json.dumps({'already_restored':False,'temporary_path':str(tmp)}))
`.replace('PAYLOAD',encoded)));
 if(!check.already_restored){
  execFileSync('scp',['-q','-p','-o','BatchMode=yes','-o','ConnectTimeout=15',row.local_path,HOST+':'+check.temporary_path],{stdio:'inherit',timeout:300000});
  const payload=Buffer.from(JSON.stringify({...row,temporary_path:check.temporary_path})).toString('base64');
  remote(String.raw`import base64,json,hashlib,os
from pathlib import Path
r=json.loads(base64.b64decode('PAYLOAD'));p=Path(r['remote_path']);tmp=Path(r['temporary_path']);assert not p.exists()
h=hashlib.sha256()
with tmp.open('rb') as f:
 for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
assert h.hexdigest()==r['sha256'] and tmp.stat().st_size==r['bytes'];os.link(tmp,p);tmp.unlink()
print(json.dumps({'restored':str(p),'sha256':r['sha256']}))
`.replace('PAYLOAD',payload));
 }
 const receipt={restored:true,remote_path:row.remote_path,sha256:row.sha256,archive_copy_retained:row.local_path};
 fs.writeFileSync(path.join(L,'checkpoint_archive_20261104','restore_'+row.sha256+'.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
