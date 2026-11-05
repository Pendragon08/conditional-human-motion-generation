// Preserve nonactive project-derived previews and a completed diagnostic checkpoint.
// Verify every streamed tar member against the server source twice before any unlink.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),cp=require('child_process');
const L=path.resolve(__dirname,'..'),A=path.join(L,'parallel_archive_20261104');fs.mkdirSync(A,{recursive:true});
const HOST='10.147.17.32',ROOT='/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029';
const remote=s=>cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3','-'],{input:s,encoding:'utf8',maxBuffer:12*1024*1024,timeout:120000});
const inventory=JSON.parse(remote(String.raw`import hashlib,json,os
from pathlib import Path
R=Path('ROOT');folder=R/'omni_variants_gpu_20261101';diagnostic=R/'stage3_supermimic29_20261103/parallel_pick500_20261104/integration500_v1/resume_final.pt'
assert json.loads((diagnostic.parent/'status.json').read_text())['status']=='interaction_budget_completed'
paths=sorted(folder.glob('*.npz'))+[diagnostic];assert len(paths)>1000
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
rows=[]
for p in paths:
 assert p.is_file() and not p.is_symlink() and R in p.resolve().parents
 s=p.stat();rows.append({'remote_path':str(p),'member':str(p.relative_to(R)),'bytes':s.st_size,'sha256':sha(p),'inode':s.st_ino,'device':s.st_dev})
targets={r['remote_path'] for r in rows}
for proc in Path('/proc').glob('[0-9]*'):
 try:
  for fd in (proc/'fd').iterdir():assert os.path.realpath(fd) not in targets,'source file is open'
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps({'root':str(R),'rows':rows,'bytes':sum(r['bytes'] for r in rows),'scope':'old derived Omni previews and completed diagnostic; no raw source/current model/checkpoint touched'}))
`.replaceAll('ROOT',ROOT)));
const volume=fs.statfsSync(A),floor=1024**3+100*1024**2;
const overhead=inventory.rows.length*2048+10240;
if(Number(volume.bavail*volume.bsize)-inventory.bytes-overhead<floor)throw Error('Local retained-space floor would be crossed');
fs.writeFileSync(path.join(A,'inventory.json'),JSON.stringify(inventory,null,2),{flag:'wx'});
const tar=path.join(A,'inactive_previews_and_diagnostic.tar');
const input=Buffer.from(JSON.stringify(inventory.rows)).toString('base64');
const python=String.raw`import base64,json,tarfile,sys
from pathlib import Path
rows=json.loads(base64.b64decode('PAYLOAD'))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|',format=tarfile.USTAR_FORMAT) as tf:
 for r in rows:
  p=Path(r['remote_path']);assert p.stat().st_size==r['bytes'];tf.add(str(p),arcname=r['member'],recursive=False)
`.replace('PAYLOAD',input);
const fd=fs.openSync(tar,'wx');
const streamed=cp.spawnSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3','-'],{input:python,stdio:['pipe',fd,'pipe'],timeout:300000});fs.closeSync(fd);
if(streamed.status!==0)throw Error('Preserved incomplete archive: '+streamed.stderr);
function verify(){
 const fd=fs.openSync(tar,'r'),actual=new Map();let pos=0;
 while(true){
  const header=Buffer.alloc(512);if(fs.readSync(fd,header,0,512,pos)!==512)throw Error('Truncated tar');
  if(header.every(x=>x===0))break;
  const str=(a,b)=>header.subarray(a,b).toString('utf8').replace(/\0.*$/s,'');
  const name=str(0,100),prefix=str(345,500),member=(prefix?prefix+'/':'')+name;
  const size=parseInt(str(124,136).trim(),8),kind=str(156,157);if(kind!=='0'&&kind!=='')throw Error('Unexpected nonregular tar member');
  const h=crypto.createHash('sha256'),b=Buffer.alloc(4*1024**2);let remain=size,offset=pos+512;
  while(remain){const n=fs.readSync(fd,b,0,Math.min(remain,b.length),offset);if(n<=0)throw Error('Truncated member');h.update(b.subarray(0,n));offset+=n;remain-=n;}
  if(actual.has(member))throw Error('Duplicate tar member');actual.set(member,{bytes:size,sha256:h.digest('hex')});pos+=512+Math.ceil(size/512)*512;
 }
 fs.closeSync(fd);if(actual.size!==inventory.rows.length)throw Error('Tar count mismatch');
 for(const r of inventory.rows){const a=actual.get(r.member);if(!a||a.bytes!==r.bytes||a.sha256!==r.sha256)throw Error('Archive member differs: '+r.member);}
 return true;
}
verify();verify();
const tarHash=crypto.createHash('sha256').update(fs.readFileSync(tar)).digest('hex');
const manifest={...inventory,archive_path:tar,archive_sha256:tarHash,archive_bytes:fs.statSync(tar).size,verified_twice:true,created_at:new Date().toISOString()};
fs.writeFileSync(path.join(A,'manifest.json'),JSON.stringify(manifest,null,2),{flag:'wx'});
const payload=Buffer.from(JSON.stringify(manifest)).toString('base64');
const result=JSON.parse(remote(String.raw`import base64,json,hashlib,os,shutil
from pathlib import Path
m=json.loads(base64.b64decode('PAYLOAD'));R=Path('ROOT');rows=m['rows'];targets={r['remote_path'] for r in rows}
assert m['verified_twice'] and m['archive_path'].endswith('.tar')
for r in rows:
 p=Path(r['remote_path']);assert R in p.resolve().parents and not p.is_symlink();s=p.stat()
 assert (s.st_size,s.st_ino,s.st_dev)==(r['bytes'],r['inode'],r['device'])
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 assert h.hexdigest()==r['sha256']
for proc in Path('/proc').glob('[0-9]*'):
 try:
  for fd in (proc/'fd').iterdir():assert os.path.realpath(fd) not in targets,'source file is open'
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
out=R/'checkpoint_archival_20261104/parallel_inactive_manifest.json';out.open('x').write(json.dumps(m,indent=2))
before=shutil.disk_usage('/data').free
for r in rows:Path(r['remote_path']).unlink()
marker={'archive':m['archive_path'],'manifest':str(out),'all_file_bytes_preserved':True,'local_archive_sha256':m['archive_sha256']}
(R/'omni_variants_gpu_20261101/npz.cold_archived.json').open('x').write(json.dumps(marker,indent=2))
p=Path(rows[-1]['remote_path']);p.with_suffix(p.suffix+'.archived.json').open('x').write(json.dumps(marker,indent=2))
print(json.dumps({'files':len(rows),'source_bytes_preserved':sum(r['bytes'] for r in rows),'free_before':before,'free_after':shutil.disk_usage('/data').free,'archive':m['archive_path'],'manifest':str(out)}))
`.replace('PAYLOAD',payload).replaceAll('ROOT',ROOT)));
fs.writeFileSync(path.join(A,'receipt.json'),JSON.stringify(result,null,2),{flag:'wx'});console.log(JSON.stringify(result));
