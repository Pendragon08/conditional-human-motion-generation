// Cold-archive only explicitly selected intermediates from the completed native2000 run.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process');
const group=process.argv[2];
const L=path.resolve(__dirname,'..');
const destinations={
 native2000_intermediates_c:path.join(L,'training_space_archive_20261105'),
 native2000_intermediates_d:'D:/DreamControl_v2_checkpoint_archive_20261104'
};
if(!Object.hasOwn(destinations,group))throw Error('Select an allowed native2000 archive group');
const attempt=process.argv[3];
if(!/^v[1-9][0-9]*$/.test(attempt))throw Error('Explicit versioned attempt required');
const A=path.join(destinations[group],group,attempt);
fs.mkdirSync(A,{recursive:true});
if(fs.readdirSync(A).length)throw Error('Existing archive attempt: inspect before continuing');
const HOST='10.147.17.32',ROOT='/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029';
const ssh=['-o','BatchMode=yes','-o','ConnectTimeout=15',HOST];
const remote=code=>{
 const r=cp.spawnSync('ssh',[...ssh,'nice','-n','10','python3','-'],{input:code,encoding:'utf8',maxBuffer:24*1024**2,timeout:120000});
 if(r.status!==0)throw Error(r.stderr||r.stdout||'SSH failed');return r.stdout;
};
const inventory=JSON.parse(remote(String.raw`
import pathlib,json,hashlib,os,fcntl,signal
R=pathlib.Path('ROOT');group='GROUP'
N=R/'stage3_supermimic29_20261103/native_universal_20261104'
run=N/'formal2000_sequence_v1';main=run/'main_resume_v1/logs'
active=json.loads((N/'active_native2000_controller.json').read_text())
assert active['controller_status']==str(run/'recovered_completion_status.json')
state=json.loads(pathlib.Path(active['controller_status']).read_text())
assert state['status']=='native2000_training_and_body_tracking_evaluation_completed'
assert not pathlib.Path('/proc',str(active['pid'])).exists()
assert (main/'last.pt').is_file() and (main/'ckpt_028748.pt').is_file()
if group=='native2000_intermediates_c':
 paths=[main/f'ckpt_{i:06}.pt' for i in range(2000,26000,2000)]
 paths.append(run/'early600_v1/logs/last.pt')
else:
 paths=[main/'ckpt_026000.pt',main/'ckpt_028000.pt']
expected={str(run/'early600_v1/logs/last.pt'):'86f4c61f9289abbb0201c09b0a75fbdc93a99352adabed22573a9730e22ee1ea'}
scope='completed native2000 intermediate snapshots only; final last.pt and ckpt_028748.pt stay on server'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def closed(p):
 fd=os.open(str(p),os.O_RDWR)
 try:
  fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK);assert fcntl.fcntl(fd,fcntl.F_GETLEASE)==fcntl.F_WRLCK
 finally:
  fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK);os.close(fd)
rows=[]
for p in paths:
 assert p.is_file() and not p.is_symlink() and R in p.resolve().parents
 s=p.stat();assert s.st_uid==os.getuid() and s.st_nlink==1;closed(p);h=digest(p)
 if str(p) in expected:assert h==expected[str(p)]
 rows.append({'remote_path':str(p),'member':str(p.relative_to(R)),'bytes':s.st_size,'sha256':h,
              'inode':str(s.st_ino),'device':str(s.st_dev),'mtime_ns':str(s.st_mtime_ns),'uid':s.st_uid})
print(json.dumps({'root':str(R),'group':group,'rows':rows,'bytes':sum(r['bytes'] for r in rows),'scope':scope}))
`.replaceAll('ROOT',ROOT).replaceAll('GROUP',group)));
const floor=1024**3+100*1024**2,stats=fs.statfsSync(A),overhead=inventory.rows.length*2048+10240;
if(Number(stats.bavail*stats.bsize)-inventory.bytes-overhead<floor)throw Error('Local retained-space floor would be crossed');
fs.writeFileSync(path.join(A,'inventory.json'),JSON.stringify(inventory,null,2),{flag:'wx'});
const archive=path.join(A,'payload.tar'),fd=fs.openSync(archive,'wx');
const rows64=Buffer.from(JSON.stringify(inventory.rows)).toString('base64');
const pack=String.raw`import base64,json,pathlib,tarfile,sys
rows=json.loads(base64.b64decode('ROWS'))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|',format=tarfile.USTAR_FORMAT) as tf:
 for r in rows:
  p=pathlib.Path(r['remote_path']);s=p.stat();assert (s.st_size,str(s.st_ino),str(s.st_dev),str(s.st_mtime_ns))==(r['bytes'],r['inode'],r['device'],r['mtime_ns'])
  tf.add(str(p),arcname=r['member'],recursive=False)
`.replace('ROWS',rows64);
const streamed=cp.spawnSync('ssh',[...ssh,'nice','-n','10','python3','-'],{input:pack,stdio:['pipe',fd,'pipe'],timeout:600000});
fs.closeSync(fd);if(streamed.status!==0)throw Error('Incomplete copy retained: '+streamed.stderr);
function verify(){
 const fd=fs.openSync(archive,'r');let pos=0;const actual=new Map(),buffer=Buffer.alloc(4*1024**2);
 try{
  while(true){
   const head=Buffer.alloc(512);if(fs.readSync(fd,head,0,512,pos)!==512)throw Error('Truncated tar header');
   if(head.every(x=>x===0))break;
   const str=(a,b)=>head.subarray(a,b).toString().replace(/\0.*$/s,'');
   let checksum=0;for(let i=0;i<512;i++)checksum+=i>=148&&i<156?32:head[i];
   if(checksum!==parseInt(str(148,156).trim(),8))throw Error('Tar header checksum');
   const member=(str(345,500)?str(345,500)+'/':'')+str(0,100),bytes=parseInt(str(124,136).trim(),8);
   if(!['','0'].includes(str(156,157))||actual.has(member))throw Error('Nonregular/duplicate member');
   const hash=crypto.createHash('sha256');let left=bytes,offset=pos+512;
   while(left){const n=fs.readSync(fd,buffer,0,Math.min(left,buffer.length),offset);if(n<=0)throw Error('Truncated member');hash.update(buffer.subarray(0,n));left-=n;offset+=n;}
   actual.set(member,{bytes,sha256:hash.digest('hex')});pos+=512+Math.ceil(bytes/512)*512;
  }
 }finally{fs.closeSync(fd);}
 if(actual.size!==inventory.rows.length)throw Error('Archive member count');
 for(const row of inventory.rows){const got=actual.get(row.member);if(!got||got.bytes!==row.bytes||got.sha256!==row.sha256)throw Error('Member mismatch: '+row.member);}
}
async function digest(file){const h=crypto.createHash('sha256');for await(const block of fs.createReadStream(file))h.update(block);return h.digest('hex');}
(async()=>{
 verify();verify();
 const free=fs.statfsSync(A);
 if(Number(free.bavail*free.bsize)<floor)throw Error('Local retained-space floor crossed during transfer');
 const manifest={...inventory,archive_path:archive,archive_sha256:await digest(archive),archive_bytes:fs.statSync(archive).size,
                 verified_twice:true,created_at:new Date().toISOString(),source_release_completed:false,
                 archiver_sha256:await digest(__filename),attempt};
 fs.writeFileSync(path.join(A,'manifest.json'),JSON.stringify(manifest,null,2),{flag:'wx'});
 const payload=Buffer.from(JSON.stringify(manifest)).toString('base64');
 const result=JSON.parse(remote(String.raw`import base64,json,pathlib,hashlib,os,fcntl,signal,shutil,time
m=json.loads(base64.b64decode('PAYLOAD'));R=pathlib.Path('ROOT');rows=m['rows'];group=m['group']
assert m['verified_twice'] and len({r['member'] for r in rows})==len(rows)
out=R/'checkpoint_archival_20261104'/('training_space_'+group+'_'+m['attempt']+'_manifest.json');assert not out.exists()
out.open('x').write(json.dumps(m,indent=2));journal=out.with_name(out.stem+'_released.jsonl');before=shutil.disk_usage('/data').free
broken=[False];signal.signal(signal.SIGIO,lambda *_:broken.__setitem__(0,True));freed=0
for r in rows:
 p=pathlib.Path(r['remote_path']);assert not p.is_symlink() and R in p.resolve().parents
 run=R/'stage3_supermimic29_20261103/native_universal_20261104/formal2000_sequence_v1'
 main=run/'main_resume_v1/logs'
 if group=='native2000_intermediates_c':
  allowed=p in {main/f'ckpt_{i:06}.pt' for i in range(2000,26000,2000)} or p==run/'early600_v1/logs/last.pt'
 else:
  assert group=='native2000_intermediates_d'
  allowed=p in {main/'ckpt_026000.pt',main/'ckpt_028000.pt'}
 assert allowed and p.stat().st_uid==os.getuid() and p.stat().st_nlink==1
 assert not pathlib.Path(str(p)+'.archived.json').exists()
 s=p.stat();assert (s.st_size,str(s.st_ino),str(s.st_dev),str(s.st_mtime_ns))==(r['bytes'],r['inode'],r['device'],r['mtime_ns'])
 fd=os.open(str(p),os.O_RDWR);broken[0]=False
 try:
  fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK);assert fcntl.fcntl(fd,fcntl.F_GETLEASE)==fcntl.F_WRLCK
  h=hashlib.sha256()
  for b in iter(lambda:os.read(fd,4*1024**2),b''):h.update(b)
  assert h.hexdigest()==r['sha256'] and not broken[0]
  s=p.stat();assert (s.st_size,str(s.st_ino),str(s.st_dev),str(s.st_mtime_ns))==(r['bytes'],r['inode'],r['device'],r['mtime_ns'])
  assert fcntl.fcntl(fd,fcntl.F_GETLEASE)==fcntl.F_WRLCK and not broken[0];p.unlink()
 finally:
  fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK);os.close(fd)
 freed+=r['bytes']
 with journal.open('a') as f:f.write(json.dumps({'member':r['member'],'bytes':r['bytes'],'sha256':r['sha256'],'kernel_exclusive_lease_held_through_release':True})+'\n')
 if p.suffix=='.pt':
  marker={'manifest':str(out),'archive':m['archive_path'],'source_sha256':r['sha256'],'source_bytes':r['bytes'],'complete_file_bytes_preserved':True}
  pathlib.Path(str(p)+'.archived.json').open('x').write(json.dumps(marker,indent=2))
receipt={'status':'complete_verified_cold_archive_source_release_finished','group':group,'files':len(rows),
 'preserved_source_bytes':freed,'local_archive':m['archive_path'],'archive_sha256':m['archive_sha256'],
 'server_manifest':str(out),'released_journal':str(journal),'free_before':before,'free_after':shutil.disk_usage('/data').free,
 'no_other_open_FD_proof':'per-file kernel F_WRLCK lease held through final complete SHA/source identity check and unlink',
 'raw_sources_current_final_models_unchanged':True}
out.with_name(out.stem+'_receipt.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
`.replace('PAYLOAD',payload).replaceAll('ROOT',ROOT)));
 fs.writeFileSync(path.join(A,'receipt.json'),JSON.stringify(result,null,2),{flag:'wx'});
 console.log(JSON.stringify(result));
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
