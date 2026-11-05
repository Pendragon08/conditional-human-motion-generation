// Restore an entire cold group or one exact member. Keep the local backup.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process');
const group=process.argv[2],attempt=process.argv[3],member=process.argv[4];
if(!['native2000_intermediates_c','native2000_intermediates_d'].includes(group))throw Error('Select an archive group');
if(!/^v[1-9][0-9]*$/.test(attempt))throw Error('Explicit versioned attempt required');
const base=group==='native2000_intermediates_c'
 ? path.resolve(__dirname,'../training_space_archive_20261105')
 : 'D:/DreamControl_v2_checkpoint_archive_20261104';
const A=path.join(base,group,attempt),m=JSON.parse(fs.readFileSync(path.join(A,'manifest.json')));
if(m.group!==group||m.attempt!==attempt||!m.verified_twice||
   path.resolve(m.archive_path)!==path.resolve(A,'payload.tar'))throw Error('Archive binding mismatch');
if(member&&!m.rows.some(r=>r.member===member))throw Error('Pass one exact archive member or omit it to restore all');
const script=String.raw`import sys,json,tarfile,pathlib,hashlib,shutil,os
m=json.loads(sys.stdin.buffer.readline());R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');wanted=m.get('restore_member')
rows={r['member']:r for r in m['rows']};restored=0;existing=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tf:
 for entry in tf:
  assert entry.isfile() and entry.name in rows
  if wanted and entry.name!=wanted:continue
  r=rows[entry.name];p=pathlib.Path(r['remote_path']);assert R in p.resolve().parents and entry.size==r['bytes']
  content=tf.extractfile(entry);h=hashlib.sha256();n=0;tmp=p.with_name(p.name+'.restoring_'+r['sha256'][:10]);target=None
  if not p.exists():
   assert shutil.disk_usage('/data').free-r['bytes']>=8*1024**3+456*1024**2
   target=tmp.open('xb')
  try:
   for block in iter(lambda:content.read(4*1024**2),b''):
    h.update(block);n+=len(block)
    if target:target.write(block)
  finally:
   if target:target.flush();os.fsync(target.fileno());target.close()
  assert n==r['bytes'] and h.hexdigest()==r['sha256']
  if p.exists():
   other=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(4*1024**2),b''):other.update(block)
   assert other.hexdigest()==r['sha256'],'refuse to overwrite different existing content';existing+=1
  else:os.link(tmp,p);tmp.unlink();restored+=1
print(json.dumps({'restored':restored,'existing_verified':existing,'local_backup_retained':True}))
`;
const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
(async()=>{
 const h=crypto.createHash('sha256');for await(const b of fs.createReadStream(m.archive_path))h.update(b);
 if(h.digest('hex')!==m.archive_sha256)throw Error('Local archive SHA mismatch');
 const child=cp.spawn('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','nice -n 10 python3 -c '+quote(script)],{stdio:['pipe','inherit','inherit']});
 child.stdin.write(JSON.stringify({...m,restore_member:member||null})+'\n');
 fs.createReadStream(m.archive_path).pipe(child.stdin);
 child.on('close',code=>{process.exitCode=code;});
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
