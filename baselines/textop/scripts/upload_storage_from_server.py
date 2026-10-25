"""Direct official GitHub uploads of already-verified software archive parts.

Token arrives only over encrypted SSH stdin and stays in memory. No credential
is logged/stored. No source deletion, no motion/model assets, no proxy changes.
"""
import concurrent.futures,hashlib,http.client,json,os,pathlib,ssl,sys,threading,time,urllib.parse
assert os.getuid()==1013
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
args=json.load(sys.stdin);token=args.pop('token');assert token
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
WORK=pathlib.Path('/dev/shm/ouyanghao_storage_20261025');REPO='Pendragon08/conditional-human-motion-generation'
proof=json.loads((WORK/'backup_index.json').read_text());release_id=int(args['release_id'])
assert release_id==423171904 and proof['status']=='ARCHIVE_CRC_AND_EVERY_FILE_SHA_VERIFIED'
parts=[x for x in proof['parts'] if x['name'] in args['parts']];assert len(parts)==3
assert set(args['parts'])=={'inactive_engineai_and_caches.tar.gz.part002','inactive_engineai_and_caches.tar.gz.part003','inactive_engineai_and_caches.tar.gz.part005'}
lock=threading.Lock();state={'status':'DIRECT_UPLOAD_RUNNING','parts':{},'credentials_stored':False,'source_files_deleted':0}
def record(name,**values):
 with lock:
  state['parts'].setdefault(name,{}).update(values)
  (ROOT/'receipts/storage_direct_upload_progress.json').write_text(json.dumps(state,indent=2))
  if values.get('status'):print(json.dumps({'part':name,**values}),flush=True)
def request(host,path,method='GET',body=None,extra=None):
 c=http.client.HTTPSConnection(host,timeout=90,context=ssl.create_default_context(),blocksize=1048576)
 headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'personal-storage-recovery',
          'X-GitHub-Api-Version':'2022-11-28'}
 if extra:headers.update(extra)
 try:
  c.request(method,path,body=body,headers=headers);r=c.getresponse();payload=r.read()
  assert r.status in [200,201,204],'GitHub HTTP'+str(r.status)
  return json.loads(payload) if payload else {}
 finally:c.close()
def release():
 for i in range(5):
  try:return request('api.github.com','/repos/'+REPO+'/releases/'+str(release_id))
  except Exception:
   if i==4:raise
   time.sleep(2**i)
def find(item):
 asset=next((x for x in release()['assets'] if x['name']==item['name']),None)
 if asset and asset['state']=='uploaded' and asset['size']==item['bytes'] and asset.get('digest')=='sha256:'+item['sha256']:return asset
 if asset:raise RuntimeError('Existing incomplete backup asset needs explicit inspection: '+item['name'])
 return None
def worker(item):
 path=WORK/item['name'];assert path.parent==WORK and path.stat().st_uid==1013 and not path.is_symlink()
 assert path.stat().st_size==item['bytes']
 for attempt in range(3):
  asset=find(item)
  if asset:break
  record(item['name'],status='STREAMING_DIRECT_FROM_SERVER',attempt=attempt+1)
  try:
   with path.open('rb') as source:
    class Reader:
     def __init__(self):self.count=0;self.digest=hashlib.sha256();self.last=0
     def read(self,n=-1):
      b=source.read(1048576 if n<0 else n);self.count+=len(b);self.digest.update(b)
      if self.count-self.last>=64*1024**2:record(item['name'],bytes_sent=self.count);self.last=self.count
      return b
    body=Reader()
    request('uploads.github.com','/repos/'+REPO+'/releases/'+str(release_id)+'/assets?'+urllib.parse.urlencode({'name':item['name']}),
            method='POST',body=body,extra={'Content-Type':'application/octet-stream','Content-Length':str(item['bytes'])})
    assert body.count==item['bytes'] and body.digest.hexdigest()==item['sha256']
   asset=find(item);assert asset;break
  except Exception as error:
   record(item['name'],status='RETRYABLE_UPLOAD_ERROR_SOURCE_RETAINED',error_type=type(error).__name__)
   if attempt==2:raise
   time.sleep(3)
 assert asset['size']==item['bytes'] and asset.get('digest')=='sha256:'+item['sha256']
 record(item['name'],status='UPLOADED_SIZE_AND_DIGEST_VERIFIED',bytes_sent=item['bytes'],asset_id=asset['id'],sha256=item['sha256'])
 return asset['id']
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for future in concurrent.futures.as_completed([pool.submit(worker,x) for x in parts]):future.result()
 state['status']='ALL_DIRECT_PARTS_UPLOADED_DIGEST_VERIFIED'
 (ROOT/'receipts/storage_direct_upload_progress.json').write_text(json.dumps(state,indent=2));print('STORAGE_DIRECT_UPLOAD_VERIFIED',flush=True)
finally:del token
