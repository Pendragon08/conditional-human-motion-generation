"""Upload verified idle software recovery files; token only in SSH stdin/RAM."""
import hashlib,http.client,json,os,pathlib,ssl,sys,time,urllib.parse

ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
WORK=pathlib.Path('/dev/shm/ouyanghao_dreamcontrol_backup_20261026')
OUT=ROOT/'receipts/idle_dreamcontrol_20261026'
REPO='Pendragon08/conditional-human-motion-generation'
TAG='inactive-dreamcontrol-overlay-20261026'
assert os.getuid()==1013
os.sched_setaffinity(0,{0});os.nice(19)
args=json.load(sys.stdin);token=args.pop('token');release_id=int(args['release_id'])
assert token
proof=json.loads((WORK/'archive_index.json').read_text())
assert proof['status']=='ARCHIVE_ALL_MEMBERS_AND_FILE_HASHES_VERIFIED'
state={'status':'UPLOADING_RECOVERY_SOFTWARE','files':{},'credentials_stored':False,'source_deleted':False}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def save():
 p=OUT/'upload_progress.json';q=p.with_suffix('.partial');q.write_text(json.dumps(state,indent=2));q.replace(p)

def request(host,path,method='GET',body=None,extra=None):
 connection=http.client.HTTPSConnection(host,timeout=180,context=ssl.create_default_context(),blocksize=1048576)
 headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
          'User-Agent':'textop-personal-software-recovery','X-GitHub-Api-Version':'2022-11-28'}
 if extra:headers.update(extra)
 try:
  connection.request(method,path,body=body,headers=headers);response=connection.getresponse();payload=response.read()
  assert response.status in [200,201], 'GitHub HTTP '+str(response.status)
  return json.loads(payload)
 finally:connection.close()

def release():
 value=request('api.github.com','/repos/'+REPO+'/releases/'+str(release_id))
 assert value['tag_name']==TAG
 return value

def find(item):
 asset=next((x for x in release()['assets'] if x['name']==item['name']),None)
 if asset:
  assert asset['state']=='uploaded' and asset['size']==item['bytes'] and asset.get('digest')=='sha256:'+item['sha256'], 'Existing asset is incomplete or mismatched; retain source'
 return asset

items=list(proof['parts'])
for name in ['archive_index.json','source_manifest.json','packages.json']:
 p=WORK/name;items.append({'name':name,'bytes':p.stat().st_size,'sha256':sha(p)})
try:
 release();save()
 for item in items:
  p=WORK/item['name'];assert p.parent==WORK and not p.is_symlink() and p.stat().st_uid==1013
  assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
  state['files'][item['name']]={'status':'PENDING','bytes':item['bytes']};save()
  asset=None
  for attempt in range(3):
   asset=find(item)
   if asset:break
   try:
    with p.open('rb') as source:
     class Reader:
      def __init__(self):self.count=0;self.hash=hashlib.sha256();self.last=0
      def read(self,n=-1):
       b=source.read(1048576 if n<0 else n);self.count+=len(b);self.hash.update(b)
       if self.count-self.last>=64*1024**2:
        state['files'][item['name']].update(status='STREAMING',bytes_sent=self.count);save();self.last=self.count
       return b
     body=Reader()
     request('uploads.github.com','/repos/'+REPO+'/releases/'+str(release_id)+'/assets?'+urllib.parse.urlencode({'name':item['name']}),
             'POST',body,{'Content-Type':'application/octet-stream','Content-Length':str(item['bytes'])})
     assert body.count==item['bytes'] and body.hash.hexdigest()==item['sha256']
    asset=find(item);assert asset;break
   except Exception as error:
    state['files'][item['name']].update(status='UPLOAD_FAILED_SOURCE_RETAINED',error_type=type(error).__name__,attempt=attempt+1);save()
    if attempt==2:raise
    time.sleep(3)
  state['files'][item['name']]={**item,'status':'UPLOADED_SIZE_AND_SHA_VERIFIED','asset_id':asset['id']};save()
 verified={'status':'ALL_GITHUB_SOFTWARE_ASSET_DIGESTS_VERIFIED','repository':REPO,'tag':TAG,'release_id':release_id,
           'release_url':release()['html_url'],'items':items,'credentials_stored':False,'source_deleted':False}
 (OUT/'upload_verified.json').write_text(json.dumps(verified,indent=2))
 state['status']=verified['status'];save();print('IDLE_SOFTWARE_RECOVERY_UPLOAD_VERIFIED',flush=True)
except Exception as error:
 state.update(status='UPLOAD_FAILED_SOURCE_RETAINED',error_type=type(error).__name__);save();raise
finally:del token
