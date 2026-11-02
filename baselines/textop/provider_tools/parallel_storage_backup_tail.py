"""Upload only tail pieces4-6 in parallel while the main worker handles1-3.

No deletions or cleanup authorization. The main worker must independently
verify every digest before it can advance to source cleanup.
"""
import concurrent.futures,json,pathlib,threading,time
from finish_personal_storage_cleanup import DEST,REPOSITORY,TAG,gh,stream_part_to_github
index=json.loads((DEST/'archive_index.json').read_text(encoding='utf-8'))
assert len(index['parts'])==6
path=DEST/'parallel_tail_upload.json';assert not path.exists(),'Tail upload already exists'
lock=threading.Lock();state={'status':'UPLOADING_TAIL_ONLY','parts':{}}
def record(name,status,**extra):
 with lock:
  state['parts'][name]={'status':status,**extra};path.write_text(json.dumps(state,indent=2),encoding='utf-8')
  print(name,status,flush=True)
def worker(item):
 try:
  release=json.loads(gh('api','repos/'+REPOSITORY+'/releases/tags/'+TAG))
  asset=next((x for x in release['assets'] if x['name']==item['name']),None)
  if asset is None:
   record(item['name'],'STREAMING');stream_part_to_github(item,release['id'])
   release=json.loads(gh('api','repos/'+REPOSITORY+'/releases/tags/'+TAG));asset=next(x for x in release['assets'] if x['name']==item['name'])
  assert asset['state']=='uploaded' and asset['size']==item['bytes'] and asset.get('digest')=='sha256:'+item['sha256']
  record(item['name'],'UPLOADED_DIGEST_VERIFIED',asset_id=asset['id'],bytes=item['bytes'],sha256=item['sha256'])
 except BaseException as error:record(item['name'],'FAILED_SOURCE_RETAINED',error=repr(error));raise
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for future in concurrent.futures.as_completed([pool.submit(worker,item) for item in index['parts'][3:]]):future.result()
state['status']='ALL_TAIL_PIECES_UPLOADED_DIGEST_VERIFIED';path.write_text(json.dumps(state,indent=2),encoding='utf-8')
