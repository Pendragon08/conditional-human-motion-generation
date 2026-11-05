"""Read-only GitHub archive verification. No payload staging or persisted credentials."""
import hashlib,http.client,json,tarfile,time
from urllib.parse import urlsplit

REPO='Pendragon08/conditional-human-motion-generation'

def get_asset(token,asset_id):
    c=http.client.HTTPSConnection('api.github.com',timeout=90)
    try:
        c.request('GET',f'/repos/{REPO}/releases/assets/{int(asset_id)}',headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'DreamControl-archive-verifier','X-GitHub-Api-Version':'2022-11-28'})
        r=c.getresponse();body=r.read()
        if r.status!=200:raise RuntimeError('GitHub metadata HTTP '+str(r.status))
        return json.loads(body)
    finally:c.close()

def assert_asset(actual,expected):
    if actual['id']!=expected['id'] or actual['name']!=expected['name'] or actual['size']!=expected['bytes'] or actual.get('digest')!='sha256:'+expected['sha256'] or actual['state']!='uploaded':
        raise RuntimeError('GitHub asset identity/digest mismatch: '+expected['name'])

def download_range(token,asset_id,start,end,total):
    url=f'https://api.github.com/repos/{REPO}/releases/assets/{int(asset_id)}?readback_nonce={time.time_ns()}'
    for hop in range(5):
        u=urlsplit(url)
        allowed=u.hostname in {'api.github.com','github.com','release-assets.githubusercontent.com','objects.githubusercontent.com','github-releases.githubusercontent.com'} or (u.hostname or '').endswith('.s3.amazonaws.com')
        if u.scheme!='https' or not allowed:raise RuntimeError('Unsafe archive redirect')
        headers={'Accept':'application/octet-stream','User-Agent':'DreamControl-archive-verifier','Range':f'bytes={start}-{end}','Cache-Control':'no-cache'}
        if u.hostname=='api.github.com':headers.update({'Authorization':'Bearer '+token,'X-GitHub-Api-Version':'2022-11-28'})
        c=http.client.HTTPSConnection(u.hostname,timeout=30)
        c.request('GET',u.path+('?' +u.query if u.query else ''),headers=headers);r=c.getresponse()
        if r.status in [301,302,303,307,308]:url=r.getheader('Location');r.read();c.close();continue
        if r.status!=206 or r.getheader('Content-Range')!=f'bytes {start}-{end}/{total}' or int(r.getheader('Content-Length') or -1)!=end-start+1:
            code=r.status;c.close();raise RuntimeError('Range response identity mismatch HTTP '+str(code))
        return c,r
    raise RuntimeError('Too many range redirects')

class RangeHashReader:
    def __init__(self,token,asset,progress):
        self.token=token;self.asset=asset;self.expected_bytes=asset['bytes'];self.progress=progress;self.bytes=0;self.hash=hashlib.sha256();self.last=time.monotonic();self.conn=None;self.response=None;self.segment_end=-1;self.reconnects=0
    def close(self):
        if self.conn:self.conn.close()
        self.conn=None;self.response=None
    def read(self,n=-1):
        if self.bytes==self.expected_bytes:self.close();return b''
        if n<0:n=256*1024
        for retry in range(5):
            try:
                if self.response is None or self.bytes>self.segment_end:
                    self.close();self.segment_start=self.bytes;self.segment_end=min(self.expected_bytes-1,self.bytes+16*1024**2-1);self.segment_started=time.monotonic()
                    self.conn,self.response=download_range(self.token,self.asset['id'],self.bytes,self.segment_end,self.expected_bytes)
                elapsed=time.monotonic()-self.segment_started
                if elapsed>60 and (self.bytes-self.segment_start)/elapsed<256*1024:raise TimeoutError('Slow range connection')
                b=self.response.read(min(n,256*1024,self.segment_end-self.bytes+1))
                if not b:raise EOFError('Truncated range')
                self.bytes+=len(b);self.hash.update(b)
                if self.progress and time.monotonic()-self.last>=30:self.progress(self.bytes);self.last=time.monotonic()
                return b
            except (OSError,EOFError,http.client.HTTPException):
                self.close();self.reconnects+=1
                if retry==4:raise
        raise RuntimeError('Range retry exhausted')

def inspect_archive(token,asset,manifest_bytes,progress=None,restore_member=None,restore_sink=None):
    """Hash every member and the entire TAR, including trailing bytes, without staging."""
    assert_asset(get_asset(token,asset['id']),asset)
    manifest=json.loads(manifest_bytes);expected={r['path']:r for r in manifest['files']}
    if len(expected)!=len(manifest['files']):raise RuntimeError('Duplicate path in fixed manifest')
    reader=RangeHashReader(token,asset,progress)
    seen={};manifest_seen=False;started=time.time();restored=False
    try:
        with tarfile.open(fileobj=reader,mode='r|',bufsize=4*1024**2) as archive:
            for member in archive:
                name=member.name
                if not member.isfile() or name.startswith('/') or '..' in name.split('/') or name in seen:raise RuntimeError('Unsafe or duplicate archive member')
                source=archive.extractfile(member)
                if name=='PUBLICATION_MANIFEST.json':
                    if manifest_seen or member.size!=len(manifest_bytes):raise RuntimeError('Embedded manifest mismatch')
                    body=source.read()
                    if body!=manifest_bytes:raise RuntimeError('Embedded manifest differs from fixed uploaded manifest')
                    manifest_seen=True;continue
                row=expected.get(name)
                if row is None or member.size!=row['bytes']:raise RuntimeError('Unexpected archive member or size: '+name)
                h=hashlib.sha256();total=0
                for block in iter(lambda:source.read(4*1024**2),b''):
                    h.update(block);total+=len(block)
                    if name==restore_member:
                        if restore_sink is None:raise RuntimeError('Missing restoration sink')
                        restore_sink.write(block)
                value=h.hexdigest()
                if total!=row['bytes'] or value!=row['sha256']:raise RuntimeError('Member content SHA mismatch: '+name)
                seen[name]={'bytes':total,'sha256':value}
                if name==restore_member:restored=True
        while reader.read(4*1024**2):pass
    finally:reader.close()
    if not manifest_seen or set(seen)!=set(expected):raise RuntimeError('Incomplete archive member coverage')
    if reader.bytes!=asset['bytes'] or reader.hash.hexdigest()!=asset['sha256']:raise RuntimeError('Complete TAR byte count/SHA mismatch')
    if restore_member is not None and not restored:raise RuntimeError('Requested restore member not present')
    assert_asset(get_asset(token,asset['id']),asset)
    return {'asset_id':asset['id'],'asset_name':asset['name'],'archive_bytes':reader.bytes,'archive_sha256':reader.hash.hexdigest(),'fixed_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'member_count':len(seen),'members':seen,'all_member_sha256_verified':True,'whole_tar_sha256_verified':True,'metadata_verified_before_and_after':True,'elapsed_seconds':time.time()-started,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
