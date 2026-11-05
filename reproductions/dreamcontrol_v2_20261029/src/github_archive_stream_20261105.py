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

def download_response(token,asset_id):
    url=f'https://api.github.com/repos/{REPO}/releases/assets/{int(asset_id)}?readback_nonce={time.time_ns()}'
    for hop in range(5):
        u=urlsplit(url)
        if u.scheme!='https' or not u.hostname:raise RuntimeError('Unsafe archive redirect')
        allowed=u.hostname in {'api.github.com','github.com','release-assets.githubusercontent.com','objects.githubusercontent.com','github-releases.githubusercontent.com'} or u.hostname.endswith('.s3.amazonaws.com')
        if not allowed:raise RuntimeError('Unexpected archive redirect host')
        headers={'Accept':'application/octet-stream','User-Agent':'DreamControl-archive-verifier','Cache-Control':'no-cache'}
        if u.hostname=='api.github.com':headers.update({'Authorization':'Bearer '+token,'X-GitHub-Api-Version':'2022-11-28'})
        c=http.client.HTTPSConnection(u.hostname,timeout=120)
        c.request('GET',u.path+('?' +u.query if u.query else ''),headers=headers)
        r=c.getresponse()
        if r.status in [301,302,303,307,308]:
            url=r.getheader('Location');r.read();c.close();continue
        if r.status!=200:
            code=r.status;r.read();c.close();raise RuntimeError('Archive download HTTP '+str(code))
        if 'json' in (r.getheader('Content-Type') or ''):
            r.read();c.close();raise RuntimeError('Archive endpoint returned JSON instead of bytes')
        return c,r
    raise RuntimeError('Too many archive redirects')

class HashReader:
    def __init__(self,response,expected_bytes,progress):
        self.response=response;self.expected_bytes=expected_bytes;self.progress=progress;self.bytes=0;self.hash=hashlib.sha256();self.last=time.monotonic()
    def read(self,n=-1):
        if n<0:n=4*1024**2
        b=self.response.read(n);self.bytes+=len(b);self.hash.update(b)
        if self.bytes>self.expected_bytes:raise RuntimeError('Archive exceeds bound byte count')
        if self.progress and time.monotonic()-self.last>=30:
            self.progress(self.bytes);self.last=time.monotonic()
        return b

def inspect_archive(token,asset,manifest_bytes,progress=None,restore_member=None,restore_sink=None):
    """Hash every member and the entire TAR, including trailing bytes, without staging."""
    assert_asset(get_asset(token,asset['id']),asset)
    manifest=json.loads(manifest_bytes);expected={r['path']:r for r in manifest['files']}
    if len(expected)!=len(manifest['files']):raise RuntimeError('Duplicate path in fixed manifest')
    conn,response=download_response(token,asset['id']);reader=HashReader(response,asset['bytes'],progress)
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
    finally:conn.close()
    if not manifest_seen or set(seen)!=set(expected):raise RuntimeError('Incomplete archive member coverage')
    if reader.bytes!=asset['bytes'] or reader.hash.hexdigest()!=asset['sha256']:raise RuntimeError('Complete TAR byte count/SHA mismatch')
    if restore_member is not None and not restored:raise RuntimeError('Requested restore member not present')
    assert_asset(get_asset(token,asset['id']),asset)
    return {'asset_id':asset['id'],'asset_name':asset['name'],'archive_bytes':reader.bytes,'archive_sha256':reader.hash.hexdigest(),'fixed_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'member_count':len(seen),'members':seen,'all_member_sha256_verified':True,'whole_tar_sha256_verified':True,'metadata_verified_before_and_after':True,'elapsed_seconds':time.time()-started,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
