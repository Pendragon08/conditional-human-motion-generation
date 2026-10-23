"""Upload exact authorized artifacts directly; stdin credential stays in memory.

No installs, credential files, deletion, redirects, or training mutations. Only
the two existing reproduction release tags and explicit source directories are
accepted. A retry verifies an existing asset and never overwrites it.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote

ROOT = Path('/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022')
REPO = 'Pendragon08/conditional-human-motion-generation'
TAGS = {'paper-reproduction-progress-20261023', 'server-unused-cache-backup-20261023'}


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def run(a):
    payload = json.load(sys.stdin)
    credential = payload.pop('credential')
    state = {'status':'VERIFYING_INPUTS', 'credential_saved':False, 'deletes_files':False,
        'gpu_used':False, 'script_sha256':digest(Path(__file__)), 'assets':[]}

    def save():
        temp = a.receipt.with_suffix('.partial')
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(a.receipt)

    def api(path):
        c = http.client.HTTPSConnection('api.github.com', timeout=30)
        c.request('GET', path, headers={'Authorization':'Bearer '+credential,
            'User-Agent':'authorized-reproduction-backup', 'Accept':'application/vnd.github+json'})
        r = c.getresponse();data = r.read();status = r.status;c.close()
        if status != 200:
            raise RuntimeError('GitHub API status '+str(status))
        return json.loads(data)

    try:
        if payload['tag'] not in TAGS or a.receipt.exists():
            raise ValueError('Wrong release or existing execution receipt')
        if a.receipt.parent.resolve().parent != ROOT:
            raise ValueError('Receipt must be in a new immediate child of the authorized project')
        if api('/user')['login'].lower() != 'pendragon08':
            raise ValueError('GitHub account differs from authorized repository owner')
        info = api('/repos/'+REPO+'/releases/tags/'+payload['tag'])
        if info['html_url'] != 'https://github.com/'+REPO+'/releases/tag/'+payload['tag']:
            raise ValueError('Unexpected release identity')
        state['release_url'] = info['html_url'];save()
        for row in payload['files']:
            p = Path(row['path'])
            checkpoint = p.parent == ROOT/'training_fp64_sdpa_v4/checkpoints' and p.name in {
                'model000012000.pt','opt000012000.pt'}
            code_archive = p == ROOT/'storage_cleanup_code_packages_20261023_v2/inactive_source_packages_20261023_v2.tar.gz'
            if not (checkpoint or code_archive) or p.resolve() != p or p.is_symlink():
                raise ValueError('Artifact outside explicit whitelist')
            if checkpoint != (payload['tag'] == 'paper-reproduction-progress-20261023'):
                raise ValueError('Wrong release role')
            if row['bytes'] != p.stat().st_size or row['sha256'] != digest(p):
                raise ValueError('Original artifact digest/size differs')
            name = 'reactive_v4_'+p.name if checkpoint else p.name
            existing = [v for v in info['assets'] if v['name'] == name]
            if not existing:
                state.update(status='UPLOADING', current_name=name);save()
                c = http.client.HTTPSConnection('uploads.github.com', timeout=300)
                url = '/repos/'+REPO+'/releases/'+str(info['id'])+'/assets?name='+quote(name, safe='')
                c.putrequest('POST', url)
                for k,v in {'Authorization':'Bearer '+credential, 'User-Agent':'authorized-reproduction-backup',
                    'Content-Type':'application/octet-stream', 'Content-Length':str(row['bytes'])}.items():
                    c.putheader(k,v)
                c.endheaders()
                sent = 0
                with p.open('rb') as f:
                    for block in iter(lambda: f.read(1048576), b''):
                        c.send(block);sent += len(block)
                        if sent % (64*2**20) == 0:
                            print(json.dumps({'name':name,'sent_bytes':sent}), flush=True)
                r = c.getresponse();body = r.read();status = r.status;c.close()
                if status != 201:
                    raise RuntimeError('GitHub upload status '+str(status))
                print(json.dumps({'name':name,'upload_response':status}), flush=True)
            info = api('/repos/'+REPO+'/releases/tags/'+payload['tag'])
            matches = [v for v in info['assets'] if v['name'] == name]
            if len(matches) != 1 or matches[0]['size'] != row['bytes'] or matches[0].get('digest') != 'sha256:'+row['sha256']:
                raise ValueError('GitHub asset size/digest mismatch; never overwrite')
            asset = matches[0]
            state['assets'].append({'name':name,'source_path':str(p),'bytes':row['bytes'],
                'sha256':row['sha256'],'digest':asset['digest'],'url':asset['browser_download_url']})
            save()
        state['status'] = 'SERVER_AND_GITHUB_SHA256_VERIFIED';save()
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc));save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,required=True)
    run(p.parse_args())
