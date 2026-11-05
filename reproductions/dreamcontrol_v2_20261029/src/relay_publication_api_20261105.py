"""Authorized repository API traffic originates on the server. Credentials remain in memory."""
import json
import sys
import urllib.request
import urllib.error

data = json.load(sys.stdin)
endpoint = data['endpoint']
if not endpoint.startswith('repos/Pendragon08/conditional-human-motion-generation/'):
    raise ValueError('Unexpected repository')
if any(x in endpoint for x in ['..', '\\', '\n', '\r']):
    raise ValueError('Invalid endpoint')
payload = json.dumps(data['body']).encode() if 'body' in data else None
req = urllib.request.Request('https://api.github.com/' + endpoint, data=payload, method=data.get('method', 'GET'), headers={'Authorization': 'Bearer ' + data.pop('token'), 'User-Agent': 'DreamControl-authorized-publication', 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=90) as response:
        sys.stdout.buffer.write(response.read())
except urllib.error.HTTPError as error:
    sys.stderr.write('GitHub HTTP {}: {}'.format(error.code, error.read().decode('utf-8', errors='replace')[:1000]))
    raise SystemExit(1)
