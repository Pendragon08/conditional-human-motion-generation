"""Relay one authorized release part through memory; no credential or payload files."""
import http.client
import json
import sys
import urllib.parse

request = json.loads(sys.stdin.buffer.readline())
remaining = int(request['bytes'])
if not 0 < remaining <= 32 * 1024 * 1024:
    raise ValueError('Part outside allowed size')
if request['repo'] != 'Pendragon08/conditional-human-motion-generation' or int(request['release']) != 434987063:
    raise ValueError('Unexpected destination')
connection = http.client.HTTPSConnection('uploads.github.com', timeout=180)
route = '/repos/{}/releases/{}/assets?name={}'.format(request['repo'], request['release'], urllib.parse.quote(request['name'], safe=''))
connection.putrequest('POST', route)
for key, value in {'Authorization': 'Bearer ' + request.pop('token'), 'User-Agent': 'DreamControl-publication-relay', 'Content-Type': 'application/octet-stream', 'Content-Length': str(remaining), 'Accept': 'application/vnd.github+json'}.items():
    connection.putheader(key, value)
connection.endheaders()
while remaining:
    block = sys.stdin.buffer.read(min(1024 * 1024, remaining))
    if not block:
        raise EOFError('Truncated input')
    connection.send(block)
    remaining -= len(block)
response = connection.getresponse()
body = response.read()
connection.close()
if response.status != 201:
    print(json.dumps({'error': 'GitHub HTTP {}: {}'.format(response.status, body.decode('utf-8', errors='replace')[:400])}))
else:
    print(body.decode('utf-8'))
