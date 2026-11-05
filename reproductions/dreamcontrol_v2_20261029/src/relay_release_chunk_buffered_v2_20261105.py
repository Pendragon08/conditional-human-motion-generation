"""Buffer a bounded part in server memory before starting GitHub's upload timer."""
import http.client
import json
import sys
import urllib.parse

request = json.loads(sys.stdin.buffer.readline())
size = int(request['bytes'])
if not 0 < size <= 32 * 1024 * 1024:
    raise ValueError('Part outside allowed size')
if request['repo'] != 'Pendragon08/conditional-human-motion-generation' or int(request['release']) != 434987063:
    raise ValueError('Unexpected destination')
payload = sys.stdin.buffer.read(size)
if len(payload) != size:
    raise EOFError('Incomplete ZeroTier transfer; no GitHub upload started')
connection = http.client.HTTPSConnection('uploads.github.com', timeout=180)
route = '/repos/{}/releases/{}/assets?name={}'.format(request['repo'], request['release'], urllib.parse.quote(request['name'], safe=''))
connection.putrequest('POST', route)
for key, value in {'Authorization': 'Bearer ' + request.pop('token'), 'User-Agent': 'DreamControl-buffered-publication-relay', 'Content-Type': 'application/octet-stream', 'Content-Length': str(size), 'Accept': 'application/vnd.github+json'}.items():
    connection.putheader(key, value)
connection.endheaders()
for offset in range(0, size, 1024 * 1024):
    connection.send(memoryview(payload)[offset:offset + 1024 * 1024])
response = connection.getresponse()
body = response.read()
connection.close()
if response.status != 201:
    print(json.dumps({'error': 'GitHub HTTP {}: {}'.format(response.status, body.decode('utf-8', errors='replace')[:400])}))
else:
    print(body.decode('utf-8'))
