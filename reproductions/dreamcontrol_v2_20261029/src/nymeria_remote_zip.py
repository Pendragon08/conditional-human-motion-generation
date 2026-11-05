"""Read an authorized Nymeria zip by HTTP ranges without fetching its videos.

Signed URLs are held only in memory and never printed. A successful extracted
member is checked by Python's zipfile CRC-32, not the enclosing archive SHA-1.
The caller must retain that distinction in provenance.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile


class RemoteZipReader(io.RawIOBase):
    def __init__(self, url: str, size: int, block_size: int = 262144):
        if size <= 0 or block_size <= 0:
            raise ValueError('invalid remote archive size or block size')
        self.url = url
        self.size = size
        self.block_size = block_size
        self.position = 0
        self.cache_start = -1
        self.cache = b''
        self.bytes_transferred = 0
        self.requests = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        new = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset if whence == io.SEEK_END else -1
        if new < 0:
            raise ValueError('invalid archive offset')
        self.position = new
        return new

    def _fill(self):
        start = (self.position // self.block_size) * self.block_size
        end = min(self.size, start + self.block_size) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}',
                                                         'Accept-Encoding': 'identity',
                                                         'User-Agent': 'DreamControl-v2 research data preparation'})
        with urllib.request.urlopen(req, timeout=60) as response:
            content_range = response.headers.get('Content-Range', '')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
            if response.status != 206 or not match or tuple(map(int, match.groups())) != (start, end, self.size):
                raise ValueError('server did not honor the requested byte range')
            data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise ValueError('short or oversized range response')
        self.requests += 1
        self.bytes_transferred += len(data)
        self.cache_start, self.cache = start, data

    def read(self, size=-1):
        if self.position >= self.size:
            return b''
        remaining = self.size - self.position if size is None or size < 0 else min(size, self.size - self.position)
        chunks = []
        while remaining:
            if not self.cache_start <= self.position < self.cache_start + len(self.cache):
                self._fill()
            off = self.position - self.cache_start
            part = self.cache[off:off + remaining]
            if not part:
                raise ValueError('empty range cache')
            chunks.append(part)
            self.position += len(part)
            remaining -= len(part)
        return b''.join(chunks)


def list_members(url: str, size: int) -> tuple[list[dict], dict]:
    stream = RemoteZipReader(url, size)
    with zipfile.ZipFile(stream) as z:
        files = [{'name': item.filename, 'compressed_bytes': item.compress_size,
                  'uncompressed_bytes': item.file_size, 'crc32': f'{item.CRC:08x}'}
                 for item in z.infolist() if not item.is_dir()]
    return files, {'bytes_transferred': stream.bytes_transferred, 'requests': stream.requests}
