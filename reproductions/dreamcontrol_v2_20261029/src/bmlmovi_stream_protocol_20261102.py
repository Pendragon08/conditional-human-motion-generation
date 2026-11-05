"""Bounded framed transport for verified BMLmovi source members."""
import json
import struct

MAX_HEADER=64*1024
MAX_PAYLOAD=64*1024*1024


def read_exact(stream,n):
    if n<0 or n>MAX_PAYLOAD:raise ValueError('invalid frame length')
    chunks=[]
    while n:
        chunk=stream.read(min(n,1024*1024))
        if not chunk:raise EOFError('truncated framed stream')
        chunks.append(chunk);n-=len(chunk)
    return b''.join(chunks)


def write_frame(stream,header,payload):
    if not isinstance(payload,bytes) or not 0<len(payload)<=MAX_PAYLOAD:raise ValueError('invalid payload')
    encoded=json.dumps(header,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf8')
    if not 0<len(encoded)<=MAX_HEADER:raise ValueError('invalid header')
    stream.write(struct.pack('!I',len(encoded)));stream.write(encoded)
    stream.write(struct.pack('!Q',len(payload)));stream.write(payload);stream.flush()


def end_stream(stream):
    stream.write(struct.pack('!I',0));stream.flush()


def read_frame(stream):
    size=struct.unpack('!I',read_exact(stream,4))[0]
    if size==0:return None
    if size>MAX_HEADER:raise ValueError('header too large')
    header=json.loads(read_exact(stream,size).decode('utf8'))
    if not isinstance(header,dict):raise ValueError('header must be object')
    size=struct.unpack('!Q',read_exact(stream,8))[0]
    if not 0<size<=MAX_PAYLOAD:raise ValueError('payload size invalid')
    return header,read_exact(stream,size)
