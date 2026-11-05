import json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
src=root/'data/bmlmovi_full_stream_queue_20261102.json'
out=root/'data/bmlmovi_stream_smoke_queue_20261102.json'
d=json.loads(src.read_text(encoding='utf8'))
assert d['training_ready'] is False and len(d['members'])==1781
with out.open('x',encoding='utf8') as f:json.dump({'members':d['members'][:2],'training_ready':False},f,ensure_ascii=False,indent=2)
