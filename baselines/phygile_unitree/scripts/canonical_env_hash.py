"""Canonical JSON digest used by the compute ledger (ASCII escaping explicit)."""
from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('spec',type=Path);a=p.parse_args()
value=json.loads(a.spec.read_text(encoding='utf-8'))
canonical=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf-8')
h=hashlib.sha256(canonical).hexdigest()
print(json.dumps({'sha256':h,'short':h[:8],'canonicalization':'JSON parse; recursively sorted keys; compact separators; ensure_ascii=True; UTF-8 bytes; no newline','canonical_bytes':len(canonical)}))
