"""Prepare a negative integration case without changing the valid config."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE/'config.json').read_text())
cfg.pop('caption_evidence_root')
cfg.pop('caption_evidence_manifest')
with (HERE/'missing_evidence_config.json').open('x') as f:
    json.dump(cfg,f,indent=2);f.write('\n')
