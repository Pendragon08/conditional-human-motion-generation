"""Import the exact archived 500-update parent into a clean continuation workspace."""
from pathlib import Path
import argparse
import hashlib
import shutil

EXPECTED = 'afad21ce7192dc4ec10e04fd2bc7684e9278f901dbffcb053cf45b8b04bda162'
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=Path)
args = parser.parse_args()
source = args.checkpoint.resolve(strict=True)
if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED:
    raise ValueError('Expected the published 500-update seed-42 parent; hash differs')
root = Path(__file__).resolve().parents[1]
destination = root / 'logs/rsl_rl/phygile_moe_pilot/2026-10-17_09-16-39_train20_nostate_seed42/model_499.pt'
if destination.exists():
    if hashlib.sha256(destination.read_bytes()).hexdigest() != EXPECTED:
        raise FileExistsError('Refusing to replace a different parent checkpoint')
else:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
assert hashlib.sha256(destination.read_bytes()).hexdigest() == EXPECTED
print('Verified parent checkpoint:', destination)
