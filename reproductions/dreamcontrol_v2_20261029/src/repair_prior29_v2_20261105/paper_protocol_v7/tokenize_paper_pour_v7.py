"""Produce the repaired Pick caption tokens with the exact legacy tokenizer."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request


ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE = ROOT / 'repair_prior29_v2_20261105'
MODEL_ASSET = 195734380
MODEL_BYTES = 12806118
WHEEL = 'en_core_web_sm-3.8.0-py3-none-any.whl'
TOKENIZE = r"""
import hashlib,json,pathlib,spacy,sys
source,target,receipt=map(pathlib.Path,sys.argv[1:4])
old=json.loads(source.read_text())
nlp=spacy.load('en_core_web_sm')
assert spacy.__version__ == old['tokenizer']['spacy'] == '3.8.11'
assert nlp.meta['version'] == old['tokenizer']['model'] == '3.8.0'
def tokens(text):
    return [f"{t.lemma_ if t.pos_ in ('NOUN','VERB') and t.text != 'left' else t.text}/{t.pos_}"
            for t in nlp(text.replace('-','')) if t.text.isalpha()]
mismatch=[caption for caption,expected in old['tokens'].items() if tokens(caption)!=expected]
assert not mismatch, ('Legacy tokenizer regression', len(mismatch), mismatch[:3])
caption='a person picks an object with the right hand and performs a pouring action with the same hand'
assert caption not in old['tokens']
old['tokens'][caption]=tokens(caption)
with target.open('x') as f:json.dump(old,f,indent=2)
with receipt.open('x') as f:
    json.dump({'legacy_input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
               'new_output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
               'legacy_token_rows_matched':len(old['tokens'])-1,'tokenizer':old['tokenizer'],
               'model_github_asset':195734380,'model_wheel_sha256':sys.argv[4],
               'new_caption':caption,'new_tokens':old['tokens'][caption]},f,indent=2)
print(receipt.read_text())
"""


def main():
    legacy = STAGE / 'reports/caption_tokens_repaired_v1.json'
    output = STAGE / 'reports/caption_tokens_paper_v7.json'
    receipt = STAGE / 'reports/caption_tokens_paper_v7_receipt.json'
    assert not output.exists() and not receipt.exists()
    assert shutil.disk_usage('/dev/shm').free >= 3 * 1024**3
    with tempfile.TemporaryDirectory(prefix='dreamcontrol_tokenize_v2_', dir='/dev/shm') as private:
        scratch = Path(private)
        wheel = scratch / WHEEL
        url = f'https://api.github.com/repos/explosion/spacy-models/releases/assets/{MODEL_ASSET}?readback={time.time_ns()}'
        request = urllib.request.Request(url, headers={'Accept': 'application/octet-stream',
                                         'User-Agent': 'DreamControl-eval'})
        with urllib.request.urlopen(request, timeout=30) as response, wheel.open('xb') as stream:
            body = response.read(MODEL_BYTES + 1)
            assert len(body) == MODEL_BYTES and body.startswith(b'PK\x03\x04'), 'Unexpected wheel response'
            stream.write(body)
        assert hashlib.sha256(body).hexdigest() == '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85'
        print('Official wheel verified', hashlib.sha256(body).hexdigest(), flush=True)
        packages = scratch / 'packages'
        env = {**os.environ, 'TMPDIR': private, 'PYTHONDONTWRITEBYTECODE': '1',
               'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'}
        subprocess.run([sys.executable, '-B', '-m', 'pip', '--isolated', 'install',
                        '--disable-pip-version-check', '--no-cache-dir', '--only-binary', ':all:',
                        '--index-url', 'https://pypi.org/simple', '--timeout', '60', '--retries', '3',
                        '--target', str(packages), 'spacy==3.8.11', str(wheel)],
                       env=env, check=True, timeout=2400)
        env['PYTHONPATH'] = str(packages)
        subprocess.run([sys.executable, '-B', '-c', TOKENIZE, str(legacy), str(output), str(receipt),
                        hashlib.sha256(body).hexdigest()], env=env, check=True, timeout=120)


if __name__ == '__main__':
    main()
