"""Reconstruct this server-specific overlay from the declared spec; never replace an env.

The recorded initial build used equivalent bootstrap operations. This helper was
added afterwards and has not itself been executed against the existing env.
Inherited environments and cached DistilBERT are prerequisites, not downloads.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--spec', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/env_spec.json')
a = p.parse_args()
s = json.loads(a.spec.read_text())
target = Path(s['environment_directory'])
if target.exists():
    raise SystemExit('Environment already exists; refusing to replace it.')
for entry in [s['python_executable'], *s['inherit_site_packages'], s['weight_dirs']['text_encoder']['path']]:
    if not Path(entry).exists():
        raise SystemExit('Required inherited resource absent: ' + entry)
subprocess.run([s['python_executable'], '-m', 'venv', str(target)], check=True)
python = target / 'bin/python'
site = subprocess.check_output([str(python), '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
(Path(site) / 'reactivebfm_existing_packages.pth').write_text('\n'.join(s['inherit_site_packages']) + '\n')
env = os.environ.copy()
temp = target.parent / 'tmp'
temp.mkdir(exist_ok=True)
env.update({'TMPDIR': str(temp), 'PIP_NO_CACHE_DIR': '1'})
for phase in s['pip_phases']:
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', *phase], env=env, check=True)
print('Overlay built. Run a fresh seeded witness and document-following check before trusting it.')
