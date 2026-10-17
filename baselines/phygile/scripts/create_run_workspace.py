"""Create a clean run directory without copying archived result directories."""
from pathlib import Path
import argparse, shutil, os
p=argparse.ArgumentParser();p.add_argument('--destination',required=True,type=Path);a=p.parse_args()
src=Path(__file__).resolve().parents[1];dest=a.destination.resolve()
if dest.exists():raise FileExistsError(dest)
if dest==src or src in dest.parents:raise ValueError('Choose a directory outside the archived baseline folder')
if os.name=='nt':
 src=Path('\\\\?\\'+str(src));dest=Path('\\\\?\\'+str(dest))
dest.mkdir(parents=True)
for name in ['upstream','scripts','configs','docs','patches','reimplementation']:
 shutil.copytree(src/name,dest/name,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['README.md','upstream_manifest.json']:
 shutil.copy2(src/name,dest/name)
(dest/'results').mkdir()
print('Created clean run workspace:',dest)
