"""Remove only regenerable CPython bytecode under this project; preserve sources and results."""
import json,os,shutil
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029').resolve(strict=True)
REPORT=ROOT/'python_bytecode_cleanup_20261103.json'
def main():
 if REPORT.exists():raise FileExistsError(REPORT)
 before=shutil.disk_usage(ROOT).free;items=[];errors=[]
 for root,dirs,files in os.walk(ROOT,followlinks=False):
  dirs[:]=[d for d in dirs if not (Path(root)/d).is_symlink()]
  if Path(root).name!='__pycache__':continue
  for name in files:
   p=Path(root)/name
   if not name.endswith('.pyc') or p.is_symlink() or not p.is_file():continue
   actual=p.resolve(strict=True)
   if os.path.commonpath((str(ROOT),str(actual)))!=str(ROOT) or actual.parent.name!='__pycache__':raise ValueError('unsafe path')
   try:
    size=actual.stat().st_size;actual.unlink();items.append({'path':str(actual.relative_to(ROOT)),'bytes':size})
   except OSError as e:errors.append({'path':str(actual.relative_to(ROOT)),'error':str(e)})
 result={'project_root':str(ROOT),'type':'regenerable_python_bytecode_only','removed_files':len(items),
         'removed_bytes':sum(x['bytes'] for x in items),'free_bytes_before':before,'free_bytes_after':shutil.disk_usage(ROOT).free,
         'removed':items,'errors':errors}
 with REPORT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False)
 print(json.dumps({k:v for k,v in result.items() if k!='removed'}))
if __name__=='__main__':main()
