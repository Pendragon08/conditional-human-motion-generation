"""Restore privately archived, unchanged browser JS compilation caches only."""
import argparse,ctypes,hashlib,json,os,pathlib,re,shutil,subprocess,zipfile
ROOTS={"chrome-default":pathlib.Path("C:/Users/Pende/AppData/Local/Google/Chrome/User Data/Default/Code Cache/js"),"edge-default":pathlib.Path("C:/Users/Pende/AppData/Local/Microsoft/Edge/User Data/Default/Code Cache/js")}
SIDS=re.findall(r"S-1-[0-9-]+",subprocess.check_output(["whoami","/user","/fo","csv","/nh"],text=True,encoding="utf-8"));assert len(SIDS)==1
SID=SIDS[0]
adv=ctypes.WinDLL("advapi32",use_last_error=True);krn=ctypes.WinDLL("kernel32",use_last_error=True)
adv.GetNamedSecurityInfoW.argtypes=[ctypes.c_wchar_p,ctypes.c_int,ctypes.c_ulong,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]
adv.GetNamedSecurityInfoW.restype=ctypes.c_ulong
adv.ConvertSidToStringSidW.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)];adv.ConvertSidToStringSidW.restype=ctypes.c_int
krn.LocalFree.argtypes=[ctypes.c_void_p];krn.LocalFree.restype=ctypes.c_void_p

def owned(p):
 owner=ctypes.c_void_p();desc=ctypes.c_void_p();s=ctypes.c_void_p()
 err=adv.GetNamedSecurityInfoW(str(p),1,1,ctypes.byref(owner),None,None,None,ctypes.byref(desc));assert err==0,("owner unavailable",err)
 try:
  assert adv.ConvertSidToStringSidW(owner,ctypes.byref(s))
  return ctypes.wstring_at(s)==SID
 finally:
  if s:krn.LocalFree(s)
  if desc:krn.LocalFree(desc)

def guard(root,p):
 root,p=pathlib.Path(root),pathlib.Path(p);p.relative_to(root)
 assert root.resolve()==root and p.resolve()==p
 for q in [root]+list(reversed(p.relative_to(root).parents)):
  q=root/q if not q.is_absolute() else q
  if q.exists():assert not q.is_symlink() and not(q.stat().st_file_attributes&0x400) and owned(q)
 assert owned(p) and not(p.stat().st_file_attributes&0x400) and not p.is_symlink()
 return True

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()

def fingerprint(p):
 s=pathlib.Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink]

def target(row):
 root=ROOTS[row["root_id"]];rel=pathlib.PurePosixPath(row["relative_path"])
 assert not rel.is_absolute() and ".." not in rel.parts and rel.parts
 assert rel.name not in ("index","the-real-index") and "index-dir" not in rel.parts
 p=root.joinpath(*rel.parts);assert p.resolve()==p;p.relative_to(root);return root,p

def restore(folder):
 folder=pathlib.Path(folder).resolve();m=json.loads((folder/"browser_code_cache_manifest.private.json").read_text(encoding="utf-8"));a=folder/"browser_code_cache.private.zip"
 assert sha(a)==m["archive_sha256"]
 assert shutil.disk_usage("C:/").free>=m["source_bytes"]+2*1024**3,"Restore requires payload plus 2GiB free"
 count=0
 with zipfile.ZipFile(a) as z:
  assert z.testzip() is None
  for row in m["files"]:
   root,p=target(row);data=z.read(row["member"]);assert len(data)==row["bytes"] and hashlib.sha256(data).hexdigest()==row["sha256"]
   if p.exists():guard(root,p);assert sha(p)==row["sha256"],"Preserve regenerated cache; do not overwrite";continue
   assert p.parent.exists();guard(root,p.parent)
   with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
   guard(root,p);assert sha(p)==row["sha256"];count+=1
 return {"restored_files":count,"archive_unchanged":True}
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--private-folder",required=True);x=a.parse_args();print(json.dumps(restore(x.private_folder)))
