import fcntl,json,os,pathlib,signal
base=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_supermimic29_20261103/runs/pick500_8912_72000_parallel_v2')
signal.signal(signal.SIGIO,lambda *_:None)
for name in ['resume_early_96.pt','resume_failure_3424.pt']:
 p=base/name;before=p.stat();fd=os.open(p,os.O_RDWR)
 try:
  fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK)
  assert fcntl.fcntl(fd,fcntl.F_GETLEASE)==fcntl.F_WRLCK
  print(json.dumps({'file':str(p),'kernel_exclusive_file_lease_granted':True,'bytes':before.st_size,'inode':before.st_ino}))
  fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK)
 finally:os.close(fd)
 after=p.stat();assert before.st_mtime_ns==after.st_mtime_ns and before.st_size==after.st_size and before.st_ino==after.st_ino
